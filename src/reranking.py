"""
Candidate Reranking Script Using Euclidean, Hyperbolic, Hybrid, and ColBERT Models

This script performs reranking of candidate Human Phenotype Ontology (HPO) terms based on multiple embedding and similarity strategies:

- Hyperbolic distance-based reranking using HierarchyTransformer embeddings.
- Hybrid reranking combining cosine similarity and normalized hyperbolic distances.
- Weighted fallback reranking with thresholding.
- Cosine similarity reranking on both hyperbolic and euclidean embeddings.
- Reranking using an alternative SentenceTransformer model.
- Late interaction reranking using a custom ColBERT implementation.

Features:
- Supports batch processing of candidates for efficient scoring.
- Normalizes distances with configurable modes.
- Loads pretrained models and tokenizers.
- Processes input CSV files containing candidate spans and outputs reranked results with scores.
"""


from hyperrag.config import *
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from transformers import AutoTokenizer, AutoModel
from hierarchy_transformers import HierarchyTransformer
from utils import normalize_distances
from late_interaction_training import tokenize_query, tokenize_doc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Custom ColBERT implementation to avoid dependency issues
class ColBERT(torch.nn.Module):
    def __init__(self, base_model, dim=128, similarity_metric='cosine'):
        super().__init__()
        self.dim = dim
        self.similarity_metric = similarity_metric
        self.base_model = base_model
        self.linear = torch.nn.Linear(base_model.config.hidden_size, dim, bias=False)

    def forward(self, input_ids, attention_mask=None, is_query=False):
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state

        # Apply linear projection
        embeddings = self.linear(embeddings)

        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=2)

        # For queries, mask out the [CLS] token if needed
        if is_query:
            pass  # In the original ColBERT, there's special handling for queries

        return embeddings

    def score(self, q_reps, d_reps):
        # MaxSim late interaction scoring
        # q_reps: [batch_size, query_length, dim]
        # d_reps: [batch_size, doc_length, dim]

        # Calculate similarity matrix
        similarity = torch.bmm(q_reps, d_reps.transpose(1, 2))  # [batch_size, query_length, doc_length]

        # Max pooling over document dimension
        max_sim = similarity.max(dim=2)[0]  # [batch_size, query_length]

        # Sum/mean pooling over query terms
        scores = max_sim.mean(dim=1)  # [batch_size]

        return scores


ENABLE_BASE_RERANKING = True
ENABLE_COLBERT_RERANKING = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rerank_candidates(
        df: pd.DataFrame,
        euclidean_model: SentenceTransformer,
        hyperbolic_model: HierarchyTransformer,
        alternative_model: SentenceTransformer,
        alpha: float = 0.5,
        threshold: float = 0.5,
        normalization_mode: str = "minmax",
        global_max_distance: float = 1.0,
        top_k: int = 25,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> pd.DataFrame:
    """
    Rerank candidates using hyperbolic and hybrid strategies
    """
    # Create new columns for reranked results
    df['hyperbolic_reranking_terms'] = None
    df['hyperbolic_reranking'] = None
    df['hybrid_reranking_terms'] = None
    df['hybrid_reranking'] = None
    df['hybrid_reranking_scores'] = None
    df['weighted_fallback_reranking_terms'] = None
    df['weighted_fallback_reranking'] = None
    df['hyperbolic_cosine_reranking_terms'] = None
    df['hyperbolic_cosine_reranking'] = None
    df['euclidean_cosine_reranking_terms'] = None
    df['euclidean_cosine_reranking'] = None
    df['alternative_reranking_terms'] = None
    df['alternative_reranking'] = None
    df['alternative_scores'] = None

    hyperbolic_model = hyperbolic_model.to(device)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        span = row['span']
        if pd.isna(span) or str(span).strip() == "":
            df.at[idx, 'hyperbolic_reranking_terms'] = ""
            df.at[idx, 'hyperbolic_reranking'] = ""
            df.at[idx, 'hybrid_reranking_terms'] = ""
            df.at[idx, 'hybrid_reranking'] = ""
            df.at[idx, 'hybrid_reranking_scores'] = ""
            df.at[idx, 'weighted_fallback_reranking_terms'] = ""
            df.at[idx, 'weighted_fallback_reranking'] = ""
            df.at[idx, 'hyperbolic_cosine_reranking_terms'] = ""
            df.at[idx, 'hyperbolic_cosine_reranking'] = ""
            df.at[idx, 'euclidean_cosine_reranking_terms'] = ""
            df.at[idx, 'euclidean_cosine_reranking'] = ""
            df.at[idx, 'alternative_reranking_terms'] = ""
            df.at[idx, 'alternative_reranking'] = ""
            df.at[idx, 'alternative_scores'] = ""
            continue

        # Get euclidean candidates and their IDs
        candidates_terms = row[f'rag_candidates_euclidean_terms_top_{top_k}'].split('||')
        candidates_ids = row[f'rag_candidates_euclidean_top_{top_k}'].split(',')

        # Create terms-to-ids mapping
        terms_to_ids = dict(zip(candidates_terms, candidates_ids))

        # Compute embeddings
        # Euclidean embeddings
        span_euc_emb = euclidean_model.encode([span], convert_to_numpy=True)
        candidates_euc_emb = euclidean_model.encode(candidates_terms, convert_to_numpy=True)

        # Hyperbolic embeddings - keep as tensors
        with torch.no_grad():
            span_hyp_emb = hyperbolic_model.encode([span], convert_to_tensor=True).to(device)
            candidates_hyp_emb = hyperbolic_model.encode(candidates_terms, convert_to_tensor=True).to(device)

        # Compute hyperbolic distances for euclidean candidates
        hyperbolic_distances = []
        for cand_emb in candidates_hyp_emb:
            dist = hyperbolic_model.manifold.dist(span_hyp_emb, cand_emb.unsqueeze(0))
            hyperbolic_distances.append(dist.cpu().item())
        hyperbolic_distances = np.array(hyperbolic_distances)

        # 1. Rerank based on hyperbolic distances
        hyp_indices = np.argsort(hyperbolic_distances)
        hyperbolic_reranked_terms = [candidates_terms[i] for i in hyp_indices]
        hyperbolic_reranked_ids = [terms_to_ids[term] for term in hyperbolic_reranked_terms]

        df.at[idx, 'hyperbolic_reranking_terms'] = '||'.join(hyperbolic_reranked_terms)
        df.at[idx, 'hyperbolic_reranking'] = ','.join(hyperbolic_reranked_ids)

        # 2. Hybrid reranking
        # Compute cosine similarities
        cosine_similarities = cosine_similarity(span_euc_emb, candidates_euc_emb)[0]

        # Normalize hyperbolic distances
        norm_hyperbolic_distances = normalize_distances(
            hyperbolic_distances,
            normalization_mode,
            global_max_distance
        )

        # Compute hybrid scores
        hybrid_scores = alpha * cosine_similarities - (1 - alpha) * norm_hyperbolic_distances

        # Rerank based on hybrid scores
        hybrid_indices = np.argsort(-hybrid_scores)  # negative for descending order
        hybrid_reranked_terms = [candidates_terms[i] for i in hybrid_indices]
        hybrid_reranked_ids = [terms_to_ids[term] for term in hybrid_reranked_terms]
        hybrid_reranked_scores = [round(float(hybrid_scores[i]), 4) for i in hybrid_indices]

        df.at[idx, 'hybrid_reranking_terms'] = '||'.join(hybrid_reranked_terms)
        df.at[idx, 'hybrid_reranking'] = ','.join(hybrid_reranked_ids)
        df.at[idx, 'hybrid_reranking_scores'] = ','.join(map(str, hybrid_reranked_scores))

        # 3. Weighted reranking with fallback
        cosine_scores = cosine_similarities  # Reusing cosine scores

        # Appliquer le seuil et calculer les scores finaux
        final_scores = np.zeros_like(cosine_scores)
        for i, cosine_score in enumerate(cosine_scores):
            if cosine_score < threshold:
                final_scores[i] = alpha * cosine_score - (1 - alpha) * norm_hyperbolic_distances[i]
            else:
                final_scores[i] = cosine_score

        # Rerank based on final scores
        weighted_fallback_indices = np.argsort(-final_scores)  # negative for descending order
        weighted_fallback_reranked_terms = [candidates_terms[i] for i in weighted_fallback_indices]
        weighted_fallback_reranked_ids = [terms_to_ids[term] for term in weighted_fallback_reranked_terms]

        df.at[idx, 'weighted_fallback_reranking_terms'] = '||'.join(weighted_fallback_reranked_terms)
        df.at[idx, 'weighted_fallback_reranking'] = ','.join(weighted_fallback_reranked_ids)

        # 4. Hyperbolic cosine similarity reranking
        span_hyp_np = span_hyp_emb.cpu().numpy()
        candidates_hyp_np = candidates_hyp_emb.cpu().numpy()
        hyperbolic_cosine_similarities = cosine_similarity(span_hyp_np, candidates_hyp_np)[0]

        hyp_cos_indices = np.argsort(-hyperbolic_cosine_similarities)
        hyperbolic_cosine_reranked_terms = [candidates_terms[i] for i in hyp_cos_indices]
        hyperbolic_cosine_reranked_ids = [terms_to_ids[term] for term in hyperbolic_cosine_reranked_terms]

        df.at[idx, 'hyperbolic_cosine_reranking_terms'] = '||'.join(hyperbolic_cosine_reranked_terms)
        df.at[idx, 'hyperbolic_cosine_reranking'] = ','.join(hyperbolic_cosine_reranked_ids)

        # 5. Euclidean cosine reranking of hyperbolic candidates
        hyperbolic_candidates_terms = row[f'rag_candidates_hyperbolic_terms_top_{top_k}'].split('||')
        hyperbolic_candidates_ids = row[f'rag_candidates_hyperbolic_top_{top_k}'].split(',')
        hyp_terms_to_ids = dict(zip(hyperbolic_candidates_terms, hyperbolic_candidates_ids))

        # Compute euclidean embeddings for span and hyperbolic candidates
        span_euc_emb = euclidean_model.encode([span], convert_to_numpy=True)
        hyp_candidates_euc_emb = euclidean_model.encode(hyperbolic_candidates_terms, convert_to_numpy=True)

        # Compute cosine similarities
        euc_cosine_similarities = cosine_similarity(span_euc_emb, hyp_candidates_euc_emb)[0]

        # Rerank based on euclidean cosine similarities
        euc_cos_indices = np.argsort(-euc_cosine_similarities)
        euclidean_cosine_reranked_terms = [hyperbolic_candidates_terms[i] for i in euc_cos_indices]
        euclidean_cosine_reranked_ids = [hyp_terms_to_ids[term] for term in euclidean_cosine_reranked_terms]

        df.at[idx, 'euclidean_cosine_reranking_terms'] = '||'.join(euclidean_cosine_reranked_terms)
        df.at[idx, 'euclidean_cosine_reranking'] = ','.join(euclidean_cosine_reranked_ids)

        # ==== Step 6: Bi-encoder reranking ====
        # Encode span and candidate terms using alternative model (e.g. bi-encoder)
        span_emb_alt = alternative_model.encode([span], convert_to_tensor=True).to(device)
        candidates_emb_alt = alternative_model.encode(candidates_terms, convert_to_tensor=True).to(device)

        # Compute cosine similarities between span and each candidate
        cosine_scores = cos_sim(span_emb_alt, candidates_emb_alt)[0].cpu().numpy()

        # Sort candidates by similarity
        alt_indices = np.argsort(-cosine_scores)
        alt_model_reranked_terms = [candidates_terms[i] for i in alt_indices]
        alt_model_reranked_ids = [terms_to_ids[term] for term in alt_model_reranked_terms]
        alt_model_scores = [round(float(cosine_scores[i]), 4) for i in alt_indices]

        df.at[idx, 'alternative_reranking_terms'] = '||'.join(alt_model_reranked_terms)
        df.at[idx, 'alternative_reranking'] = ','.join(alt_model_reranked_ids)
        df.at[idx, 'alternative_scores'] = ','.join(map(str, alt_model_scores))

    return df


def rerank_colbert(df, model, tokenizer, top_k=10, device='cuda'):
    """
    Applique le reranking ColBERT sur le DataFrame.
    """
    df["ft-colbert_reranking_terms"] = None
    df["ft-colbert_reranking"] = None
    df["ft-colbert_scores"] = None

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            span = row["span"]
            if pd.isna(row[f"rag_candidates_euclidean_terms_top_{top_k}"]) or pd.isna(row[f"rag_candidates_euclidean_top_{top_k}"]):
                continue

            candidate_terms = row[f"rag_candidates_euclidean_terms_top_{top_k}"].split("||")
            candidate_ids = row[f"rag_candidates_euclidean_top_{top_k}"].split(",")

            reranked_terms, reranked_ids, reranked_scores = rerank_candidates_colbert(
                span, candidate_terms, candidate_ids, model, tokenizer, device)

            df.at[idx, "ft-colbert_reranking_terms"] = "||".join(reranked_terms)
            df.at[idx, "ft-colbert_reranking"] = ",".join(reranked_ids)
            df.at[idx, "ft-colbert_scores"] = ",".join([str(score) for score in reranked_scores])
        except Exception as e:
            print(f"Error on line {idx}: {e}")
            continue
    return df

def rerank_candidates_colbert(span, candidate_terms, candidate_ids, model, tokenizer):
    model.eval()
    with torch.no_grad():
        # Tokenize query with padding
        q_input_ids = tokenize_query(span, tokenizer).to(DEVICE)
        q_attention_mask = (q_input_ids != tokenizer.pad_token_id).long().to(DEVICE)

        # Get query representation
        q_reps = model(q_input_ids, q_attention_mask, is_query=True)

        scores = []
        # Process in batches to avoid OOM for large candidate sets
        batch_size = 16
        for i in range(0, len(candidate_terms), batch_size):
            batch_terms = candidate_terms[i:i + batch_size]

            # Tokenize documents in batch
            d_input_ids_list = []
            d_attention_mask_list = []

            for term in batch_terms:
                d_input_ids = tokenize_doc(term, tokenizer)
                d_attention_mask = (d_input_ids != tokenizer.pad_token_id).long()
                d_input_ids_list.append(d_input_ids.squeeze(0))
                d_attention_mask_list.append(d_attention_mask.squeeze(0))

            # Stack tensors
            d_input_ids_batch = torch.stack(d_input_ids_list).to(DEVICE)
            d_attention_mask_batch = torch.stack(d_attention_mask_list).to(DEVICE)

            # Get document representations
            d_reps = model(d_input_ids_batch, d_attention_mask_batch, is_query=False)

            # Calculate scores for batch
            for j in range(len(batch_terms)):
                # Extract single document representation
                d_rep = d_reps[j:j + 1]

                # Calculate score
                score = model.score(q_reps, d_rep).item()
                scores.append(score)

        # Sort candidates by score
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        sorted_terms = [candidate_terms[i] for i in sorted_indices]
        sorted_ids = [candidate_ids[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]

        return sorted_terms, sorted_ids, sorted_scores
    pass


def full_reranking_pipeline(
    df: pd.DataFrame,
    euclidean_model: SentenceTransformer = None,
    hyperbolic_model: HierarchyTransformer = None,
    alternative_model: SentenceTransformer = None,
    colbert_model=None,
    colbert_tokenizer=None,
    alpha: float = 0.5,
    threshold: float = 0.5,
    normalization_mode: str = "global",
    global_max_distance: float = 1.0,
    top_k: int = 25,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> pd.DataFrame:
    """
    Applique tous les rerankings souhaités sur le DataFrame.
    """
    # 1. Reranking classique (euclidien/hyperbolique/bi-encoder)
    if ENABLE_BASE_RERANKING:
        df = rerank_candidates(
            df,
            euclidean_model,
            hyperbolic_model,
            alternative_model,
            alpha=alpha,
            threshold=threshold,
            normalization_mode=normalization_mode,
            global_max_distance=global_max_distance,
            top_k=top_k,
            device=device
        )

    # 2. Reranking ColBERT (late interaction)
    if ENABLE_COLBERT_RERANKING:
        df = rerank_colbert(
            df,
            colbert_model,
            colbert_tokenizer,
            top_k=top_k,
            device=device
        )

    return df


def main():

    # Paths
    euclidean_model_path = models_path / euclidean_model
    hyperbolic_model_path = models_path / hyperbolic_model
    alternative_model_path = models_path / alternative_model
    late_interaction_model_path = late_interaction_model_ft
    input_csv_path = data_path / f"rag/output_rag_{target_dataset}_candidates_{top_k}_{mips}_{euc_model}_{hit_model}.csv"
    output_csv_path = data_path / f"rag/reranking/output_rag_{target_dataset}_candidates_{top_k}_reranked_{alpha}_{threshold}_{euc_model_rerank}_{hit_model_rerank}_{normalization_mode}.csv"

    # ==== Load models ====
    euclidean_mod = SentenceTransformer(euclidean_model_path)
    hyperbolic_mod = HierarchyTransformer.load_pretrained(hyperbolic_model_path)
    alternative_mod = SentenceTransformer(alternative_model_path)

    # Load the fine-tuned ColBERT model for inference
    print(f"Loading model from {late_interaction_model_path}...")
    base_model_inference = AutoModel.from_pretrained(late_interaction_model_base)
    colbert_tokenizer = AutoTokenizer.from_pretrained(late_interaction_model_base)
    inference_model = ColBERT(base_model_inference)

    # Try to load from checkpoint first, fall back to final model
    try:
        checkpoint_path = os.path.join(late_interaction_model_path, "colbert_hpo_checkpoint_epoch_2.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            inference_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from checkpoint (epoch 2)")
        else:
            inference_model.load_state_dict(
                torch.load(os.path.join(late_interaction_model_path, "colbert_hpo_model_triplet.pt"), map_location=DEVICE))
            print("Loaded model from final saved model")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Initializing model with random weights")

    inference_model.to(DEVICE)
    inference_model.eval()

    # ==== Load Data ====
    df = pd.read_csv(input_csv_path)

    # ==== Perform reranking ====
    df_reranked = full_reranking_pipeline(
        df,
        euclidean_mod,
        hyperbolic_mod,
        alternative_mod,
        inference_model,
        colbert_tokenizer,
        alpha=alpha,
        threshold=threshold,
        normalization_mode=normalization_mode,
        global_max_distance=hyperbolic_max_distance,
        top_k=top_k,
        device=DEVICE
    )

    # Save results
    df_reranked.to_csv(output_csv_path, index=False)
    print(f"Results written to {output_csv_path}")


if __name__ == "__main__":
    main()
