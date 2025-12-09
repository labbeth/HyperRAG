"""
Fine-tuning and Inference Script for ColBERT-based Late Interaction Model on HPO Data

This script provides functionality to:

- Fine-tune a ColBERT model using triplet loss on Human Phenotype Ontology (HPO) span and label triplets.
- Tokenize queries and documents with proper padding.
- Train the model with mixed precision and gradient accumulation.
- Validate the model during training.
- Save checkpoints and the final trained model.
- Perform inference to rerank candidate HPO terms for given spans using the fine-tuned ColBERT model.
- Apply reranking to an entire CSV file of candidate spans and save the enriched results.

Features:
- Custom ColBERT implementation to avoid external dependency issues.
- Efficient batch processing for inference.
- Graceful loading of model checkpoints with fallback.
- Configurable training parameters via external config.
"""


from hyperrag.config import *
import os
import json
import random
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn import TripletMarginLoss
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler


# Configuration
MODEL_NAME = "colbert-ir/colbertv2.0"
OUTPUT_DIR = data_path / "models/colbert_hpo_finetuned2"
BATCH_SIZE = li_batch_size
LEARNING_RATE = li_learning_rate
NUM_EPOCHS = li_num_epochs
MAX_QUERY_LENGTH = li_max_query_length
MAX_DOC_LENGTH = li_max_query_length
MARGIN = li_margin
GRADIENT_ACCUMULATION_STEPS = li_gradient_accumulation_steps
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MIXED_PRECISION = True if torch.cuda.is_available() else False  # Use mixed precision if on GPU

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Custom tokenization functions with proper padding
def tokenize_query(query, tokenizer, max_length=MAX_QUERY_LENGTH):
    """Tokenize a query with proper padding"""
    tokens = tokenizer(
        query,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",  # Add padding to ensure consistent length
        truncation=True,
        return_tensors="pt"
    )
    return tokens["input_ids"]


def tokenize_doc(doc, tokenizer, max_length=MAX_DOC_LENGTH):
    """Tokenize a document with proper padding"""
    tokens = tokenizer(
        doc,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",  # Add padding to ensure consistent length
        truncation=True,
        return_tensors="pt"
    )
    return tokens["input_ids"]


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


# Initialize ColBERT model
base_model = AutoModel.from_pretrained(late_interaction_model_base)
model = ColBERT(base_model)


# Load training data with triplet format
class HPOTripletDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data = []
        self.tokenizer = tokenizer
        self.all_hpo_labels = set()
        self.all_spans = []
        self.span_to_pos_hpo = {}

        # First pass: collect all HPO labels and spans
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                span = item["span"]
                hpo_label = item["hpo_label"]
                score = item["score"]

                # Only consider positive examples (score > threshold)
                if score > 0.8:  # Adjust threshold as needed
                    self.all_hpo_labels.add(hpo_label)
                    if span not in self.span_to_pos_hpo:
                        self.span_to_pos_hpo[span] = []
                        self.all_spans.append(span)
                    self.span_to_pos_hpo[span].append(hpo_label)

                self.data.append(item)

        self.all_hpo_labels = list(self.all_hpo_labels)
        print(f"Loaded {len(self.all_spans)} unique spans and {len(self.all_hpo_labels)} unique HPO labels")

    def __len__(self):
        return len(self.all_spans)

    def __getitem__(self, idx):
        # Get anchor (query)
        span = self.all_spans[idx]

        # Get positive example
        pos_hpo = random.choice(self.span_to_pos_hpo[span])

        # Select a random negative example (different from positive)
        while True:
            neg_idx = random.randint(0, len(self.all_hpo_labels) - 1)
            neg_hpo = self.all_hpo_labels[neg_idx]
            if neg_hpo not in self.span_to_pos_hpo[span]:
                break

        # Tokenize with padding
        q_input_ids = tokenize_query(span, self.tokenizer)
        q_attention_mask = (q_input_ids != self.tokenizer.pad_token_id).long()

        pos_input_ids = tokenize_doc(pos_hpo, self.tokenizer)
        pos_attention_mask = (pos_input_ids != self.tokenizer.pad_token_id).long()

        neg_input_ids = tokenize_doc(neg_hpo, self.tokenizer)
        neg_attention_mask = (neg_input_ids != self.tokenizer.pad_token_id).long()

        return {
            "q_input_ids": q_input_ids.squeeze(0),
            "q_attention_mask": q_attention_mask.squeeze(0),
            "pos_input_ids": pos_input_ids.squeeze(0),
            "pos_attention_mask": pos_attention_mask.squeeze(0),
            "neg_input_ids": neg_input_ids.squeeze(0),
            "neg_attention_mask": neg_attention_mask.squeeze(0),
        }


# Create dataset and dataloader
train_dataset = HPOTripletDataset(data_path / "training_pairs.jsonl", tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Prepare optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_dataloader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

# Initialize mixed precision scaler if using mixed precision
scaler = GradScaler() if USE_MIXED_PRECISION else None


# Function to run validation on a subset of data
def run_validation(model, dataset, device, num_samples=100):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]

            # Move to device
            q_input_ids = sample["q_input_ids"].unsqueeze(0).to(device)
            q_attention_mask = sample["q_attention_mask"].unsqueeze(0).to(device)
            pos_input_ids = sample["pos_input_ids"].unsqueeze(0).to(device)
            pos_attention_mask = sample["pos_attention_mask"].unsqueeze(0).to(device)
            neg_input_ids = sample["neg_input_ids"].unsqueeze(0).to(device)
            neg_attention_mask = sample["neg_attention_mask"].unsqueeze(0).to(device)

            # Get representations
            q_reps = model(q_input_ids, q_attention_mask, is_query=True)
            pos_reps = model(pos_input_ids, pos_attention_mask, is_query=False)
            neg_reps = model(neg_input_ids, neg_attention_mask, is_query=False)

            # Calculate scores
            pos_score = model.score(q_reps, pos_reps).item()
            neg_score = model.score(q_reps, neg_reps).item()

            # Check if positive scored higher than negative
            if pos_score > neg_score:
                correct += 1
            total += 1

    accuracy = correct / total
    print(f"Validation accuracy: {accuracy:.4f} ({correct}/{total})")
    model.train()
    return accuracy


# Training loop with triplet loss
model.to(DEVICE)
model.train()

print(f"Starting training on {DEVICE} with triplet loss...")
print(f"Using mixed precision: {USE_MIXED_PRECISION}")
print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")

for epoch in range(NUM_EPOCHS):
    print(f"Starting epoch {epoch + 1}/{NUM_EPOCHS}")
    total_loss = 0
    optimizer.zero_grad()  # Zero gradients at the beginning of epoch

    for batch_idx, batch in enumerate(train_dataloader):
        # Get batch data
        q_input_ids = batch["q_input_ids"].to(DEVICE)
        q_attention_mask = batch["q_attention_mask"].to(DEVICE)
        pos_input_ids = batch["pos_input_ids"].to(DEVICE)
        pos_attention_mask = batch["pos_attention_mask"].to(DEVICE)
        neg_input_ids = batch["neg_input_ids"].to(DEVICE)
        neg_attention_mask = batch["neg_attention_mask"].to(DEVICE)

        # Forward pass with mixed precision if enabled
        if USE_MIXED_PRECISION:
            with autocast():
                # Forward pass
                q_reps = model(q_input_ids, q_attention_mask, is_query=True)
                pos_reps = model(pos_input_ids, pos_attention_mask, is_query=False)
                neg_reps = model(neg_input_ids, neg_attention_mask, is_query=False)

                # Calculate scores
                pos_scores = model.score(q_reps, pos_reps)
                neg_scores = model.score(q_reps, neg_reps)

                # Triplet loss: maximize pos_scores, minimize neg_scores
                loss = torch.mean(torch.clamp(neg_scores - pos_scores + MARGIN, min=0))
                loss = loss / GRADIENT_ACCUMULATION_STEPS  # Scale loss for gradient accumulation

            # Backward pass with scaler
            scaler.scale(loss).backward()

            # Step optimizer and scaler if needed
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
        else:
            # Standard forward pass without mixed precision
            q_reps = model(q_input_ids, q_attention_mask, is_query=True)
            pos_reps = model(pos_input_ids, pos_attention_mask, is_query=False)
            neg_reps = model(neg_input_ids, neg_attention_mask, is_query=False)

            # Calculate scores
            pos_scores = model.score(q_reps, pos_reps)
            neg_scores = model.score(q_reps, neg_reps)

            # Triplet loss: maximize pos_scores, minimize neg_scores
            loss = torch.mean(torch.clamp(neg_scores - pos_scores + MARGIN, min=0))
            loss = loss / GRADIENT_ACCUMULATION_STEPS  # Scale loss for gradient accumulation

            # Backward pass
            loss.backward()

            # Step optimizer if needed
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS  # Adjust for scaling

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}")
            if batch_idx > 0:  # Skip first batch for more stable metrics
                print(
                    f"    Avg Pos Score: {pos_scores.mean().item():.4f}, Avg Neg Score: {neg_scores.mean().item():.4f}")

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

    # Run validation
    val_accuracy = run_validation(model, train_dataset, DEVICE, num_samples=100)

    # Save checkpoint after each epoch
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_loss,
        'val_accuracy': val_accuracy
    }
    torch.save(checkpoint, os.path.join(OUTPUT_DIR, f"colbert_hpo_checkpoint_epoch_{epoch + 1}.pt"))
    print(f"Checkpoint saved for epoch {epoch + 1}")

# Save the final model
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "colbert_hpo_model_triplet.pt"))
print(f"Final model saved to {OUTPUT_DIR}")


# Inference function for reranking
def rerank_candidates(span, candidate_terms, candidate_ids, model, tokenizer):
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


# Function to apply reranking to the entire CSV file
def rerank_csv(input_csv, output_csv, model, tokenizer, top_k=10):
    print(f"Loading CSV file: {input_csv}")
    df = pd.read_csv(input_csv, encoding="utf-8")

    print(f"Processing {len(df)} rows for reranking...")

    # Initialize new columns
    df["ft-colbert_reranking_terms"] = None
    df["ft-colbert_reranking"] = None
    df["ft-colbert_scores"] = None

    skipped_rows = 0
    processed_rows = 0

    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"Processing row {idx}/{len(df)}")

        try:
            # Get the text span
            span = row["span"]  # Adjust column name if needed

            # Skip rows with missing data
            if pd.isna(row[f"rag_candidates_euclidean_terms_top_{top_k}"]) or pd.isna(
                    row[f"rag_candidates_euclidean_top_{top_k}"]):
                skipped_rows += 1
                continue

            # Get candidate terms and IDs, converting to string first
            candidate_terms = row[f"rag_candidates_euclidean_terms_top_{top_k}"].split("||")
            candidate_ids = row[f"rag_candidates_euclidean_top_{top_k}"].split(",")

            reranked_terms, reranked_ids, reranked_scores = rerank_candidates(
                span, candidate_terms, candidate_ids, model, tokenizer
            )

            # Update the dataframe with reranked results
            df.at[idx, "ft-colbert_reranking_terms"] = "||".join(reranked_terms)
            df.at[idx, "ft-colbert_reranking"] = ",".join(reranked_ids)
            df.at[idx, "ft-colbert_scores"] = ",".join([str(score) for score in reranked_scores])

            processed_rows += 1

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            skipped_rows += 1
            continue

    print(f"Reranking completed: {processed_rows} rows processed, {skipped_rows} rows skipped")
    print(f"Saving results to {output_csv}")
    df.to_csv(output_csv, index=False)
    print("Reranking completed!")


# Example usage after training
if __name__ == "__main__":
    # === Configuration ===
    target_dataset = "chu50_v2"
    top_k = 50
    alpha = 0.5
    threshold = 0.8
    euc_model = "base"
    hit_model = "snomed"
    euc_model_rerank = "base"
    hit_model_rerank = "snomed"
    normalization_mode = "global"
    mips = "hyp-knn"

    # For applying the model to rerank candidates
    INPUT_CSV = data_path / f"rag/reranking/output_rag_{target_dataset}_candidates_{top_k}_reranked_{alpha}_{threshold}_{euc_model_rerank}_{hit_model_rerank}_{normalization_mode}.csv"
    OUTPUT_CSV = data_path / f"late-interaction/reranking/output_rag_{target_dataset}_candidates_{top_k}_reranked_{alpha}_{threshold}_{euc_model_rerank}_{hit_model_rerank}_{normalization_mode}_ft-colbert-triplet.csv"

    # Check if we should train or just run inference
    RUN_TRAINING = False  # Set to True to run training, False to just run inference

    if RUN_TRAINING:
        # Training code will be executed
        print("Running training...")
        # (The training code above would be executed)
    else:
        print("Skipping training, running inference only...")

    # Load the fine-tuned model for inference
    print(f"Loading model from {OUTPUT_DIR}...")
    base_model_inference = AutoModel.from_pretrained(MODEL_NAME)
    inference_model = ColBERT(base_model_inference)

    # Try to load from checkpoint first, fall back to final model
    try:
        checkpoint_path = os.path.join(OUTPUT_DIR, "colbert_hpo_checkpoint_epoch_2.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            inference_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from checkpoint (epoch 2)")
        else:
            inference_model.load_state_dict(
                torch.load(os.path.join(OUTPUT_DIR, "colbert_hpo_model_triplet.pt"), map_location=DEVICE))
            print("Loaded model from final saved model")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Initializing model with random weights")

    inference_model.to(DEVICE)
    inference_model.eval()

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Apply reranking to the CSV file
    print(f"Starting reranking process...")
    rerank_csv(INPUT_CSV, OUTPUT_CSV, inference_model, tokenizer, top_k)
    print(f"Reranking completed. Results saved to {OUTPUT_CSV}")
