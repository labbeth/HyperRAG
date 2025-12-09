"""
Retrieval-Augmented Generation Script

This script implements a vector store for Human Phenotype Ontology (HPO) terms using both Euclidean and hyperbolic embeddings.

Features:
- Loads HPO terms and their synonyms from JSON files.
- Builds vector indices using SentenceTransformer or HierarchyTransformer models.
- Supports embedding sharing for synonyms.
- Provides querying functionality to retrieve top-k similar HPO terms for input text.
- Saves and loads vector stores to/from disk.
- Processes input CSV files containing spans to generate candidate HPO terms using both Euclidean and hyperbolic vector stores.
- Outputs enriched CSV files with candidate HPO IDs and optionally matched terms.
"""


from hyperrag.config import *
import pandas as pd
import numpy as np
import json
import faiss
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from hierarchy_transformers import HierarchyTransformer
from typing import List, Dict, Tuple, Union
import os
from tqdm import tqdm
import pickle
import geoopt


class HPOVectorStore:
    def __init__(self, model_name: str, is_hyperbolic: bool = False):
        """
        Initialize HPO vector store with specified embedding model

        Args:
            model_name: Name of the sentence transformer model to use
            is_hyperbolic: Whether the model uses hyperbolic embeddings
        """
        self.model_name = model_name
        self.is_hyperbolic = is_hyperbolic

        if is_hyperbolic:
            self.embedding_model = HierarchyTransformer.load_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.embedding_model.to(self.device)
        else:
            self.embedding_model = SentenceTransformer(model_name)

        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.embeddings = None  # Store embeddings directly instead of using FAISS
        self.id_to_hpo = {}
        self.hpo_to_id = {}
        self.id_to_term = {}

    def load_hpo_data(self, original_terms_path: str, synonym_mapping_path: str):
        """
        Load HPO terms and synonyms from JSON files

        Args:
            original_terms_path: Path to original_terms.json
            synonym_mapping_path: Path to synonym_mapping.json
        """
        # Load original terms
        with open(original_terms_path, 'r') as f:
            original_terms = json.load(f)

        # Load synonym mapping
        with open(synonym_mapping_path, 'r') as f:
            synonym_mapping = json.load(f)

        # Create a mapping from HPO ID to all its terms (label + synonyms)
        self.hpo_to_terms = {}

        # Add original terms
        for hpo_id, term in original_terms.items():
            self.hpo_to_terms[hpo_id] = [term]

        # Add synonyms
        for synonym, hpo_id in synonym_mapping.items():
            if hpo_id in self.hpo_to_terms:
                self.hpo_to_terms[hpo_id].append(synonym)

        # Create lists to store all terms and their corresponding HPO IDs
        all_terms = []
        all_hpo_ids = []

        for hpo_id, terms in self.hpo_to_terms.items():
            for term in terms:
                all_terms.append(term)
                all_hpo_ids.append(hpo_id)

        # Create mapping from index to HPO ID and term
        self.id_to_hpo = {i: hpo_id for i, hpo_id in enumerate(all_hpo_ids)}
        self.id_to_term = {i: term for i, term in enumerate(all_terms)}
        self.hpo_to_id = {hpo_id: i for i, hpo_id in self.id_to_hpo.items()}

        # Store for later use
        self.all_terms = all_terms
        self.all_hpo_ids = all_hpo_ids

        print(
            f"Loaded {len(self.hpo_to_terms)} unique HPO terms with {len(all_terms)} total terms (including synonyms)")



    def build_index(self, share_synonym_embeddings=False):
        """
        Build index from all terms

        Args:
            share_synonym_embeddings: If True, synonyms will share the same embedding as their original term
        """
        print(f"Building index with {self.model_name} (hyperbolic: {self.is_hyperbolic})")
        print(f"Share synonym embeddings: {share_synonym_embeddings}")

        if share_synonym_embeddings:
            # Compute embeddings only for original terms
            original_terms = set()
            term_to_original = {}  # Maps each term to its original term

            for hpo_id, terms in self.hpo_to_terms.items():
                original_term = terms[0]  # First term is the original
                original_terms.add(original_term)

                # Map all terms to the original term
                for term in terms:
                    term_to_original[term] = original_term

            original_terms_list = list(original_terms)

            if self.is_hyperbolic:
                with torch.no_grad():
                    original_embeddings = self.embedding_model.encode(
                        original_terms_list,
                        convert_to_tensor=True
                    ).to(self.device)

                term_to_embedding = {term: emb for term, emb in zip(original_terms_list, original_embeddings)}
                self.embeddings = torch.stack([
                    term_to_embedding[term_to_original[term]]
                    for term in self.all_terms
                ])
            else:
                original_embeddings = self.embed_terms(original_terms_list)
                term_to_embedding = {term: emb for term, emb in zip(original_terms_list, original_embeddings)}
                final_embeddings = np.vstack([
                    term_to_embedding[term_to_original[term]]
                    for term in self.all_terms
                ])
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index.add(final_embeddings)

        else:
            # Original behavior: compute unique embeddings for all terms
            if self.is_hyperbolic:
                with torch.no_grad():
                    self.embeddings = self.embedding_model.encode(
                        self.all_terms,
                        convert_to_tensor=True
                    ).to(self.device)
            else:
                embeddings = self.embed_terms(self.all_terms)
                self.index = faiss.IndexFlatL2(self.dimension)
                self.index.add(embeddings)

        print(f"Index built with {len(self.all_terms)} vectors of dimension {self.dimension}")

    def embed_terms(self, terms: List[str]) -> np.ndarray:
        """
        Embed a list of terms using the embedding model

        Args:
            terms: List of strings to embed

        Returns:
            Array of embeddings
        """
        # Embed in batches to avoid OOM issues
        batch_size = 128
        all_embeddings = []

        for i in range(0, len(terms), batch_size):
            batch = terms[i:i + batch_size]
            embeddings = self.embedding_model.encode(batch, convert_to_numpy=True)
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def query(self, query_text: str, k: int = 10, include_terms: bool = False):
        """
        Query the index for similar HPO terms
        """
        if not isinstance(query_text, str):
            query_text = str(query_text)

        if not query_text or query_text.strip() == "":
            return []

        if self.is_hyperbolic:
            # Hyperbolic search
            with torch.no_grad():
                query_embedding = self.embedding_model.encode(
                    [query_text],
                    convert_to_tensor=True
                ).to(self.device)

                # # Compute hyperbolic distances
                # distances = torch.stack([
                #     self.embedding_model.manifold.dist(
                #         query_embedding,
                #         emb.unsqueeze(0)
                #     )
                #     for emb in self.embeddings
                # ])

                # Compute all distances at once
                distances = self.embedding_model.manifold.dist(
                    query_embedding.expand(self.embeddings.shape[0], -1),
                    self.embeddings
                )

                # Get top-k using torch.topk
                distances, indices = torch.topk(distances.squeeze(), k, largest=False)
                # return [self.id_to_hpo[int(idx)] for idx in indices.cpu().numpy()]

                # Format results
                results = []
                for idx, distance in zip(indices.cpu().numpy(), distances.cpu().numpy()):
                    if idx >= 0 and idx < len(self.id_to_hpo):
                        hpo_id = self.id_to_hpo[int(idx)]

                        if include_terms:
                            matched_term = self.id_to_term[int(idx)]
                            results.append((hpo_id, matched_term, float(distance)))
                        else:
                            results.append((hpo_id, float(distance)))

                return results

        else:
            # Euclidean search using FAISS
            query_embedding = self.embedding_model.encode(query_text, convert_to_numpy=True)
            query_embedding = query_embedding.reshape(1, -1)
            distances, indices = self.index.search(query_embedding, k)
            indices = indices[0]
            distances = distances[0]

        # Format results
        results = []
        for idx, distance in zip(indices, distances):
            if idx >= 0 and idx < len(self.id_to_hpo):
                hpo_id = self.id_to_hpo[idx]

                if include_terms:
                    matched_term = self.id_to_term[idx]
                    results.append((hpo_id, matched_term, float(distance)))
                else:
                    results.append((hpo_id, float(distance)))

        return results

    def save_to_disk(self, directory: str):
        """
        Save the vector store to disk
        """
        os.makedirs(directory, exist_ok=True)

        if self.is_hyperbolic:
            # Save embeddings tensor
            embeddings_path = os.path.join(directory, 'embeddings.pt')
            torch.save(self.embeddings.cpu(), embeddings_path)
        else:
            # Save FAISS index
            index_path = os.path.join(directory, 'index.faiss')
            faiss.write_index(self.index, index_path)

        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'is_hyperbolic': self.is_hyperbolic,
            'dimension': self.dimension,
            'id_to_hpo': self.id_to_hpo,
            'id_to_term': self.id_to_term,
            'hpo_to_id': self.hpo_to_id,
            'hpo_to_terms': self.hpo_to_terms,
            'all_terms': self.all_terms,
            'all_hpo_ids': self.all_hpo_ids
        }
        metadata_path = os.path.join(directory, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Vector store saved to {directory}")

    @classmethod
    def load_from_disk(cls, directory: str):
        """
        Load vector store from disk
        """
        # Load metadata
        metadata_path = os.path.join(directory, 'metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        # Create vector store
        vector_store = cls(metadata['model_name'], metadata['is_hyperbolic'])

        if vector_store.is_hyperbolic:
            # Load embeddings tensor
            embeddings_path = os.path.join(directory, 'embeddings.pt')
            vector_store.embeddings = torch.load(embeddings_path).to(vector_store.device)
        else:
            # Load FAISS index
            index_path = os.path.join(directory, 'index.faiss')
            vector_store.index = faiss.read_index(index_path)

        # Load other metadata
        vector_store.dimension = metadata['dimension']
        vector_store.id_to_hpo = metadata['id_to_hpo']
        vector_store.hpo_to_id = metadata['hpo_to_id']
        vector_store.hpo_to_terms = metadata['hpo_to_terms']
        vector_store.all_terms = metadata['all_terms']
        vector_store.all_hpo_ids = metadata['all_hpo_ids']
        vector_store.id_to_term = metadata.get('id_to_term', {})

        print(f"Vector store loaded from {directory}")
        print(f"Index has {len(vector_store.all_terms)} vectors of dimension {vector_store.dimension}")

        return vector_store


# ================================
# PROCESS FILE
# ================================
def process_csv(input_path: str, output_path: str, euclidean_store: HPOVectorStore,
                hyperbolic_store: HPOVectorStore, top_k: int = 10, include_terms: bool = False):
    """
    Process input CSV file and write results to output CSV

    Args:
        input_path: Path to input CSV
        output_path: Path to output CSV
        euclidean_store: HPOVectorStore with euclidean embeddings
        hyperbolic_store: HPOVectorStore with hyperbolic embeddings
        top_k: Number of top results to return
        include_terms: Whether to include matched terms in the output
    """
    # Load input CSV
    df = pd.read_csv(input_path)

    # Check if 'span' column exists
    if 'span' not in df.columns:
        raise ValueError("Input CSV must contain a 'span' column")

    # Create output columns
    df[f'rag_candidates_euclidean_top_{top_k}'] = None

    if include_terms:
        df[f'rag_candidates_euclidean_terms_top_{top_k}'] = None

    df[f'rag_candidates_hyperbolic_top_{top_k}'] = None

    if include_terms:
        df[f'rag_candidates_hyperbolic_terms_top_{top_k}'] = None

    # Process each row
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        span = row['span']

        # Skip empty spans
        if pd.isna(span) or str(span).strip() == "":
            df.at[idx, f'rag_candidates_euclidean_top_{top_k}'] = ""
            if include_terms:
                df.at[idx, f'rag_candidates_euclidean_terms_top_{top_k}'] = ""
            df.at[idx, f'rag_candidates_hyperbolic_top_{top_k}'] = ""
            if include_terms:
                df.at[idx, f'rag_candidates_hyperbolic_terms_top_{top_k}'] = ""
            continue

        # Get top-k results from euclidean index
        euclidean_results = euclidean_store.query(span, k=top_k, include_terms=include_terms)

        if include_terms:
            euclidean_hpo_ids = [result[0] for result in euclidean_results]
            euclidean_terms = [result[1] for result in euclidean_results]
            df.at[idx, f'rag_candidates_euclidean_top_{top_k}'] = ','.join(euclidean_hpo_ids)
            df.at[idx, f'rag_candidates_euclidean_terms_top_{top_k}'] = '||'.join(euclidean_terms)
        else:
            euclidean_hpo_ids = [result[0] for result in euclidean_results]
            df.at[idx, f'rag_candidates_euclidean_top_{top_k}'] = ','.join(euclidean_hpo_ids)

        # Get top-k results from hyperbolic index
        hyperbolic_results = hyperbolic_store.query(span, k=top_k, include_terms=include_terms)

        if include_terms:
            hyperbolic_hpo_ids = [result[0] for result in hyperbolic_results]
            hyperbolic_terms = [result[1] for result in hyperbolic_results]
            df.at[idx, f'rag_candidates_hyperbolic_top_{top_k}'] = ','.join(hyperbolic_hpo_ids)
            df.at[idx, f'rag_candidates_hyperbolic_terms_top_{top_k}'] = '||'.join(hyperbolic_terms)
        else:
            hyperbolic_hpo_ids = [result[0] for result in hyperbolic_results]
            df.at[idx, f'rag_candidates_hyperbolic_top_{top_k}'] = ','.join(hyperbolic_hpo_ids)

    # Write output CSV
    df.to_csv(output_path, index=False)
    print(f"Results written to {output_path}")

def main():

    # Define paths
    original_terms_path = ORIGINAL_TERMS_FILE
    synonym_mapping_path = SYNONYMS_MAPPING_FILE
    input_csv_path = data_path / f"eval_data/{target_dataset}/enriched_extracted_spans.csv"
    output_csv_path = data_path / f"rag/output_rag_{target_dataset}_candidates_{top_k}_{mips}_{euc_model}_{hit_model}.csv"

    # Define storage directories
    euclidean_store_dir = data_path / f'embeddings/euclidean_store_{euc_model}'
    hyperbolic_store_dir = data_path / f'embeddings/hyperbolic_store_{hit_model}'

    # Initialize stores
    euclidean_store = None
    hyperbolic_store = None

    # Try to load euclidean store from disk
    if os.path.exists(euclidean_store_dir):
        try:
            print("Loading euclidean vector store from disk...")
            euclidean_store = HPOVectorStore.load_from_disk(euclidean_store_dir)
        except Exception as e:
            print(f"Error loading euclidean vector store: {e}")
            euclidean_store = None

    # Build euclidean store if not loaded
    if euclidean_store is None:
        print("Building euclidean vector store...")
        euclidean_store = HPOVectorStore(euclidean_model, is_hyperbolic=False)
        euclidean_store.load_hpo_data(original_terms_path, synonym_mapping_path)
        euclidean_store.build_index(share_synonym_embeddings=False)

        # Save to disk
        euclidean_store.save_to_disk(euclidean_store_dir)

    # Try to load hyperbolic store from disk
    if os.path.exists(hyperbolic_store_dir):
        try:
            print("Loading hyperbolic vector store from disk...")
            hyperbolic_store = HPOVectorStore.load_from_disk(hyperbolic_store_dir)
        except Exception as e:
            print(f"Error loading hyperbolic vector store: {e}")
            hyperbolic_store = None

    # Build hyperbolic store if not loaded
    if hyperbolic_store is None:
        print("Building hyperbolic vector store...")
        hyperbolic_store = HPOVectorStore(hyperbolic_model, is_hyperbolic=True)
        hyperbolic_store.load_hpo_data(original_terms_path, synonym_mapping_path)
        hyperbolic_store.build_index(share_synonym_embeddings=False)

        # Save to disk
        hyperbolic_store.save_to_disk(hyperbolic_store_dir)

    # Process CSV
    print("Processing input CSV...")
    process_csv(input_csv_path, output_csv_path, euclidean_store, hyperbolic_store, top_k, include_terms=True)

    print("Done!")


if __name__ == "__main__":
    main()