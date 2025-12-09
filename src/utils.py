"""
HPO Ontology Processing and Distance Computation Utilities

This module provides functions for:

- Parsing Human Phenotype Ontology (HPO) OBO files to extract synonym-to-ID and ID-to-original-term mappings.
- Mapping candidate synonyms to original HPO terms.
- Normalizing hyperbolic distances with multiple normalization strategies.
- Managing candidate mappings between synonyms, IDs, and original terms.
- Computing pairwise hyperbolic or Euclidean distances efficiently, including GPU-accelerated batch computations.
- Loading and saving pairwise distance matrices.
- Identifying term pairs with maximum distances.
- Creating mappings between HPO IDs and embedding matrix indices.
- Merging and filtering JSON training data files with usage statistics.
"""


from hyperrag.config import *
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from collections import defaultdict


def parse_synonym_mapping(obo_file, synonym_mapping_output, original_terms_output):
    """
    Parse an OBO file to create a synonym-to-ID mapping and an ID-to-original-term mapping.

    Args:
        obo_file (str): Path to the OBO file.
        synonym_mapping_output (str): Path to save the synonym-to-ID mapping.
        original_terms_output (str): Path to save the ID-to-original-term mapping.

    Returns:
        tuple:
            - synonym_mapping (dict): Mapping from synonyms to IDs.
            - original_terms (dict): Mapping from IDs to their original terms.
    """
    synonym_mapping = {}
    original_terms = {}

    with open(obo_file, "r", encoding="utf-8") as f:
        current_id = None
        current_name = None

        for line in f:
            line = line.strip()

            # New term entry
            if line.startswith("[Term]"):
                current_id = None
                current_name = None

            # Extract ID
            elif line.startswith("id:"):
                current_id = line.split("id:")[1].strip()

            # Extract name (original term)
            elif line.startswith("name:"):
                current_name = line.split("name:")[1].strip()
                if current_id:
                    original_terms[current_id] = current_name

            # Extract synonyms
            elif line.startswith("synonym:"):
                if current_id:
                    synonym = line.split("\"")[1]  # Extract text inside quotes
                    synonym_mapping[synonym] = current_id

    # Save mappings to files
    with open(synonym_mapping_output, "w", encoding="utf-8") as f:
        json.dump(synonym_mapping, f, indent=4, ensure_ascii=False)

    with open(original_terms_output, "w", encoding="utf-8") as f:
        json.dump(original_terms, f, indent=4, ensure_ascii=False)

    return synonym_mapping, original_terms


def map_synonyms_to_originals(candidates, synonym_mapping):
    """
    Replace synonyms in the candidate list with their corresponding original terms.

    Args:
        candidates (list): List of candidate terms from RAG.
        synonym_mapping (dict): Synonym-to-original term mapping.

    Returns:
        list: List of candidates mapped to original terms.
    """
    return [synonym_mapping.get(candidate, candidate) for candidate in candidates]


def save_mapping_to_file(mapping, file_path):
    """
    Save the mapping dictionary to a JSON file.

    Args:
        mapping (dict): The synonym-to-original term mapping.
        file_path (str): Path to the output JSON file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4, ensure_ascii=False)  # ensure_ascii=False for non-ASCII characters


def load_mapping_from_file(file_path):
    """
    Load the mapping dictionary from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: The loaded synonym-to-original term mapping.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ================================
# DISTANCE NORMALIZATION
# ================================
def normalize_distances(distances, normalization_mode, global_max_distance=None):
    """
    Normalize hyperbolic distances based on the selected normalization mode.
    Global normalization: Perform global normalization using the entire HPO ontology.
    Local normalization: Add a local adjustment step for the top-k candidates,
    ensuring the normalized distances are dynamically scaled for query-specific variation.

    Args:
        distances (np.ndarray): Hyperbolic distances to normalize.
        normalization_mode (str): One of ['none', 'global', 'local', 'hybrid'].
        global_max_distance (float): Precomputed global maximum distance for 'global' or 'hybrid' normalization.

    Returns:
        np.ndarray: Normalized distances.
    """
    if normalization_mode == 'none':
        return distances  # No normalization

    distances = np.array(distances)  # Ensure distances are in array form

    if normalization_mode == 'global':
        assert global_max_distance is not None, "global_max_distance is required for 'global' normalization."
        return distances / global_max_distance

    elif normalization_mode == 'local':
        local_min = distances.min()
        local_max = distances.max()
        return (distances - local_min) / (local_max - local_min)

    elif normalization_mode == 'hybrid':
        assert global_max_distance is not None, "global_max_distance is required for 'hybrid' normalization."
        # Global normalization
        normalized_distances = distances / global_max_distance
        # Local adjustment
        local_min = normalized_distances.min()
        local_max = normalized_distances.max()
        return (normalized_distances - local_min) / (local_max - local_min)

    else:
        raise ValueError(f"Unknown normalization mode: {normalization_mode}")


def map_candidates_to_originals(rag_candidates, synonym_mapping, original_terms):
    """
    Map RAG candidates to their corresponding original names via IDs.

    Args:
        rag_candidates (list): List of candidate names or synonyms from RAG.
        synonym_mapping (dict): Mapping from synonyms/names to original IDs.
        original_terms (dict): Mapping from IDs to their original terms.

    Returns:
        tuple:
            - List of unique original terms for distance computation.
            - Dictionary mapping IDs to their synonyms and original names.
    """
    id_to_synonyms_and_original = defaultdict(lambda: {"synonyms": [], "original": None})
    unique_original_terms = set()

    for candidate in rag_candidates:
        # First, check if the candidate is a synonym and map it to its ID
        term_id = synonym_mapping.get(candidate)

        # If not found in synonym_mapping, check if it's already an original term
        if not term_id:
            term_id = next((id_ for id_, name in original_terms.items() if name == candidate), None)

        # If still not found, raise an error
        if not term_id:
            raise ValueError(f"Candidate '{candidate}' is neither in synonym mapping nor in original terms.")

        # Map the ID to its original term
        original_name = original_terms.get(term_id)
        if not original_name:
            raise ValueError(f"ID '{term_id}' does not have an original term.")

        # Populate the mapping
        id_to_synonyms_and_original[term_id]["synonyms"].append(candidate)
        id_to_synonyms_and_original[term_id]["original"] = original_name

        # Add to unique original terms
        unique_original_terms.add(original_name)

    return list(unique_original_terms), id_to_synonyms_and_original



def assign_distances_to_synonyms(ranked_originals, id_to_synonyms_and_original):
    """
    Assign distances from original terms to all synonyms sharing the same ID.

    Args:
        ranked_originals (list): List of (original_term, distance) pairs from reranking.
        id_to_synonyms_and_original (dict): Dictionary mapping IDs to their synonyms and original names.

    Returns:
        list: Ranked list of (synonym, ID, distance).
    """
    ranked_candidates = []

    # Map original terms to distances
    original_to_distance = {original: distance for original, distance in ranked_originals}

    # Assign distances to synonyms
    for term_id, data in id_to_synonyms_and_original.items():
        original_name = data["original"]
        synonyms = data["synonyms"]

        # Get distance for the original term
        distance = original_to_distance.get(original_name, float("inf"))

        # Assign the same distance to all synonyms
        for synonym in synonyms:
            ranked_candidates.append((synonym, term_id, distance))

    # Sort by distance
    return sorted(ranked_candidates, key=lambda x: x[2])



def compute_global_max_distance_optimized(embeddings, model):
    """
    Optimized computation of global maximum hyperbolic distance.

    Args:
        embeddings (torch.Tensor): Precomputed embeddings of all terms.
        model: The hyperbolic embedding model.

    Returns:
        float: The maximum distance.
    """
    max_distance = 0

    for i, emb1 in enumerate(embeddings):
        distances = model.manifold.dist(emb1.unsqueeze(0), embeddings).flatten()
        max_distance = max(max_distance, distances.max().item())

    return max_distance


def compute_global_min_distance_optimized(embeddings, model):
    """
    Optimized computation of global minimum hyperbolic distance, excluding self-distances.

    Args:
        embeddings (torch.Tensor): Precomputed embeddings of all terms.
        model: The hyperbolic embedding model.

    Returns:
        float: The minimum distance (excluding self-distances).
    """
    # Compute pairwise distances for all embeddings
    pairwise_distances = torch.zeros(len(embeddings), len(embeddings))

    for i, emb1 in enumerate(embeddings):
        pairwise_distances[i] = model.manifold.dist(emb1.unsqueeze(0), embeddings).flatten()

    # Mask the diagonal (self-distances)
    pairwise_distances.fill_diagonal_(float("inf"))

    # Find the minimum distance in the masked matrix
    min_distance = pairwise_distances.min().item()

    return min_distance


def compute_min_distance_from_file(distance_matrix):
    """
    Compute the global minimum distance, excluding self-distances.

    Args:
        distance_matrix (torch.Tensor): Pairwise distance matrix.

    Returns:
        float: Minimum distance (excluding self-distances).
    """
    # Mask self-distances
    distance_matrix.fill_diagonal_(float("inf"))

    # Debug: Check if diagonal masking is applied
    print("Diagonal after masking:", torch.diag(distance_matrix))

    # Find the minimum distance
    min_distance = distance_matrix.min().item()
    return min_distance


def compute_and_save_distances(embeddings, model, output_path="pairwise_distances.npy"):
    """
    Compute and save pairwise distances between embeddings.

    Args:
        embeddings (torch.Tensor): Precomputed embeddings of all terms.
        model: The hyperbolic embedding model.
        output_path (str): Path to save the pairwise distances file.

    Returns:
        None
    """
    # Compute pairwise distances for all embeddings
    pairwise_distances = torch.zeros(len(embeddings), len(embeddings))

    for i, emb1 in enumerate(embeddings):
        distances = model.manifold.dist(emb1.unsqueeze(0), embeddings).flatten()

        # Debug: Check for zero distances
        if torch.any(distances > 0):
            # print(f"Row {i} has valid distances.")
            continue
        else:
            print(f"Row {i} contains invalid distances.")

        pairwise_distances[i] = distances

    # Save distances as a NumPy binary file
    np.save(output_path, pairwise_distances.numpy())
    print(f"Pairwise distances saved to {output_path}")


def compute_pairwise_distances_gpu(embeddings, model):
    """
    Compute pairwise hyperbolic distances using GPU.

    Args:
        embeddings (torch.Tensor): Precomputed embeddings of all terms (on GPU if available).
        model: The hyperbolic embedding model.

    Returns:
        torch.Tensor: Pairwise distance matrix.
    """
    # Ensure embeddings are on the same device as the model (GPU if available)
    device = embeddings.device
    num_embeddings = embeddings.size(0)

    # Create an empty distance matrix
    pairwise_distances = torch.empty((num_embeddings, num_embeddings), device=device)

    # Compute distances in batches to fit GPU memory
    batch_size = 128  # Adjust batch size based on GPU memory
    for i in range(0, num_embeddings, batch_size):
        end_idx = min(i + batch_size, num_embeddings)
        batch_embeddings = embeddings[i:end_idx]

        # Compute distances between the batch and all embeddings
        distances = model.manifold.dist(batch_embeddings.unsqueeze(1), embeddings.unsqueeze(0))
        pairwise_distances[i:end_idx] = distances.squeeze(1)

    return pairwise_distances


import torch
import torch.nn.functional as F


def compute_pairwise_distances_gpu_mini(embeddings, model, distance="hyperbolic", batch_size=1024, output_path="pairwise_distances.npy"):
    """
    Compute and save pairwise distances between embeddings using mini-batches.

    Args:
        embeddings (torch.Tensor): Precomputed embeddings of all terms (on CPU).
        model: The hyperbolic embedding model.
        batch_size (int): Number of embeddings to process at once.
        output_path (str): Path to save the pairwise distances file.

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure embeddings are in NumPy format for Euclidean, and Torch format for Hyperbolic
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.tensor(embeddings, dtype=torch.float32, device=device)

    num_embeddings = embeddings.shape[0]
    pairwise_distances = np.zeros((num_embeddings, num_embeddings), dtype=np.float32)

    # Compute distances in mini-batches to prevent OOM
    for i in range(0, num_embeddings, batch_size):
        end_i = min(i + batch_size, num_embeddings)
        emb_i = embeddings[i:end_i].to(device)  # Move batch to GPU

        for j in range(0, num_embeddings, batch_size):
            end_j = min(j + batch_size, num_embeddings)
            emb_j = embeddings[j:end_j].to(device)  # Move batch to GPU

            if distance == "hyperbolic":
                # Compute hyperbolic distance
                distances = model.manifold.dist(emb_i.unsqueeze(1), emb_j.unsqueeze(0)).cpu()  # Move results to CPU
                # distances = model.compute_hyperbolic_distance(emb_i.unsqueeze(1), emb_j.unsqueeze(0)).cpu()  # using HiT distance
            elif distance == "euclidean":
                # Compute Cosine Similarity using Scikit-learn (better memory efficiency)
                emb_i_cpu = emb_i.cpu().numpy()
                emb_j_cpu = emb_j.cpu().numpy()
                distances = cosine_distances(emb_i_cpu, emb_j_cpu)
                # distances = np.clip(cosine_distances(emb_i_cpu, emb_j_cpu), 0, 1)

            else:
                raise ValueError("Invalid distance type. Choose 'hyperbolic' or 'euclidean'.")

            # Store results
            pairwise_distances[i:end_i, j:end_j] = distances

        # Free GPU memory
        del emb_i
        torch.cuda.empty_cache()

    # Save to disk
    np.save(output_path, pairwise_distances)
    print(f"Pairwise distances saved to {output_path}")


def load_pairwise_distances(file_path="pairwise_distances.npy"):
    """
    Load pairwise distances from a file.

    Args:
        file_path (str): Path to the saved pairwise distances file.

    Returns:
        torch.Tensor: Loaded pairwise distances.
    """
    distances = np.load(file_path)

    return torch.tensor(distances)


def find_max_distance_pairs(distances, term_to_index_path="term_to_index.json"):
    """
    Find pairs with maximum distance in the distance matrix.

    Args:
        distances (torch.Tensor): Pairwise distance matrix
        term_to_index_path (str): Path to term_to_index JSON file

    Returns:
        float: Maximum distance
        list: List of tuples containing pairs with maximum distance
    """
    # Load term to index mapping
    with open(term_to_index_path, 'r') as f:
        term_to_index = json.load(f)
    index_to_term = {v: k for k, v in term_to_index.items()}

    # Find maximum distance
    max_distance = torch.max(distances)

    # Find all pairs with maximum distance
    max_pairs = []
    rows, cols = torch.where(distances == max_distance)

    for row, col in zip(rows, cols):
        if row < col:  # Avoid duplicates due to symmetry
            term1 = index_to_term[row.item()]
            term2 = index_to_term[col.item()]
            max_pairs.append((term1, term2))

    return max_distance.item(), max_pairs


def create_id2idx_mapping(term_to_index_path, original_terms_path, synonym_mapping_path):
    """
    Create id2idx mapping from multiple source files

    Args:
        term_to_index_path: path to term_to_index file
        original_terms_path: path to original_terms.json
        synonym_mapping_path: path to synonym_mapping.json

    Returns:
        dict: mapping from HP IDs to matrix indices
    """
    # Load files
    with open(term_to_index_path, 'r') as f:
        term_to_index = json.load(f)

    with open(original_terms_path, 'r') as f:
        original_terms = json.load(f)

    with open(synonym_mapping_path, 'r') as f:
        synonym_mapping = json.load(f)

    # Create reverse mapping from HP IDs to terms
    hp_to_term = {v: k for k, v in synonym_mapping.items()}
    hp_to_term.update({hp_id: term for hp_id, term in original_terms.items()})

    # Create final id2idx mapping
    id2idx = {}
    for hp_id, term in hp_to_term.items():
        if term in term_to_index:
            id2idx[hp_id] = term_to_index[term]

    return id2idx


# Concatenate training data
def merge_json_files_in_folder(folder_path):
    merged_data = {
        "entries": [],
        "usage_summary": {
            "total_stats": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_cost": 0,
                "output_cost": 0,
                "total_cost": 0
            },
            "average_per_entry": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_cost": 0,
                "output_cost": 0,
                "total_cost": 0
            },
            "entries_processed": 0
        }
    }

    # Get all JSON files in the folder
    json_files = list(Path(folder_path).glob('*.json'))

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

                # Concatenate entries
                merged_data["entries"].extend(data["entries"])

                # Sum up usage statistics
                stats = data["usage_summary"]["total_stats"]
                for key in stats:
                    merged_data["usage_summary"]["total_stats"][key] += stats[key]

                print(f"Processed: {json_file}")

        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")

    # Update entries processed based on actual entries count
    total_entries = len(merged_data["entries"])
    merged_data["usage_summary"]["entries_processed"] = total_entries

    # Calculate new averages
    if total_entries > 0:
        for key in merged_data["usage_summary"]["average_per_entry"]:
            merged_data["usage_summary"]["average_per_entry"][key] = (
                    merged_data["usage_summary"]["total_stats"][key] / total_entries
            )

    return merged_data


def filter_merged_file(input_file_path):
    # Load the merged file
    try:
        with open(input_file_path, 'r') as f:
            merged_data = json.load(f)
            print(f"Successfully loaded: {input_file_path}")
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return

    # Filter entries to keep only desired fields
    filtered_data = {
        "entries": [
            {
                "hpo_label": entry["hpo_label"],
                "hpo_id": entry["hpo_id"],
                "spans": entry["spans"]
            }
            for entry in merged_data["entries"]
        ],
        "usage_summary": merged_data["usage_summary"]
    }

    # Generate output filename
    input_path = Path(input_file_path)
    output_path = input_path.parent / f"spans_{input_path.name}"

    # Save filtered data
    with open(output_path, 'w') as f:
        json.dump(filtered_data, f, indent=4)
    print(f"Filtered file saved as: {output_path}")


# ===== Usage =====

# Compute max distance
# distances = load_pairwise_distances(data_path / "hpo/pairwise_distances_hyp_snomed.npy")
# index_path = data_path / "hpo/term_to_index_hyp_snomed.json"
# max_dist, max_pairs = find_max_distance_pairs(distances, index_path)
# print(f"Maximum distance: {max_dist}")
# print("Most distant pairs:")
# for pair in max_pairs:
#     print(f"{pair[0]} - {pair[1]}")


# Parse the .obo file (once)
# obo_file = data_path + "hpo/20241212_hp.obo"
# synonym_mapping_file = data_path + "hpo/synonym_mapping.json"
# original_terms_file = data_path + "hpo/original_terms.json"
# synonym_mapping, original_terms = parse_synonym_mapping(obo_file, synonym_mapping_file, original_terms_file)


