"""
Hierarchical Evaluation and Visualization for HPO Candidate Retrieval using HyperRAG

This script provides comprehensive evaluation and visualization tools for models predicting Human Phenotype Ontology (HPO) terms.

Features include:
- Preprocessing of HPO hierarchical relationships for efficient computation.
- Computation of weighted metrics (precision, recall, MRR, NDCG) incorporating ontology relationships.
- Analysis of relationship types (exact match, ancestor, descendant, cousin, no path) and hop distributions.
- Calculation of branch coverage and miss rate metrics.
- Generation of detailed plots including distributions, heatmaps, combined metrics, and relationship type breakdowns.
- Creation of Markdown and LaTeX tables summarizing weighted metrics with best scores highlighted.
- Support for multiple models and multiple k-values in ranking evaluation.
- Extensive debugging and reporting of relationship coverage and metric statistics.
"""


from hyperrag.config import *
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd
import json
from typing import List, Dict, Tuple
import seaborn as sns



# =========================
# SOTA baselines (single-candidate systems)
# =========================
# Add your known values here. None means "unknown/not provided".
SOTA_BASELINES = {
    "ID-68": {
        "recall":          {"PhenoBERT": 0.781, "PhenoBERT_LLM": 0.5, "String_matching": 0.4},
        "recall_weighted": {"PhenoBERT": 0.773, "PhenoBERT_LLM": 0.5, "String_matching": 0.4},
        "precision":       {"PhenoBERT": 0.943, "PhenoBERT_LLM": 0.5, "String_matching": 0.4},
        "miss_rate":       {"PhenoBERT": 0.231, "PhenoBERT_LLM": 0.5, "String_matching": 0.4},
    },
    "CHU-50": {
        "recall":          {"PhenoBERT": 0.395, "PhenoBERT_LLM": 0.5, "String_matching": 0.4},
        "recall_weighted": {"PhenoBERT": 0.395, "PhenoBERT_LLM": 0.5, "String_matching": 0.4},
        "precision":       {"PhenoBERT": 0.671, "PhenoBERT_LLM": 0.5, "String_matching": 0.4},
        "miss_rate":       {"PhenoBERT": 0.606, "PhenoBERT_LLM": 0.5, "String_matching": 0.4},
    }
}

def resolve_sota_dataset_key(target_dataset: str) -> str:
    """
    Map your run dataset name to the SOTA dataset key.
    Adjust this function if your dataset names differ.
    """
    name = str(target_dataset).lower()
    if "chu" in name:
        return "CHU-50"
    if "id-68" in name or "id68" in name or "id_68" in name:
        return "ID-68"
    # fallback: use as-is
    return target_dataset

def build_sota_dataframe(k_values: List[int],
                         dataset_key: str,
                         baselines: Dict[str, Dict[str, Dict[str, float]]] = SOTA_BASELINES) -> pd.DataFrame:
    """
    Build a DataFrame with SOTA rows repeated for each k.
    Only the four supported metrics (recall, weighted_recall, precision, miss_rate) are filled.
    The rest are NaN, so they won't appear in other plots.
    """
    if dataset_key not in baselines:
        return pd.DataFrame()

    base = baselines[dataset_key]
    rows = []
    for system_name in {"PhenoBERT", "PhenoBERT_LLM", "String_matching"}:
        # Read values (may be None)
        recall = base.get("recall", {}).get(system_name)
        wrec = base.get("recall_weighted", {}).get(system_name)
        prec = base.get("precision", {}).get(system_name)
        miss = base.get("miss_rate", {}).get(system_name)

        # Skip systems with no values at all
        if all(v is None for v in [recall, wrec, prec, miss]):
            continue

        for k in k_values:
            rows.append({
                "k": k,
                "model": system_name,
                "hierarchical_recall": recall if recall is not None else np.nan,
                "weighted_recall": wrec if wrec is not None else np.nan,
                "hierarchical_precision": prec if prec is not None else np.nan,
                "miss_rate": miss if miss is not None else np.nan,
                # keep others empty for SOTA
                "average_hops": np.nan,
                "branch_coverage": np.nan,
                "mrr": np.nan,
                "ndcg": np.nan,
                "weighted_mrr": np.nan,
                "weighted_ndcg": np.nan,
                "weighted_precision": np.nan
            })
    return pd.DataFrame(rows)


# ==== Utils ====
def get_formatted_filename(base_name: str,
                         target_dataset: str,
                         top_k: int,
                         alpha: float,
                         threshold: float,
                         euc_model_rerank: str,
                         hit_model_rerank: str,
                         normalization_mode: str) -> str:
    """Create formatted filename with parameters"""
    return f"{target_dataset}_{top_k}_{alpha}_{threshold}_{euc_model_rerank}_{hit_model_rerank}_{normalization_mode}_{base_name}.png"


def preprocess_hpo_relationships(hpo_relationships):
    """
    Preprocess HPO relationships for faster access
    """
    print("Prétraitement des relations HPO pour optimiser les calculs...")

    # Create direct lookup tables
    ancestor_lookup = {}
    descendant_lookup = {}
    hops_lookup = {}

    # Build ancestor and descendant lookups
    for term, data in hpo_relationships.items():
        ancestor_lookup[term] = set(data['ancestors'])
        hops_lookup[term] = data['hops']

        # Build descendant lookup
        for ancestor in data['ancestors']:
            if ancestor not in descendant_lookup:
                descendant_lookup[ancestor] = set()
            descendant_lookup[ancestor].add(term)

    return {
        'ancestor_lookup': ancestor_lookup,
        'descendant_lookup': descendant_lookup,
        'hops_lookup': hops_lookup,
        'original': hpo_relationships
    }


# ==== Weighted metrics utils ====
def compute_relationship_weight(candidate: str,
                                target: str,
                                processed_hpo: Dict,
                                alpha: float = 1.0,
                                beta: float = 1.0,
                                weight_cache: Dict = None) -> float:
    """
    Compute weight between candidate and target based on their relationship
    """
    # Use cache if provided
    if weight_cache is not None:
        cache_key = (candidate, target)
        if cache_key in weight_cache:
            return weight_cache[cache_key]

    # Handle cases where terms are not in relationships
    if candidate not in processed_hpo['original'] or target not in processed_hpo['original']:
        result = 0.0
    elif candidate == target:
        result = 1.0
    else:
        # Get lookups
        ancestor_lookup = processed_hpo['ancestor_lookup']
        hops_lookup = processed_hpo['hops_lookup']

        is_direct = False
        d = 0
        p = 0

        # Check if candidate is ancestor of target
        if candidate in ancestor_lookup.get(target, set()):
            is_direct = True
            d = hops_lookup[target][candidate]
            # Count terms between them in the ancestry path
            p = sum(1 for term, hops in hops_lookup[target].items()
                    if hops < d and hops > 0)

        # Check if target is ancestor of candidate
        elif target in ancestor_lookup.get(candidate, set()):
            is_direct = True
            d = hops_lookup[candidate][target]
            # Count terms between them in the ancestry path
            p = sum(1 for term, hops in hops_lookup[candidate].items()
                    if hops < d and hops > 0)

        if is_direct:
            # Direct relationship weight
            result = alpha / (p * (1 + abs(d))) if p > 0 else 1.0
        else:
            # Find lowest common ancestor (shared superclass)
            candidate_ancestors = ancestor_lookup.get(candidate, set())
            target_ancestors = ancestor_lookup.get(target, set())
            common_ancestors = candidate_ancestors & target_ancestors

            if not common_ancestors:
                result = 0.0
            else:
                # Find the most specific common ancestor
                lca = min(common_ancestors,
                          key=lambda x: max(hops_lookup[candidate].get(x, float('inf')),
                                            hops_lookup[target].get(x, float('inf'))))

                # Count immediate children of LCA
                c = sum(1 for term in processed_hpo['descendant_lookup'].get(lca, set())
                        if hops_lookup[term].get(lca, float('inf')) == 1)

                # Find distance to farthest leaf from candidate
                d_l = max(hops_lookup[candidate].values(), default=0)

                # Indirect relationship weight
                result = beta / (c * (1 + d_l))

    # Store in cache if provided
    if weight_cache is not None:
        weight_cache[(candidate, target)] = result

    return result



def compute_weighted_metrics_at_k(predicted_lists: List[List[str]],
                                  ground_truth: List[str],
                                  k: int,
                                  hpo_relationships: Dict,
                                  alpha: float = 1.0,
                                  beta: float = 1.0) -> Dict[str, float]:
    """
    Compute weighted precision and recall @k
    """
    weighted_recalls = []
    weighted_precisions = []

    # Add debug counters
    debug_info = {
        'total_cases': 0,
        'missing_terms': set(),
        'zero_weights': 0,
        'direct_relationships': 0,
        'indirect_relationships': 0
    }

    for preds, truth in zip(predicted_lists, ground_truth):
        if not truth or pd.isna(truth):
            continue

        debug_info['total_cases'] += 1
        top_k_preds = preds[:k]

        # Compute weights for all predictions
        weights = []
        for pred in top_k_preds:
            weight = compute_relationship_weight(pred, truth, hpo_relationships, alpha, beta)
            weights.append(weight)

            # Collect debug information
            if pred not in hpo_relationships:
                debug_info['missing_terms'].add(pred)
            if weight == 0:
                debug_info['zero_weights'] += 1
            elif pred == truth or (pred in hpo_relationships.get(truth, {}).get('hops', {}) or
                                   truth in hpo_relationships.get(pred, {}).get('hops', {})):
                debug_info['direct_relationships'] += 1
            else:
                debug_info['indirect_relationships'] += 1

        # Weighted recall: maximum weight among predictions
        max_weight = max(weights) if weights else 0.0
        weighted_recalls.append(max_weight)

        # Weighted precision: average weight of predictions
        avg_weight = sum(weights) / k if k > 0 else 0.0
        weighted_precisions.append(avg_weight)

    # Print debug information
    print("\nWeighted Metrics Debug Information:")
    print(f"Total cases analyzed: {debug_info['total_cases']}")
    print(f"Terms not in relationships file: {len(debug_info['missing_terms'])}")
    print(f"Number of zero weights: {debug_info['zero_weights']}")
    print(f"Direct relationships found: {debug_info['direct_relationships']}")
    print(f"Indirect relationships found: {debug_info['indirect_relationships']}")

    return {
        'weighted_recall': np.mean(weighted_recalls) if weighted_recalls else 0.0,
        'weighted_precision': np.mean(weighted_precisions) if weighted_precisions else 0.0
    }


# ==== Ontology-based metrics ====
def analyze_hop_distribution(candidates: List[str],
                             target: str,
                             k: int,
                             hpo_relationships: Dict) -> Dict[str, int]:
    """
    Analyze the distribution of hops for all candidates in relation to the target
    Returns a dictionary with relationship types and their counts
    """
    if not candidates or k == 0 or target not in hpo_relationships:
        return {}

    candidates_at_k = candidates[:k]
    distribution = {
        'exact_match': 0,
        'ancestor': defaultdict(int),
        'descendant': defaultdict(int),
        'cousin': defaultdict(int),
        'no_path': 0
    }

    for candidate in candidates_at_k:
        if candidate not in hpo_relationships:
            continue

        # Exact match
        if candidate == target:
            distribution['exact_match'] += 1

        # Ancestor relationship
        elif candidate in hpo_relationships[target]['ancestors']:
            hops = hpo_relationships[target]['hops'][candidate]
            distribution['ancestor'][hops] += 1

        # Descendant relationship
        elif target in hpo_relationships[candidate]['ancestors']:
            hops = hpo_relationships[candidate]['hops'][target]
            distribution['descendant'][hops] += 1

        # Cousin relationship (common ancestor)
        else:
            common_ancestors = (set(hpo_relationships[candidate]['ancestors']) &
                                set(hpo_relationships[target]['ancestors']))
            if common_ancestors:
                min_total_hops = float('inf')
                for ancestor in common_ancestors:
                    total_hops = (hpo_relationships[candidate]['hops'][ancestor] +
                                  hpo_relationships[target]['hops'][ancestor])
                    min_total_hops = min(min_total_hops, total_hops)
                if min_total_hops != float('inf'):
                    distribution['cousin'][min_total_hops] += 1
            else:
                distribution['no_path'] += 1

    return distribution


def compute_branch_overlap_at_k(candidates: List[str],
                                target: str,
                                k: int,
                                hpo_relationships: Dict) -> float:
    """
    Compute proportion of candidates in same branch as target
    """
    if not candidates or k == 0 or target not in hpo_relationships:
        return 0.0

    target_ancestors = set(hpo_relationships[target]['ancestors'])
    target_ancestors.add(target)  # Include the target term itself

    candidates_at_k = candidates[:k]
    valid_candidates = [c for c in candidates_at_k if c in hpo_relationships]

    if not valid_candidates:
        return 0.0

    branch_matches = sum(1 for c in valid_candidates if
                         c == target or  # Exact match
                         c in target_ancestors or  # Candidate is ancestor
                         target in hpo_relationships[c]['ancestors'] or  # Target is ancestor
                         set(hpo_relationships[c]['ancestors']) & target_ancestors)  # Share ancestors

    coverage = branch_matches / min(k, len(valid_candidates))

    return coverage


# ==== Retrieval metrics ====
def compute_miss_rate(predicted_lists: List[List[str]],
                     ground_truth: List[str],
                     k: int,
                     hpo_relationships: Dict) -> float:
    """Compute miss rate (FN/(TP + FN))"""
    TPs = 0
    FNs = 0

    for preds, truth in zip(predicted_lists, ground_truth):
        if not truth or pd.isna(truth):
            continue

        # Create truth set with target and its direct parents
        truth_set = {truth}
        if truth in hpo_relationships:
            for ancestor, hops in hpo_relationships[truth]['hops'].items():
                if hops == 1:  # Only include direct parents
                    truth_set.add(ancestor)

        top_k_preds = preds[:k]
        if any(pred in truth_set for pred in top_k_preds):
            TPs += 1
        else:
            FNs += 1

    miss_rate = FNs / (TPs + FNs) if (TPs + FNs) > 0 else 1.0
    return miss_rate





#=======================
# PLOT
#=======================



def plot_relationship_distributions(results_df: pd.DataFrame,
                                    k_value: int,
                                    target_dataset: str,
                                    output_dir: str,
                                    params: Dict):
    """
    Create plots showing distribution of different types of relationships as percentages
    """
    plt.style.use('ggplot')

    # Prepare data
    models = results_df['model'].unique()
    relationship_types = ['exact_match', 'ancestor', 'descendant', 'cousin', 'no_path']

    # Create subplots for each relationship type
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for i, rel_type in enumerate(relationship_types):
        data = []
        for model in models:
            model_data = results_df[results_df['model'] == model]

            # Calculate total relationships for percentage
            total_rels = model_data['exact_match_dist'].values[0]  # Start with exact matches
            for rt in ['ancestor', 'descendant', 'cousin']:
                dist = model_data[f'{rt}_dist'].values[0]
                total_rels += sum(dist.values())
            total_rels += model_data['no_path_dist'].values[0]  # Add no path cases

            if rel_type in ['ancestor', 'descendant', 'cousin']:
                dist = model_data[f'{rel_type}_dist'].values[0]
                for hops, count in dist.items():
                    data.append({
                        'Model': model,
                        'Hops': hops,
                        'Percentage': (count / total_rels) * 100
                    })
            else:  # exact_match or no_path
                count = model_data[f'{rel_type}_dist'].values[0]
                data.append({
                    'Model': model,
                    'Percentage': (count / total_rels) * 100
                })

        df = pd.DataFrame(data)
        if rel_type in ['ancestor', 'descendant', 'cousin']:
            sns.barplot(data=df, x='Hops', y='Percentage', hue='Model', ax=axes[i])
            axes[i].set_title(f'{rel_type.capitalize()} Relationships (%)')
            axes[i].set_ylabel('Percentage of Total Relationships')
        else:
            sns.barplot(data=df, x='Model', y='Percentage', ax=axes[i])
            axes[i].set_title(f'{rel_type.replace("_", " ").capitalize()} (%)')
            axes[i].set_ylabel('Percentage of Total Relationships')

        axes[i].tick_params(axis='x', rotation=45)

    # Remove extra subplot
    axes[-1].remove()

    plt.suptitle(f'Relationship Distribution Analysis (k={k_value})', y=1.02)
    plt.tight_layout()
    # Format the output filename
    output_filename = get_formatted_filename(
        base_name=f'relationship_distribution_k{k_value}',
        target_dataset=params['target_dataset'],
        top_k=params['top_k'],
        alpha=params['alpha'],
        threshold=params['threshold'],
        euc_model_rerank=params['euc_model_rerank'],
        hit_model_rerank=params['hit_model_rerank'],
        normalization_mode=params['normalization_mode']
    )

    plt.savefig(output_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_relationship_heatmaps(results_df: pd.DataFrame,
                               k_values: List[int],
                               target_dataset: str,
                               output_dir: str,
                               params: Dict):
    """
    Create separate heatmaps for each relationship type
    """
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    axes = axes.flatten()

    relationship_types = {
        'Exact Matches': 'exact_match',
        'Ancestor Relationships': 'ancestor',
        'Descendant Relationships': 'descendant',
        'Cousin Relationships': 'cousin'
    }

    # Collect all data first
    all_data = []
    for k in k_values:
        # Get results for this k value
        distributions = []
        for model_name, candidates in models_data:
            model_dist = {
                'exact_match': 0,
                'ancestor': defaultdict(int),
                'descendant': defaultdict(int),
                'cousin': defaultdict(int),
                'no_path': 0
            }

            for cands, target in zip(candidates, target_values):
                if pd.isna(target):
                    continue
                dist = analyze_hop_distribution(cands[:k], target, k, hpo_relationships)
                for rel_type, counts in dist.items():
                    if rel_type in ['exact_match', 'no_path']:
                        model_dist[rel_type] += counts
                    else:
                        for hops, count in counts.items():
                            model_dist[rel_type][hops] += count

            # Add to data collection
            for rel_type in relationship_types.values():
                if rel_type == 'exact_match':
                    count = model_dist[rel_type]
                else:
                    count = sum(model_dist[rel_type].values())
                all_data.append({
                    'k': k,
                    'model': model_name,
                    'relationship': rel_type,
                    'count': count
                })

    # Create heatmaps
    for idx, (title, rel_type) in enumerate(relationship_types.items()):
        # Filter data for this relationship type
        rel_data = [d for d in all_data if d['relationship'] == rel_type]
        df_pivot = pd.DataFrame(rel_data).pivot(index='model', columns='k', values='count')

        sns.heatmap(df_pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[idx])
        axes[idx].set_title(title)

    # Format the output filename
    output_filename = get_formatted_filename(
        base_name='relationship_heatmaps',
        target_dataset=params['target_dataset'],
        top_k=params['top_k'],
        alpha=params['alpha'],
        threshold=params['threshold'],
        euc_model_rerank=params['euc_model_rerank'],
        hit_model_rerank=params['hit_model_rerank'],
        normalization_mode=params['normalization_mode']
    )

    plt.savefig(output_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.close()


def compute_average_hops(predicted_lists: List[List[str]],
                         ground_truth: List[str],
                         k: int,
                         hpo_relationships: Dict) -> float:
    """Compute average number of hops between predictions and ground truth"""
    all_hops = []

    for preds, truth in zip(predicted_lists, ground_truth):
        if not truth or pd.isna(truth):
            continue

        top_k_preds = preds[:k]
        for pred in top_k_preds:
            if pred == truth:
                # all_hops.append(0)  # Exact match
                continue
            elif truth in hpo_relationships and pred in hpo_relationships:
                # Check if pred is in truth's relationships
                if pred in hpo_relationships[truth]['hops']:
                    all_hops.append(hpo_relationships[truth]['hops'][pred])
                # Check if truth is in pred's relationships
                elif truth in hpo_relationships[pred]['hops']:
                    all_hops.append(hpo_relationships[pred]['hops'][truth])
                # Could add else condition for no path found

    return np.mean(all_hops) if all_hops else float('nan')


def plot_average_hops_heatmap(models_data: List[Tuple[str, List[List[str]]]],
                              target_values: List[str],
                              k_values: List[int],
                              hpo_relationships: Dict,
                              output_dir: str,
                              params: Dict):
    """Create heatmap of average hops for each model and k value"""
    # Collect data
    data = []
    for k in k_values:
        print(f"\nComputing average hops for k={k}")
        for model_name, candidates in models_data:
            print(f"Processing {model_name}...")
            avg_hops = compute_average_hops(candidates, target_values, k, hpo_relationships)
            data.append({
                'k': k,
                'model': model_name,
                'avg_hops': avg_hops
            })

    # Create pivot table
    df = pd.DataFrame(data)
    pivot_df = df.pivot(index='model', columns='k', values='avg_hops')

    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlOrRd_r')  # _r for reversed (darker = better/lower hops)
    plt.title('Average Number of Hops between Predictions and Ground Truth')
    plt.xlabel('k')
    plt.ylabel('Model')
    plt.tight_layout()

    # Format the output filename
    output_filename = get_formatted_filename(
        base_name='average_hops_heatmap',
        target_dataset=params['target_dataset'],
        top_k=params['top_k'],
        alpha=params['alpha'],
        threshold=params['threshold'],
        euc_model_rerank=params['euc_model_rerank'],
        hit_model_rerank=params['hit_model_rerank'],
        normalization_mode=params['normalization_mode']
    )
    plt.savefig(output_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    return df


def compute_relationship_specific_hops(predicted_lists: List[List[str]],
                                       ground_truth: List[str],
                                       k: int,
                                       hpo_relationships: Dict) -> Dict[str, float]:
    """Compute average hops separately for ancestors, descendants, and cousins"""
    ancestor_hops = []
    descendant_hops = []
    cousin_hops = []

    for preds, truth in zip(predicted_lists, ground_truth):
        if not truth or pd.isna(truth):
            continue

        top_k_preds = preds[:k]
        for pred in top_k_preds:
            if pred == truth:
                continue  # Skip exact matches
            elif truth in hpo_relationships and pred in hpo_relationships:
                # Check ancestor relationship (pred is ancestor of truth)
                if pred in hpo_relationships[truth]['hops']:
                    ancestor_hops.append(hpo_relationships[truth]['hops'][pred])
                # Check descendant relationship (truth is ancestor of pred)
                elif truth in hpo_relationships[pred]['hops']:
                    descendant_hops.append(hpo_relationships[pred]['hops'][truth])
                # Check cousin relationship (share common ancestor)
                else:
                    # Find shortest path through common ancestor
                    min_cousin_hops = float('inf')
                    for ancestor in hpo_relationships[truth]['hops']:
                        if ancestor in hpo_relationships[pred]['hops']:
                            total_hops = (hpo_relationships[truth]['hops'][ancestor] +
                                          hpo_relationships[pred]['hops'][ancestor])
                            min_cousin_hops = min(min_cousin_hops, total_hops)
                    if min_cousin_hops != float('inf'):
                        cousin_hops.append(min_cousin_hops)

    return {
        'ancestor': np.mean(ancestor_hops) if ancestor_hops else float('nan'),
        'descendant': np.mean(descendant_hops) if descendant_hops else float('nan'),
        'cousin': np.mean(cousin_hops) if cousin_hops else float('nan')
    }


def plot_relationship_hops_heatmaps(models_data: List[Tuple[str, List[List[str]]]],
                                    target_values: List[str],
                                    k_values: List[int],
                                    hpo_relationships: Dict,
                                    output_dir: str,
                                    params: Dict):
    """Create separate heatmaps for average hops by relationship type"""
    # Collect data
    data = []
    for k in k_values:
        print(f"\nComputing relationship-specific hops for k={k}")
        for model_name, candidates in models_data:
            print(f"Processing {model_name}...")
            avg_hops = compute_relationship_specific_hops(candidates, target_values, k, hpo_relationships)
            data.append({
                'k': k,
                'model': model_name,
                **avg_hops  # Unpack the dictionary of relationship-specific averages
            })

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Create heatmaps for each relationship type
    relationship_types = {
        'ancestor': 'Ancestor Relationships',
        'descendant': 'Descendant Relationships',
        'cousin': 'Cousin Relationships'
    }

    df = pd.DataFrame(data)
    for idx, (rel_type, title) in enumerate(relationship_types.items()):
        pivot_df = df.pivot(index='model', columns='k', values=rel_type)

        sns.heatmap(pivot_df,
                    annot=True,
                    fmt='.2f',
                    cmap='YlOrRd_r',  # _r for reversed (darker = better/lower hops)
                    ax=axes[idx])
        axes[idx].set_title(f'Average Hops for {title}')
        axes[idx].set_xlabel('k')
        axes[idx].set_ylabel('Model')

    plt.suptitle('Average Number of Hops by Relationship Type', y=1.05)
    plt.tight_layout()
    # Format the output filename
    output_filename = get_formatted_filename(
        base_name='relationship_hops_heatmaps',
        target_dataset=params['target_dataset'],
        top_k=params['top_k'],
        alpha=params['alpha'],
        threshold=params['threshold'],
        euc_model_rerank=params['euc_model_rerank'],
        hit_model_rerank=params['hit_model_rerank'],
        normalization_mode=params['normalization_mode']
    )
    plt.savefig(output_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.close()

    return df


def plot_branch_coverage_heatmap(hierarchical_results: pd.DataFrame,
                                 target_dataset: str,
                                 output_dir: str,
                                 params: Dict):
    """
    Create a heatmap of branch coverage for each model and k value
    """
    # Create pivot table
    pivot_df = hierarchical_results.pivot(index='model', columns='k', values='branch_coverage')

    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlGnBu', vmin=0, vmax=1)
    plt.title('Branch Coverage by Model and k')
    plt.xlabel('k')
    plt.ylabel('Model')
    plt.tight_layout()

    # Format the output filename
    output_filename = get_formatted_filename(
        base_name='branch_coverage_heatmap',
        target_dataset=params['target_dataset'],
        top_k=params['top_k'],
        alpha=params['alpha'],
        threshold=params['threshold'],
        euc_model_rerank=params['euc_model_rerank'],
        hit_model_rerank=params['hit_model_rerank'],
        normalization_mode=params['normalization_mode']
    )
    plt.savefig(output_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_combined_hops_coverage_metrics(hierarchical_results: pd.DataFrame,
                                        models_order: List[str],
                                        model_colors: Dict[str, str],
                                        model_markers: Dict[str, str],
                                        k_values: List[int] = None,
                                        output_dir: str = None,
                                        target_dataset: str = "dataset",
                                        figsize: Tuple[int, int] = (8, 4),
                                        dpi: int = 300,
                                        linewidth: float = 0.8,
                                        markersize: float = 3) -> None:
    """
    Create a compact, combined plot showing average hops and branch coverage metrics.

    Args:
        hierarchical_results: DataFrame containing evaluation results
        models_order: List of model names in the desired order
        model_colors: Dictionary mapping model names to colors
        model_markers: Dictionary mapping model names to marker styles
        k_values: List of k values to include (if None, use all k values in the results)
        output_dir: Directory to save the plot
        target_dataset: Name of the dataset (for file naming)
        figsize: Figure size as (width, height) in inches
        dpi: Resolution for the saved figure
        linewidth: Width of the lines in the plot (smaller values = thinner lines)
        markersize: Size of the markers (smaller values = smaller markers)
    """
    # Filter k values if specified
    if k_values:
        filtered_results = hierarchical_results[hierarchical_results['k'].isin(k_values)]
    else:
        filtered_results = hierarchical_results
        k_values = sorted(hierarchical_results['k'].unique())

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    # Set common style
    plt.style.use('ggplot')

    # Plot Average Hops (first subplot)
    ax1 = axes[0]
    for model in models_order:
        model_data = filtered_results[filtered_results['model'] == model]
        if not model_data.empty:
            ax1.plot(model_data['k'], model_data['average_hops'],
                     marker=model_markers.get(model, 'o'),
                     color=model_colors.get(model, None),
                     linewidth=linewidth,
                     markersize=markersize,
                     label=model if model == models_order[0] else None)  # Only add label for first subplot

    ax1.set_title('Average Hops')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Hops')
    # Set y-axis limits based on data range
    y_min = filtered_results['average_hops'].min() * 0.9
    y_max = filtered_results['average_hops'].max() * 1.1
    ax1.set_ylim(y_min, y_max)
    ax1.grid(True, alpha=0.3)

    # Plot Branch Coverage (second subplot)
    ax2 = axes[1]
    for model in models_order:
        model_data = filtered_results[filtered_results['model'] == model]
        if not model_data.empty:
            ax2.plot(model_data['k'], model_data['branch_coverage'],
                     marker=model_markers.get(model, 'o'),
                     color=model_colors.get(model, None),
                     linewidth=linewidth,
                     markersize=markersize)

    ax2.set_title('Branch Coverage')
    ax2.set_xlabel('k')
    ax2.set_ylabel('Coverage')
    # Set y-axis limits based on data range
    y_min = filtered_results['branch_coverage'].min() * 0.9
    y_max = filtered_results['branch_coverage'].max() * 1.1
    ax2.set_ylim(y_min, y_max)
    ax2.grid(True, alpha=0.3)

    # Create a single legend for all subplots
    handles, labels = [], []
    for model in models_order:
        handles.append(plt.Line2D([0], [0], color=model_colors.get(model, None),
                                  marker=model_markers.get(model, 'o'),
                                  linewidth=linewidth,
                                  markersize=markersize,
                                  label=model))

    # Place legend below the subplots
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.15),
               ncol=len(handles), frameon=True, fontsize='small')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for the legend

    # Add a common title
    # fig.suptitle('Ontological Structure Metrics', y=1.02)

    # Save the figure
    if output_dir:
        output_filename = f"{target_dataset}_combined_hops_coverage_metrics.png"
        plt.savefig(Path(output_dir) / output_filename, dpi=dpi, bbox_inches='tight')
        print(f"Combined hops and coverage metrics plot saved to {output_filename}")

    plt.close()


# ===========================
# MRR and NDCG
# ===========================
def compute_mrr(predicted_lists: List[List[str]],
                ground_truth: List[str],
                k: int) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) at k

    MRR = (1/|Q|) * sum(1/rank_i) where rank_i is the position of the first relevant item
    """
    reciprocal_ranks = []

    for preds, truth in zip(predicted_lists, ground_truth):
        if not truth or pd.isna(truth):
            continue

        # Get predictions up to k
        top_k_preds = preds[:k]
        if not top_k_preds:
            continue

        # Find the rank of the first relevant item
        try:
            # +1 because ranks start at 1
            rank = top_k_preds.index(truth) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            # If the truth is not in the predictions, reciprocal rank is 0
            reciprocal_ranks.append(0.0)

    # Mean of reciprocal ranks
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def compute_weighted_mrr(predicted_lists: List[List[str]],
                         ground_truth: List[str],
                         k: int,
                         hpo_relationships: Dict,
                         alpha: float = 1.0,
                         beta: float = 1.0) -> float:
    """
    Compute Mean Reciprocal Rank (MRR) at k with relationship weights

    MRR = (1/|Q|) * sum(weight_i/rank_i) where rank_i is the position of each item
    """
    weighted_reciprocal_ranks = []

    for preds, truth in zip(predicted_lists, ground_truth):
        if not truth or pd.isna(truth):
            continue

        # Get predictions up to k
        top_k_preds = preds[:k]
        if not top_k_preds:
            continue

        # Find the weighted reciprocal rank
        max_weighted_rr = 0.0
        for i, pred in enumerate(top_k_preds):
            # Calculate relationship weight
            weight = compute_relationship_weight(pred, truth, hpo_relationships, alpha, beta)

            # Calculate weighted reciprocal rank for this prediction
            if weight > 0:
                # +1 because ranks start at 1
                weighted_rr = weight / (i + 1)
                max_weighted_rr = max(max_weighted_rr, weighted_rr)

        weighted_reciprocal_ranks.append(max_weighted_rr)

    # Mean of weighted reciprocal ranks
    return sum(weighted_reciprocal_ranks) / len(weighted_reciprocal_ranks) if weighted_reciprocal_ranks else 0.0


def compute_ndcg(predicted_lists: List[List[str]],
                 ground_truth: List[str],
                 k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at k
    """
    ndcg_scores = []

    for preds, truth in zip(predicted_lists, ground_truth):
        if not truth or pd.isna(truth):
            continue

        # Get predictions up to k
        top_k_preds = preds[:k]
        if not top_k_preds:
            continue

        # Calculate relevance scores (1 for relevant, 0 for not relevant)
        relevance = [1 if pred == truth else 0 for pred in top_k_preds]

        # Calculate DCG
        dcg = 0.0
        for i, rel in enumerate(relevance):
            if rel > 0:
                dcg += rel / np.log2(i + 2)

        # Calculate IDCG - for binary relevance with a single relevant item
        # IDCG is 1.0 (relevant item at position 1)
        idcg = 1.0

        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    # Mean NDCG
    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0


def compute_weighted_ndcg(predicted_lists: List[List[str]],
                          ground_truth: List[str],
                          k: int,
                          hpo_relationships: Dict,
                          alpha: float = 1.0,
                          beta: float = 1.0) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) at k with relationship weights

    NDCG@k = DCG@k / IDCG@k
    DCG@k = sum(rel_i / log2(i+1)) for i=1 to k
    IDCG@k = DCG@k for the ideal ranking
    """
    ndcg_scores = []

    for preds, truth in zip(predicted_lists, ground_truth):
        if not truth or pd.isna(truth):
            continue

        # Get predictions up to k
        top_k_preds = preds[:k]
        if not top_k_preds:
            continue

        # Calculate relevance scores based on relationship weights
        relevance = [compute_relationship_weight(pred, truth, hpo_relationships, alpha, beta)
                     for pred in top_k_preds]

        # Calculate DCG
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance) if rel > 0)

        # Calculate IDCG (ideal DCG)
        # For weighted relevance, the ideal ranking has items sorted by decreasing weight
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance) if rel > 0)

        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcg_scores.append(ndcg)

    # Mean NDCG
    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0


# ============================
# RECALL & PRECISION WITH DIRECT PARENTS
# ============================
def compute_hierarchical_corpus_recall_at_k(predicted_lists: List[List[str]],
                                            ground_truth: List[str],
                                            k: int,
                                            hpo_relationships: Dict) -> float:
    """Compute corpus-level Recall@K including direct parents"""
    recalls = []

    for preds, truth in zip(predicted_lists, ground_truth):
        if not truth or pd.isna(truth):
            continue

        # Create truth set with target and its direct parents
        truth_set = {truth}
        if truth in hpo_relationships:
            for ancestor, hops in hpo_relationships[truth]['hops'].items():
                # Include 1-hop ancestors
                if hops == 0:
                    truth_set.add(ancestor)

        top_k_preds = preds[:k]
        hits = 1 if any(pred in truth_set for pred in top_k_preds) else 0
        recalls.append(hits)

    return sum(recalls) / len(recalls) if recalls else 0.0


def compute_hierarchical_corpus_precision_at_k(predicted_lists: List[List[str]],
                                               ground_truth: List[str],
                                               k: int,
                                               hpo_relationships: Dict) -> float:
    """Compute corpus-level Precision@K including direct parents"""
    precisions = []

    for preds, truth in zip(predicted_lists, ground_truth):
        if not truth or pd.isna(truth):
            continue

        # Create truth set with target and its direct parents
        truth_set = {truth}
        if truth in hpo_relationships:
            for ancestor, hops in hpo_relationships[truth]['hops'].items():
                if hops == 0:  # Only include direct parents
                    truth_set.add(ancestor)

        top_k_preds = preds[:k]
        hits = sum(1 for pred in top_k_preds if pred in truth_set)
        precision = hits / len(top_k_preds) if top_k_preds else 0.0
        precisions.append(precision)

    return sum(precisions) / len(precisions) if precisions else 0.0


def compute_all_weighted_metrics(predicted_lists: List[List[str]],
                                 ground_truth: List[str],
                                 k: int,
                                 processed_hpo: Dict,
                                 alpha: float = 1.0,
                                 beta: float = 1.0) -> Dict[str, float]:
    """
    Compute all weighted metrics at once with caching for better performance
    """
    # Initialize cache
    weight_cache = {}

    weighted_recalls = []
    weighted_precisions = []
    weighted_mrrs = []
    weighted_ndcgs = []

    for preds, truth in zip(predicted_lists, ground_truth):
        if not truth or pd.isna(truth):
            continue

        top_k_preds = preds[:k]
        if not top_k_preds:
            continue

        # Compute weights for all predictions at once
        weights = [compute_relationship_weight(pred, truth, processed_hpo, alpha, beta, weight_cache)
                   for pred in top_k_preds]

        # Weighted recall: maximum weight among predictions
        max_weight = max(weights) if weights else 0.0
        weighted_recalls.append(max_weight)

        # Weighted precision: average weight of predictions
        avg_weight = sum(weights) / len(top_k_preds) if top_k_preds else 0.0
        weighted_precisions.append(avg_weight)

        # Weighted MRR: maximum weighted reciprocal rank
        max_weighted_rr = 0.0
        for i, weight in enumerate(weights):
            if weight > 0:
                weighted_rr = weight / (i + 1)
                max_weighted_rr = max(max_weighted_rr, weighted_rr)
        weighted_mrrs.append(max_weighted_rr)

        # Weighted NDCG
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(weights) if rel > 0)

        # Pour IDCG, on trie les poids par ordre décroissant
        ideal_weights = sorted(weights, reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_weights) if rel > 0)

        # Normalisation
        ndcg = dcg / idcg if idcg > 0 else 0.0

        # S'assurer que NDCG est entre 0 et 1
        ndcg = min(ndcg, 1.0)

        weighted_ndcgs.append(ndcg)

    return {
        'weighted_recall': np.mean(weighted_recalls) if weighted_recalls else 0.0,
        'weighted_precision': np.mean(weighted_precisions) if weighted_precisions else 0.0,
        'weighted_mrr': np.mean(weighted_mrrs) if weighted_mrrs else 0.0,
        'weighted_ndcg': np.mean(weighted_ndcgs) if weighted_ndcgs else 0.0
    }


def plot_combined_recall_metrics(hierarchical_results: pd.DataFrame,
                                 models_order: List[str],
                                 model_colors: Dict[str, str],
                                 model_markers: Dict[str, str],
                                 k_values: List[int] = None,
                                 output_dir: str = None,
                                 target_dataset: str = "dataset",
                                 figsize: Tuple[int, int] = (8, 6),
                                 dpi: int = 300) -> None:
    """
    Create a compact, multi-metric plot showing standard recall, weighted recall, and miss rate
    on the same figure with subplots.
    """
    # Filter k values if specified
    if k_values:
        filtered_results = hierarchical_results[hierarchical_results['k'].isin(k_values)]
    else:
        filtered_results = hierarchical_results
        k_values = sorted(hierarchical_results['k'].unique())

    # Preferred plotting order: models_order first, then any other models present
    present_models = filtered_results['model'].unique().tolist()
    ordered_models = [m for m in models_order if m in present_models] + [m for m in present_models if m not in models_order]

    # Helper: determine if a model is a baseline (single-candidate SOTA)
    baseline_models = {'PhenoBERT', 'PhenoBERT_LLM', 'String_matching'}
    def line_style_for(model):
        return ':' if model in baseline_models else '-'
    def marker_for(model):
        # No markers for baselines for cleaner horizontal lines
        return None if model in baseline_models else model_markers.get(model, 'o')

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)
    plt.style.use('ggplot')

    # Generic plotting helper to avoid repeating logic
    def plot_metric(ax, metric, title, y_label, ylim=None):
        for model in ordered_models:
            model_df = filtered_results[filtered_results['model'] == model][['k', metric]].dropna()
            if model_df.empty:
                continue  # nothing to plot for this metric/model
            x = model_df['k'].values
            y = model_df[metric].values
            # skip series that are all NaN or empty
            if len(y) == 0 or not np.isfinite(y).any():
                continue

            ax.plot(
                x, y,
                color=model_colors.get(model, None),
                linestyle=line_style_for(model),
                marker=marker_for(model),
                linewidth=1.2 if model in baseline_models else 0.8,
                markersize=3
            )

        ax.set_title(title)
        ax.set_xlabel('k')
        if y_label:
            ax.set_ylabel(y_label)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.2)

    # Plot the three metrics
    plot_metric(axes[0], 'hierarchical_recall', 'Standard Recall', 'Recall', ylim=(0.3, 0.85))
    plot_metric(axes[1], 'weighted_recall', 'Weighted Recall', '', ylim=(0.3, 0.85))
    plot_metric(axes[2], 'miss_rate', 'Miss Rate', '', ylim=(0.2, 0.7))

    # Build a single legend consistent with styles/colors
    handles = []
    for model in ordered_models:
        handles.append(plt.Line2D(
            [0], [0],
            color=model_colors.get(model, None),
            linestyle=line_style_for(model),
            marker=marker_for(model),
            linewidth=1.2 if model in baseline_models else 0.8,
            markersize=3,
            label=model
        ))

    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.12),
               ncol=min(len(handles), 6), frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for the legend

    if output_dir:
        output_filename = f"{target_dataset}_combined_recall_metrics.png"
        plt.savefig(Path(output_dir) / output_filename, dpi=dpi, bbox_inches='tight')
        print(f"Combined metrics plot saved to {output_filename}")

    plt.close()



#===================
# GLOBAL EVALUATION
#===================
def evaluate_hierarchical_performance(models_data: List[Tuple[str, List[List[str]]]],
                                      target_values: List[str],
                                      k_values: List[int],
                                      hpo_relationships: Dict,
                                      alpha: float = 1.0,
                                      beta: float = 1.0) -> pd.DataFrame:
    """
    Evaluate models considering hierarchical relationships
    """
    # Prétraiter les relations HPO pour optimiser les calculs
    processed_hpo = preprocess_hpo_relationships(hpo_relationships)

    results = []

    for k in k_values:
        print(f"\nEvaluating hierarchical metrics for k={k}")
        for model_name, candidates in models_data:
            print(f"Processing {model_name}...")

            # Standard metrics
            recall = compute_hierarchical_corpus_recall_at_k(candidates, target_values, k, hpo_relationships)
            precision = compute_hierarchical_corpus_precision_at_k(candidates, target_values, k, hpo_relationships)
            avg_hops = compute_average_hops(candidates, target_values, k, hpo_relationships)
            miss_rate = compute_miss_rate(candidates, target_values, k, hpo_relationships)

            # Standard ranking metrics
            mrr = compute_mrr(candidates, target_values, k)
            ndcg = compute_ndcg(candidates, target_values, k)

            # Compute all weighted metrics at once (optimized)
            weighted_metrics = compute_all_weighted_metrics(
                candidates, target_values, k, processed_hpo, alpha, beta)

            # Compute average branch coverage
            branch_coverages = []
            for cands, target in zip(candidates, target_values):
                if not pd.isna(target):
                    coverage = compute_branch_overlap_at_k(cands, target, k, hpo_relationships)
                    branch_coverages.append(coverage)
            avg_branch_coverage = np.mean(branch_coverages) if branch_coverages else 0.0

            results.append({
                'k': k,
                'model': model_name,
                'hierarchical_recall': recall,
                'hierarchical_precision': precision,
                'average_hops': avg_hops,
                'branch_coverage': avg_branch_coverage,
                'miss_rate': miss_rate,
                'mrr': mrr,
                'ndcg': ndcg,
                **weighted_metrics  # Unpack all weighted metrics
            })

    return pd.DataFrame(results)


def plot_hierarchical_metrics(hierarchical_results: pd.DataFrame,
                              target_dataset: str,
                              output_dir: str,
                              params: Dict):
    """
    Create plots for all metrics including miss rate, weighted metrics, MRR and NDCG
    """
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(2, 2)  # 4 rows, 2 columns

    # Define marker and line styles (add dashed styles for SOTA systems)
    marker_map = {
        'Euclidean': 'o',
        'Hyperbolic Rerank': 's',
        'Hybrid Rerank': 's',
        'PhenoBERT': '^',
        'PhenoBERT_LLM': '^',
        'String_matching': 'o'
    }

    line_style_map = {
        'Euclidean': '-',
        'Hyperbolic Rerank': '-',
        'Hybrid Rerank': '-',
        'PhenoBERT': ':',
        'PhenoBERT_LLM': ':',
        'String_matching': ':'
    }

    # Plot Recall
    ax1 = fig.add_subplot(gs[0, 0])
    for model in hierarchical_results['model'].unique():
        model_data = hierarchical_results[hierarchical_results['model'] == model]
        marker = marker_map.get(model, None)
        line_style = line_style_map.get(model, None)
        ax1.plot(model_data['k'], model_data['hierarchical_recall'],
                 marker=marker, label=model, linestyle=line_style)

    ax1.set_xlabel('k')
    ax1.set_ylabel('Recall')
    ax1.set_title('Recall@k')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(0.3, 1.0)

    # # Plot Precision
    # ax2 = fig.add_subplot(gs[0, 1])
    # for model in hierarchical_results['model'].unique():
    #     model_data = hierarchical_results[hierarchical_results['model'] == model]
    #     marker = marker_map.get(model, None)
    #     line_style = line_style_map.get(model, None)
    #     ax2.plot(model_data['k'], model_data['hierarchical_precision'],
    #              marker=marker, label=model, linestyle=line_style)
    # # Add SOTA reference lines
    # for model, value in sota_values['precision'].items():
    #     ax2.axhline(y=value, color='gray', linestyle=':', label=f'{model}')
    # ax2.set_xlabel('k')
    # ax2.set_ylabel('Precision')
    # ax2.set_title('Hierarchical Precision')
    # ax2.legend()
    # ax2.grid(True)

    # Plot Weighted Recall
    ax3 = fig.add_subplot(gs[0, 1])
    for model in hierarchical_results['model'].unique():
        model_data = hierarchical_results[hierarchical_results['model'] == model]
        marker = marker_map.get(model, None)
        line_style = line_style_map.get(model, None)
        ax3.plot(model_data['k'], model_data['weighted_recall'],
                 marker=marker, label=model, linestyle=line_style)

    ax3.set_xlabel('k')
    ax3.set_ylabel('Weighted Recall')
    ax3.set_title('Weighted Recall@k (HPO Relationship-based)')
    ax3.legend()
    ax3.grid(True)
    ax3.set_ylim(0.3, 1.0)

    # # Plot Weighted Precision
    # ax4 = fig.add_subplot(gs[1, 1])
    # for model in hierarchical_results['model'].unique():
    #     model_data = hierarchical_results[hierarchical_results['model'] == model]
    #     marker = marker_map.get(model, None)
    #     line_style = line_style_map.get(model, None)
    #     ax4.plot(model_data['k'], model_data['weighted_precision'],
    #              marker=marker, label=model, linestyle=line_style)
    # ax4.set_xlabel('k')
    # ax4.set_ylabel('Weighted Precision')
    # ax4.set_title('Weighted Precision (HPO Relationship-based)')
    # ax4.legend()
    # ax4.grid(True)

    # # Plot MRR
    # ax5 = fig.add_subplot(gs[2, 0])
    # for model in hierarchical_results['model'].unique():
    #     model_data = hierarchical_results[hierarchical_results['model'] == model]
    #     marker = marker_map.get(model, None)
    #     line_style = line_style_map.get(model, None)
    #     ax5.plot(model_data['k'], model_data['mrr'],
    #              marker=marker, label=model, linestyle=line_style)
    # ax5.set_xlabel('k')
    # ax5.set_ylabel('MRR')
    # ax5.set_title('Mean Reciprocal Rank (MRR)')
    # ax5.legend()
    # ax5.grid(True)

    # # Définir des limites raisonnables pour l'axe y
    # ax5.set_ylim(0, 1.0)

    # # Plot NDCG
    # ax6 = fig.add_subplot(gs[2, 1])
    # for model in hierarchical_results['model'].unique():
    #     model_data = hierarchical_results[hierarchical_results['model'] == model]
    #     marker = marker_map.get(model, None)
    #     line_style = line_style_map.get(model, None)
    #     ax6.plot(model_data['k'], model_data['ndcg'],
    #              marker=marker, label=model, linestyle=line_style)
    # ax6.set_xlabel('k')
    # ax6.set_ylabel('NDCG')
    # ax6.set_title('Normalized Discounted Cumulative Gain (NDCG)')
    # ax6.legend()
    # ax6.grid(True)
    #
    # # Définir des limites raisonnables pour l'axe y
    # ax6.set_ylim(0, 1.0)

    # Plot Miss Rate
    ax7 = fig.add_subplot(gs[1, :])  # Span both columns
    for model in hierarchical_results['model'].unique():
        model_data = hierarchical_results[hierarchical_results['model'] == model]
        marker = marker_map.get(model, None)
        line_style = line_style_map.get(model, None)
        ax7.plot(model_data['k'], model_data['miss_rate'],
                 marker=marker, label=model, linestyle=line_style)

    ax7.set_xlabel('k')
    ax7.set_ylabel('Miss Rate')
    ax7.set_title('Miss Rate')
    ax7.legend()
    ax7.grid(True)

    # Format the output filename
    output_filename = get_formatted_filename(
        base_name='hierarchical_metrics',
        target_dataset=params['target_dataset'],
        top_k=params['top_k'],
        alpha=params['alpha'],
        threshold=params['threshold'],
        euc_model_rerank=params['euc_model_rerank'],
        hit_model_rerank=params['hit_model_rerank'],
        normalization_mode=params['normalization_mode']
    )

    plt.suptitle(f'Hierarchical Evaluation Metrics', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.close()


# MRR NDCG
def plot_weighted_metrics_comparison(hierarchical_results: pd.DataFrame,
                                     target_dataset: str,
                                     output_dir: str,
                                     params: Dict):
    """
    Create a dedicated plot comparing weighted metrics across models
    """
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # Define marker mapping
    marker_map = {
        'Euclidean': 'o',
        'Hyperbolic': 'o',
        'Hyperbolic Rerank': 's',
        'Hybrid Rerank': 's',
        'HPO-ColBERT Rerank': '^'
    }

    line_style_map = {
        'Euclidean': '-',
        'Hyperbolic': '-',
        'Hyperbolic Rerank': '-',
        'Hybrid Rerank': '-',
        'HPO-ColBERT Rerank': '-'
    }

    # Plot MRR
    ax1 = axes[0, 0]
    for model in hierarchical_results['model'].unique():
        model_data = hierarchical_results[hierarchical_results['model'] == model]
        marker = marker_map.get(model, None)
        line_style = line_style_map.get(model, None)
        ax1.plot(model_data['k'], model_data['mrr'],
                 marker=marker, label=model, linestyle=line_style)
    ax1.set_xlabel('k')
    ax1.set_ylabel('MRR')
    ax1.set_title('Standard MRR')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(0, 1.0)

    # Plot Weighted MRR
    ax2 = axes[0, 1]
    for model in hierarchical_results['model'].unique():
        model_data = hierarchical_results[hierarchical_results['model'] == model]
        marker = marker_map.get(model, None)
        line_style = line_style_map.get(model, None)
        ax2.plot(model_data['k'], model_data['weighted_mrr'],
                 marker=marker, label=model, linestyle=line_style)
    ax2.set_xlabel('k')
    ax2.set_ylabel('Weighted MRR')
    ax2.set_title('Weighted MRR (HPO Relationship-based)')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(0, 1.0)

    # Plot NDCG
    ax3 = axes[1, 0]
    for model in hierarchical_results['model'].unique():
        model_data = hierarchical_results[hierarchical_results['model'] == model]
        marker = marker_map.get(model, None)
        line_style = line_style_map.get(model, None)
        ax3.plot(model_data['k'], model_data['ndcg'],
                 marker=marker, label=model, linestyle=line_style)
    ax3.set_xlabel('k')
    ax3.set_ylabel('NDCG')
    ax3.set_title('Standard NDCG')
    ax3.legend()
    ax3.grid(True)
    ax3.set_ylim(0, 1.0)

    # Plot Weighted NDCG
    ax4 = axes[1, 1]
    for model in hierarchical_results['model'].unique():
        model_data = hierarchical_results[hierarchical_results['model'] == model]
        marker = marker_map.get(model, None)
        line_style = line_style_map.get(model, None)
        ax4.plot(model_data['k'], model_data['weighted_ndcg'],
                 marker=marker, label=model, linestyle=line_style)
    ax4.set_xlabel('k')
    ax4.set_ylabel('Weighted NDCG')
    ax4.set_title('Weighted NDCG (HPO Relationship-based)')
    ax4.legend()
    ax4.grid(True)
    ax4.set_ylim(0, 1.0)

    # Format the output filename
    output_filename = get_formatted_filename(
        base_name='mrr_ndcg_comparison',
        target_dataset=params['target_dataset'],
        top_k=params['top_k'],
        alpha=params['alpha'],
        threshold=params['threshold'],
        euc_model_rerank=params['euc_model_rerank'],
        hit_model_rerank=params['hit_model_rerank'],
        normalization_mode=params['normalization_mode']
    )

    plt.suptitle(f'MRR and NDCG Comparison', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_recall_and_miss_rate(hierarchical_results: pd.DataFrame,
                             target_dataset: str,
                             output_dir: str,
                             params: Dict):
    """
    Create a dedicated plot comparing recall and miss rate across models
    """
    plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Define marker mapping
    marker_map = {
        'Euclidean': 'o',
        # 'Hyperbolic': 'o',
        'Hyperbolic Rerank': 's',
        'Hybrid Rerank': 's',
        # 'HPO-ColBERT Rerank': '^'
    }

    line_style_map = {
        'Euclidean': '-',
        # 'Hyperbolic': '-',
        'Hyperbolic Rerank': '-',
        'Hybrid Rerank': '-',
        # 'HPO-ColBERT Rerank': '-'
    }

    # Plot Recall
    ax1 = axes[0]
    for model in hierarchical_results['model'].unique():
        model_data = hierarchical_results[hierarchical_results['model'] == model]
        marker = marker_map.get(model, None)
        line_style = line_style_map.get(model, None)
        ax1.plot(model_data['k'], model_data['hierarchical_recall'],
                 marker=marker, label=model, linestyle=line_style)

    ax1.set_xlabel('k')
    ax1.set_ylabel('Recall')
    ax1.set_title('Hierarchical Recall@k')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(0, 1.0)

    # Plot Miss Rate
    ax2 = axes[1]
    for model in hierarchical_results['model'].unique():
        model_data = hierarchical_results[hierarchical_results['model'] == model]
        marker = marker_map.get(model, None)
        line_style = line_style_map.get(model, None)
        ax2.plot(model_data['k'], model_data['miss_rate'],
                 marker=marker, label=model, linestyle=line_style)

    ax2.set_xlabel('k')
    ax2.set_ylabel('Miss Rate')
    ax2.set_title('Miss Rate@k')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(0, 1.0)

    # Format the output filename
    output_filename = get_formatted_filename(
        base_name='recall_miss_rate_comparison',
        target_dataset=params['target_dataset'],
        top_k=params['top_k'],
        alpha=params['alpha'],
        threshold=params['threshold'],
        euc_model_rerank=params['euc_model_rerank'],
        hit_model_rerank=params['hit_model_rerank'],
        normalization_mode=params['normalization_mode']
    )

    plt.suptitle(f'Recall and Miss Rate Comparison', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_relative_gain(hierarchical_results: pd.DataFrame,
                            table_k_values: List[int]):
    """
    Calculate and print the relative gain of weighted metrics over standard metrics
    """
    print("\n===== Relative Gain of Weighted Metrics =====")

    metric_pairs = [
        ('mrr', 'weighted_mrr', 'MRR'),
        ('ndcg', 'weighted_ndcg', 'NDCG'),
        ('hierarchical_recall', 'weighted_recall', 'Recall'),
        ('hierarchical_precision', 'weighted_precision', 'Precision')
    ]

    for k in table_k_values:
        print(f"\nRelative Gain at k={k}:")
        k_results = hierarchical_results[hierarchical_results['k'] == k]

        for model in k_results['model'].unique():
            print(f"\n{model}:")
            model_data = k_results[k_results['model'] == model]

            for std_metric, weighted_metric, metric_name in metric_pairs:
                std_value = model_data[std_metric].values[0]
                weighted_value = model_data[weighted_metric].values[0]

                if std_value > 0:
                    gain = ((weighted_value - std_value) / std_value) * 100
                    print(f"  {metric_name}: {gain:.2f}%")
                else:
                    print(f"  {metric_name}: N/A (standard metric is zero)")


#===========================
# WEIGHTS VIZ
#===========================
def plot_weight_distribution_by_position(models_data, target_values, k, processed_hpo, output_dir, params):
    """Visualize the distribution of weights by position in the ranked lists"""
    # Initialiser le cache pour les poids
    weight_cache = {}

    # Collecter les données
    position_weights = {model_name: [[] for _ in range(k)] for model_name, _ in models_data}

    for model_name, candidates in models_data:
        for cands, truth in zip(candidates, target_values):
            if pd.isna(truth):
                continue

            top_k_preds = cands[:k]
            if len(top_k_preds) < k:
                continue

            # Calculer les poids pour chaque position
            for i, pred in enumerate(top_k_preds):
                weight = compute_relationship_weight(
                    pred, truth, processed_hpo,
                    params['relationship_alpha'], params['relationship_beta'],
                    weight_cache
                )
                position_weights[model_name][i].append(weight)

    # Créer la heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    data = []

    for model_name in position_weights:
        for pos in range(k):
            weights = position_weights[model_name][pos]
            if weights:
                data.append({
                    'Model': model_name,
                    'Position': pos + 1,
                    'Average Weight': np.mean(weights)
                })

    df = pd.DataFrame(data)
    pivot_df = df.pivot(index='Model', columns='Position', values='Average Weight')

    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlGnBu')
    plt.title(f'Average Relationship Weight by Position (k={k})')
    plt.tight_layout()

    output_filename = get_formatted_filename(
        base_name=f'weight_distribution_by_position_k{k}',
        target_dataset=params['target_dataset'],
        top_k=params['top_k'],
        alpha=params['alpha'],
        threshold=params['threshold'],
        euc_model_rerank=params['euc_model_rerank'],
        hit_model_rerank=params['hit_model_rerank'],
        normalization_mode=params['normalization_mode']
    )
    plt.savefig(output_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_weight_histograms(models_data, target_values, k, processed_hpo, output_dir, params):
    """Create histograms of relationship weights for top-k candidates"""
    fig, axes = plt.subplots(len(models_data), 1, figsize=(12, 4 * len(models_data)))

    # Initialiser le cache pour les poids
    weight_cache = {}

    for i, (model_name, candidates) in enumerate(models_data):
        all_weights = []

        for cands, truth in zip(candidates, target_values):
            if pd.isna(truth):
                continue

            top_k_preds = cands[:k]
            weights = [compute_relationship_weight(
                pred, truth, processed_hpo,
                params['relationship_alpha'], params['relationship_beta'],
                weight_cache
            ) for pred in top_k_preds]

            all_weights.extend(weights)

        # Créer l'histogramme
        ax = axes[i] if len(models_data) > 1 else axes
        sns.histplot(all_weights, bins=20, kde=True, ax=ax)
        ax.set_title(f'{model_name} - Distribution of Relationship Weights (k={k})')
        ax.set_xlabel('Weight')
        ax.set_ylabel('Count')

    plt.tight_layout()

    output_filename = get_formatted_filename(
        base_name=f'weight_histograms_k{k}',
        target_dataset=params['target_dataset'],
        top_k=params['top_k'],
        alpha=params['alpha'],
        threshold=params['threshold'],
        euc_model_rerank=params['euc_model_rerank'],
        hit_model_rerank=params['hit_model_rerank'],
        normalization_mode=params['normalization_mode']
    )
    plt.savefig(output_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_relationship_types_by_position(models_data, target_values, k, processed_hpo, output_dir, params):
    """Visualize the distribution of relationship types by position"""
    relationship_types = ['exact_match', 'ancestor', 'descendant', 'cousin', 'no_path']

    # Utiliser la version originale des relations HPO pour l'analyse des types
    hpo_relationships = processed_hpo['original']

    # Collecter les données
    data = []

    for model_name, candidates in models_data:
        type_counts = {rel_type: [0] * k for rel_type in relationship_types}
        total_counts = [0] * k

        for cands, truth in zip(candidates, target_values):
            if pd.isna(truth):
                continue

            top_k_preds = cands[:k]
            if len(top_k_preds) < k:
                continue

            for i, pred in enumerate(top_k_preds):
                total_counts[i] += 1

                # Déterminer le type de relation
                if pred == truth:
                    type_counts['exact_match'][i] += 1
                elif truth in hpo_relationships and pred in hpo_relationships:
                    if pred in hpo_relationships[truth]['ancestors']:
                        type_counts['ancestor'][i] += 1
                    elif truth in hpo_relationships[pred]['ancestors']:
                        type_counts['descendant'][i] += 1
                    else:
                        # Vérifier s'il y a des ancêtres communs
                        common_ancestors = (
                                set(hpo_relationships[pred]['ancestors']) &
                                set(hpo_relationships[truth]['ancestors'])
                        )
                        if common_ancestors:
                            type_counts['cousin'][i] += 1
                        else:
                            type_counts['no_path'][i] += 1
                else:
                    type_counts['no_path'][i] += 1

        # Calculer les pourcentages
        for rel_type in relationship_types:
            for i in range(k):
                if total_counts[i] > 0:
                    percentage = (type_counts[rel_type][i] / total_counts[i]) * 100
                    data.append({
                        'Model': model_name,
                        'Position': i + 1,
                        'Relationship': rel_type,
                        'Percentage': percentage
                    })

    # Créer le graphique
    df = pd.DataFrame(data)

    for model_name in df['Model'].unique():
        fig, ax = plt.subplots(figsize=(14, 8))
        model_df = df[df['Model'] == model_name]

        pivot_df = model_df.pivot(
            index='Position',
            columns='Relationship',
            values='Percentage'
        )

        pivot_df.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        ax.set_title(f'{model_name} - Relationship Types by Position (k={k})')
        ax.set_xlabel('Position')
        ax.set_ylabel('Percentage (%)')
        ax.legend(title='Relationship Type')

        plt.tight_layout()

        output_filename = get_formatted_filename(
            base_name=f'relationship_types_{model_name.replace(" ", "_")}_k{k}',
            target_dataset=params['target_dataset'],
            top_k=params['top_k'],
            alpha=params['alpha'],
            threshold=params['threshold'],
            euc_model_rerank=params['euc_model_rerank'],
            hit_model_rerank=params['hit_model_rerank'],
            normalization_mode=params['normalization_mode']
        )
        plt.savefig(output_dir / output_filename, dpi=300, bbox_inches='tight')
        plt.close()


def plot_relationship_types_comparison(models_data, target_values, k, processed_hpo, output_dir, params):
    """
    Visualize the distribution of relationship types by position for all models in a single plot
    using a faceted heatmap approach, including a combined "close relationships" matrix
    """
    relationship_types = ['exact_match', 'ancestor', 'descendant', 'cousin', 'no_path', 'close_relationships']

    # Utiliser la version originale des relations HPO pour l'analyse des types
    hpo_relationships = processed_hpo['original']

    # Collecter les données
    data = []

    for model_name, candidates in models_data:
        for position in range(1, k + 1):
            type_counts = {rel_type: 0 for rel_type in relationship_types}
            total_count = 0

            for cands, truth in zip(candidates, target_values):
                if pd.isna(truth) or len(cands) < position:
                    continue

                pred = cands[position - 1]  # position is 1-indexed, list is 0-indexed
                total_count += 1

                # Déterminer le type de relation
                is_close = False

                if pred == truth:
                    type_counts['exact_match'] += 1
                    is_close = True  # Exact match est toujours "proche"
                elif truth in hpo_relationships and pred in hpo_relationships:
                    # Vérifier si c'est un ancêtre à 1 saut
                    if pred in hpo_relationships[truth]['ancestors']:
                        type_counts['ancestor'] += 1
                        # Vérifier si c'est un ancêtre à 1 saut
                        if hpo_relationships[truth]['hops'].get(pred, float('inf')) == 1:
                            is_close = True

                    # Vérifier si c'est un descendant à 1 saut
                    elif truth in hpo_relationships[pred]['ancestors']:
                        type_counts['descendant'] += 1
                        # Vérifier si c'est un descendant à 1 saut
                        if hpo_relationships[pred]['hops'].get(truth, float('inf')) == 1:
                            is_close = True

                    else:
                        # Vérifier s'il y a des ancêtres communs (cousins)
                        common_ancestors = (
                                set(hpo_relationships[pred]['ancestors']) &
                                set(hpo_relationships[truth]['ancestors'])
                        )
                        if common_ancestors:
                            type_counts['cousin'] += 1

                            # Vérifier si c'est un cousin à 2 sauts (1 saut vers l'ancêtre commun le plus proche)
                            min_total_hops = float('inf')
                            for ancestor in common_ancestors:
                                total_hops = (hpo_relationships[truth]['hops'].get(ancestor, float('inf')) +
                                              hpo_relationships[pred]['hops'].get(ancestor, float('inf')))
                                min_total_hops = min(min_total_hops, total_hops)

                            if min_total_hops <= 2:  # Cousins à 2 sauts ou moins
                                is_close = True
                        else:
                            type_counts['no_path'] += 1
                else:
                    type_counts['no_path'] += 1

                # Incrémenter le compteur des relations proches si applicable
                if is_close:
                    type_counts['close_relationships'] += 1

            # Calculer les pourcentages pour chaque type de relation
            for rel_type in relationship_types:
                percentage = (type_counts[rel_type] / total_count * 100) if total_count > 0 else 0
                data.append({
                    'Model': model_name,
                    'Position': position,
                    'Relationship': rel_type,
                    'Percentage': percentage
                })

    # Créer le DataFrame
    df = pd.DataFrame(data)

    # Créer un heatmap facetté par type de relation
    g = sns.FacetGrid(df, col="Relationship", col_wrap=3, height=4, aspect=1.2)

    # Mapper la fonction de heatmap
    def plot_heatmap(data, **kwargs):
        pivot = data.pivot(index="Model", columns="Position", values="Percentage")
        sns.heatmap(pivot, annot=True, fmt=".1f", cbar=False, **kwargs)

    # Appliquer la fonction à chaque facette
    g.map_dataframe(plot_heatmap, cmap="YlGnBu")

    # Ajuster les titres
    titles = {
        'exact_match': 'Exact Matches',
        'ancestor': 'Ancestor Relationships',
        'descendant': 'Descendant Relationships',
        'cousin': 'Cousin Relationships',
        'no_path': 'No Path Relationships',
        'close_relationships': 'Close Relationships (≤1-2 hops)'
    }

    for ax, rel_type in zip(g.axes.flat, relationship_types):
        ax.set_title(titles.get(rel_type, rel_type.replace('_', ' ').capitalize()))
        ax.set_xlabel('Position')
        ax.set_ylabel('Model')

    plt.suptitle(f'Relationship Types by Position and Model (k={k})', y=1.02)
    plt.tight_layout()

    # Format the output filename
    output_filename = get_formatted_filename(
        base_name=f'relationship_types_comparison_k{k}',
        target_dataset=params['target_dataset'],
        top_k=params['top_k'],
        alpha=params['alpha'],
        threshold=params['threshold'],
        euc_model_rerank=params['euc_model_rerank'],
        hit_model_rerank=params['hit_model_rerank'],
        normalization_mode=params['normalization_mode']
    )
    plt.savefig(output_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.close()



def plot_relationship_types_comparison_weights(models_data, target_values, k,
                                               processed_hpo, output_dir, params, weight_threshold=0.5):
    """
    Visualize the distribution of relationship types by position for all models in a single plot
    using a faceted heatmap approach, including a combined "semantically relevant" matrix
    based on relationship weights
    """
    relationship_types = ['exact_match', 'ancestor', 'descendant', 'cousin', 'semantically_relevant', 'no_path']

    # Extraire l'ordre des modèles de models_data
    models_order = [model_name for model_name, _ in models_data]

    # Utiliser la version originale des relations HPO pour l'analyse des types
    hpo_relationships = processed_hpo['original']

    # Initialiser le cache pour les poids
    weight_cache = {}

    # Collecter les données
    data = []

    for model_name, candidates in models_data:
        for position in range(1, k + 1):
            type_counts = {rel_type: 0 for rel_type in relationship_types}
            total_count = 0

            for cands, truth in zip(candidates, target_values):
                if pd.isna(truth) or len(cands) < position:
                    continue

                pred = cands[position - 1]  # position is 1-indexed, list is 0-indexed
                total_count += 1

                # Calculer le poids de la relation
                weight = compute_relationship_weight(
                    pred, truth, processed_hpo,
                    params['relationship_alpha'], params['relationship_beta'],
                    weight_cache
                )

                # Déterminer si la relation est sémantiquement pertinente selon le poids
                is_relevant = weight >= weight_threshold

                # Déterminer le type de relation
                if pred == truth:
                    type_counts['exact_match'] += 1
                elif truth in hpo_relationships and pred in hpo_relationships:
                    if pred in hpo_relationships[truth]['ancestors']:
                        type_counts['ancestor'] += 1
                    elif truth in hpo_relationships[pred]['ancestors']:
                        type_counts['descendant'] += 1
                    else:
                        # Vérifier s'il y a des ancêtres communs (cousins)
                        common_ancestors = (
                                set(hpo_relationships[pred]['ancestors']) &
                                set(hpo_relationships[truth]['ancestors'])
                        )
                        if common_ancestors:
                            type_counts['cousin'] += 1
                        else:
                            type_counts['no_path'] += 1
                else:
                    type_counts['no_path'] += 1

                # Incrémenter le compteur des relations sémantiquement pertinentes si applicable
                if is_relevant:
                    type_counts['semantically_relevant'] += 1

            # Calculer les pourcentages pour chaque type de relation
            for rel_type in relationship_types:
                percentage = (type_counts[rel_type] / total_count * 100) if total_count > 0 else 0
                data.append({
                    'Model': model_name,
                    'Position': position,
                    'Relationship': rel_type,
                    'Percentage': percentage
                })

    # Créer le DataFrame
    df = pd.DataFrame(data)

    # Forcer l'ordre des modèles
    df['Model'] = pd.Categorical(df['Model'], categories=models_order, ordered=True)

    # Créer un heatmap facetté par type de relation
    g = sns.FacetGrid(df, col="Relationship", col_wrap=3, height=4, aspect=1.2)

    # Mapper la fonction de heatmap
    def plot_heatmap(data, **kwargs):
        pivot = data.pivot(index="Model", columns="Position", values="Percentage")
        sns.heatmap(pivot, annot=True, fmt=".1f", cbar=False, **kwargs)

    # Appliquer la fonction à chaque facette
    g.map_dataframe(plot_heatmap, cmap="YlGnBu")

    # Ajuster les titres
    titles = {
        'exact_match': 'Exact Matches',
        'ancestor': 'Ancestor Relationships',
        'descendant': 'Descendant Relationships',
        'cousin': 'Cousin Relationships',
        'semantically_relevant': f'Close Relationships (score ≥ {weight_threshold})',
        'no_path': 'No Path Relationships'
    }

    for ax, rel_type in zip(g.axes.flat, relationship_types):
        ax.set_title(titles.get(rel_type, rel_type.replace('_', ' ').capitalize()))
        ax.set_xlabel('Position')
        ax.set_ylabel('Model')

    plt.suptitle(f'Relationship Types by Position and Model (k={k})', y=1.02)
    plt.tight_layout()

    # Format the output filename
    output_filename = get_formatted_filename(
        base_name=f'relationship_types_comparison_k{k}_threshold{weight_threshold}'.replace('.', ''),
        target_dataset=params['target_dataset'],
        top_k=params['top_k'],
        alpha=params['alpha'],
        threshold=params['threshold'],
        euc_model_rerank=params['euc_model_rerank'],
        hit_model_rerank=params['hit_model_rerank'],
        normalization_mode=params['normalization_mode']
    )
    plt.savefig(output_dir / output_filename, dpi=300, bbox_inches='tight')
    plt.close()


#==========================
# Mardown tables
#==========================
def generate_weighted_metrics_tables(hierarchical_results: pd.DataFrame,
                                     models_order: List[str],
                                     k_values: List[int] = None,
                                     output_dir: str = None,
                                     target_dataset: str = "dataset",
                                     single_column: bool = True) -> Tuple[str, str]:
    """
    Generate tables of weighted metrics (MRR and NDCG) in both Markdown and LaTeX formats.
    Best score in each column is highlighted in bold.

    Args:
        hierarchical_results: DataFrame containing evaluation results
        models_order: List of model names in the desired order
        k_values: List of k values to include in the table (if None, use all k values in the results)
        output_dir: Directory to save the LaTeX file (if None, only returns the string)
        target_dataset: Name of the dataset (for file naming)
        single_column: If True, generates a table optimized for a single column in a two-column paper

    Returns:
        Tuple of (markdown_table, latex_table) as strings
    """
    # Filter k values if specified
    if k_values:
        filtered_results = hierarchical_results[hierarchical_results['k'].isin(k_values)]
    else:
        filtered_results = hierarchical_results
        k_values = sorted(hierarchical_results['k'].unique())

    # Find best scores for each metric and k
    best_mrr = {}
    best_ndcg = {}
    for k in k_values:
        k_results = filtered_results[filtered_results['k'] == k]
        if not k_results.empty:
            best_mrr[k] = k_results['weighted_mrr'].max()
            best_ndcg[k] = k_results['weighted_ndcg'].max()

    # ======================
    # MARKDOWN TABLE
    # ======================
    markdown = "## Weighted Metrics by Model and k\n\n"

    # Create table for Weighted MRR
    markdown += "### Weighted MRR\n\n"
    markdown += "| Model | " + " | ".join([f"k={k}" for k in k_values]) + " |\n"
    markdown += "| --- | " + " | ".join(["---" for _ in k_values]) + " |\n"

    for model in models_order:
        model_results = filtered_results[filtered_results['model'] == model]
        row = f"| {model} | "
        for k in k_values:
            k_result = model_results[model_results['k'] == k]
            if not k_result.empty:
                mrr = k_result['weighted_mrr'].values[0]
                # Bold if best score
                if abs(mrr - best_mrr.get(k, -1)) < 1e-6:  # Use small epsilon for float comparison
                    row += f"**{mrr:.3f}** | "
                else:
                    row += f"{mrr:.3f} | "
            else:
                row += "N/A | "
        markdown += row + "\n"

    # Create table for Weighted NDCG
    markdown += "\n### Weighted NDCG\n\n"
    markdown += "| Model | " + " | ".join([f"k={k}" for k in k_values]) + " |\n"
    markdown += "| --- | " + " | ".join(["---" for _ in k_values]) + " |\n"

    for model in models_order:
        model_results = filtered_results[filtered_results['model'] == model]
        row = f"| {model} | "
        for k in k_values:
            k_result = model_results[model_results['k'] == k]
            if not k_result.empty:
                ndcg = k_result['weighted_ndcg'].values[0]
                # Bold if best score
                if abs(ndcg - best_ndcg.get(k, -1)) < 1e-6:  # Use small epsilon for float comparison
                    row += f"**{ndcg:.3f}** | "
                else:
                    row += f"{ndcg:.3f} | "
            else:
                row += "N/A | "
        markdown += row + "\n"

    # Add a combined table with both metrics and improved readability
    markdown += "\n### Combined Weighted Metrics\n\n"

    # Header row with k values
    markdown += "| Metric | Model | " + " | ".join([f"k={k}" for k in k_values]) + " |\n"
    markdown += "| --- | --- | " + " | ".join(["---" for _ in k_values]) + " |\n"

    # MRR rows
    for i, model in enumerate(models_order):
        model_results = filtered_results[filtered_results['model'] == model]

        # Add metric type in first column for first model only
        if i == 0:
            row = f"| **MRR** | {model} | "
        else:
            row = f"|  | {model} | "

        for k in k_values:
            k_result = model_results[model_results['k'] == k]
            if not k_result.empty:
                mrr = k_result['weighted_mrr'].values[0]
                # Bold if best score
                if abs(mrr - best_mrr.get(k, -1)) < 1e-6:
                    row += f"**{mrr:.3f}** | "
                else:
                    row += f"{mrr:.3f} | "
            else:
                row += "N/A | "
        markdown += row + "\n"

    # Add a separator row
    markdown += f"| | | " + " | ".join(["" for _ in k_values]) + " |\n"

    # NDCG rows
    for i, model in enumerate(models_order):
        model_results = filtered_results[filtered_results['model'] == model]

        # Add metric type in first column for first model only
        if i == 0:
            row = f"| **NDCG** | {model} | "
        else:
            row = f"|  | {model} | "

        for k in k_values:
            k_result = model_results[model_results['k'] == k]
            if not k_result.empty:
                ndcg = k_result['weighted_ndcg'].values[0]
                # Bold if best score
                if abs(ndcg - best_ndcg.get(k, -1)) < 1e-6:
                    row += f"**{ndcg:.3f}** | "
                else:
                    row += f"{ndcg:.3f} | "
            else:
                row += "N/A | "
        markdown += row + "\n"

        # ======================
        # LATEX TABLE
        # ======================
        if single_column:
            # Create a more compact table for single column in a two-column paper
            latex = "% Weighted Metrics Table (Single Column)\n"
            latex += "\\begin{table}[htbp]\n"
            latex += "\\centering\n"
            latex += "\\caption{Weighted Metrics by Model and k}\n"
            latex += "\\label{tab:weighted_metrics}\n"
            latex += "\\scriptsize\n"  # Use smaller font

            # Create two separate tables - one for MRR and one for NDCG
            # MRR Table
            latex += "\\begin{tabular}{l" + "c" * len(k_values) + "}\n"
            latex += "\\toprule\n"

            # Header row for MRR
            latex += "\\multicolumn{" + str(len(k_values) + 1) + "}{c}{\\textbf{Weighted MRR}} \\\\\n"
            latex += "Model & " + " & ".join([f"$k={k}$" for k in k_values]) + " \\\\\n"
            latex += "\\midrule\n"

            # MRR rows
            for model in models_order:
                model_results = filtered_results[filtered_results['model'] == model]
                row = f"{model} & "

                for k in k_values:
                    k_result = model_results[model_results['k'] == k]
                    if not k_result.empty:
                        mrr = k_result['weighted_mrr'].values[0]
                        # Bold if best score
                        if abs(mrr - best_mrr.get(k, -1)) < 1e-6:
                            row += f"\\textbf{{{mrr:.3f}}}"
                        else:
                            row += f"{mrr:.3f}"

                        # Add & except for last column
                        if k != k_values[-1]:
                            row += " & "
                    else:
                        row += "N/A"
                        # Add & except for last column
                        if k != k_values[-1]:
                            row += " & "

                row += " \\\\\n"
                latex += row

            # Close MRR table
            latex += "\\bottomrule\n"
            latex += "\\end{tabular}\n"

            # Add some vertical space
            latex += "\\vspace{0.5em}\n"

            # NDCG Table
            latex += "\\begin{tabular}{l" + "c" * len(k_values) + "}\n"
            latex += "\\toprule\n"

            # Header row for NDCG
            latex += "\\multicolumn{" + str(len(k_values) + 1) + "}{c}{\\textbf{Weighted NDCG}} \\\\\n"
            latex += "Model & " + " & ".join([f"$k={k}$" for k in k_values]) + " \\\\\n"
            latex += "\\midrule\n"

            # NDCG rows
            for model in models_order:
                model_results = filtered_results[filtered_results['model'] == model]
                row = f"{model} & "

                for k in k_values:
                    k_result = model_results[model_results['k'] == k]
                    if not k_result.empty:
                        ndcg = k_result['weighted_ndcg'].values[0]
                        # Bold if best score
                        if abs(ndcg - best_ndcg.get(k, -1)) < 1e-6:
                            row += f"\\textbf{{{ndcg:.3f}}}"
                        else:
                            row += f"{ndcg:.3f}"

                        # Add & except for last column
                        if k != k_values[-1]:
                            row += " & "
                    else:
                        row += "N/A"
                        # Add & except for last column
                        if k != k_values[-1]:
                            row += " & "

                row += " \\\\\n"
                latex += row

            # Close NDCG table
            latex += "\\bottomrule\n"
            latex += "\\end{tabular}\n"

            # Close the main table environment
            latex += "\\end{table}\n"
        else:
            # Original full-width table code
            latex = "% Weighted Metrics Table (Full Width)\n"
            latex += "\\begin{table}[htbp]\n"
            latex += "\\centering\n"
            latex += "\\caption{Weighted Metrics by Model and k}\n"
            latex += "\\label{tab:weighted_metrics}\n"
            latex += "\\resizebox{\\textwidth}{!}{\n"

            # Create the tabular environment
            latex += "\\begin{tabular}{ll" + "c" * len(k_values) + "}\n"
            latex += "\\toprule\n"

            # Header row
            latex += "Metric & Model & " + " & ".join([f"$k={k}$" for k in k_values]) + " \\\\\n"
            latex += "\\midrule\n"

            # MRR rows
            for i, model in enumerate(models_order):
                model_results = filtered_results[filtered_results['model'] == model]

                # Add metric type in first column for first model only
                if i == 0:
                    row = f"\\multirow{{{len(models_order)}}}{{*}}{{MRR}} & {model} & "
                else:
                    row = f" & {model} & "

                for k in k_values:
                    k_result = model_results[model_results['k'] == k]
                    if not k_result.empty:
                        mrr = k_result['weighted_mrr'].values[0]
                        # Bold if best score
                        if abs(mrr - best_mrr.get(k, -1)) < 1e-6:
                            row += f"\\textbf{{{mrr:.3f}}}"
                        else:
                            row += f"{mrr:.3f}"

                        # Add & except for last column
                        if k != k_values[-1]:
                            row += " & "
                    else:
                        row += "N/A"
                        # Add & except for last column
                        if k != k_values[-1]:
                            row += " & "

                row += " \\\\\n"
                latex += row

            # Add a separator
            latex += "\\midrule\n"

            # NDCG rows
            for i, model in enumerate(models_order):
                model_results = filtered_results[filtered_results['model'] == model]

                # Add metric type in first column for first model only
                if i == 0:
                    row = f"\\multirow{{{len(models_order)}}}{{*}}{{NDCG}} & {model} & "
                else:
                    row = f" & {model} & "

                for k in k_values:
                    k_result = model_results[model_results['k'] == k]
                    if not k_result.empty:
                        ndcg = k_result['weighted_ndcg'].values[0]
                        # Bold if best score
                        if abs(ndcg - best_ndcg.get(k, -1)) < 1e-6:
                            row += f"\\textbf{{{ndcg:.3f}}}"
                        else:
                            row += f"{ndcg:.3f}"

                        # Add & except for last column
                        if k != k_values[-1]:
                            row += " & "
                    else:
                        row += "N/A"
                        # Add & except for last column
                        if k != k_values[-1]:
                            row += " & "

                row += " \\\\\n"
                latex += row

            # Close the table
            latex += "\\bottomrule\n"
            latex += "\\end{tabular}\n"
            latex += "}\n"
            latex += "\\end{table}\n"

        # Save LaTeX to file if output_dir is provided
        if output_dir:
            latex_file = Path(
                output_dir) / f"{target_dataset}_weighted_metrics_table{'_single_column' if single_column else ''}.tex"
            with open(latex_file, 'w') as f:
                f.write(latex)
            print(f"LaTeX table saved to {latex_file}")

        return markdown, latex


def generate_baseline_comparison_table(hierarchical_results: pd.DataFrame,
                                       models_order: List[str],
                                       k_values: List[int]) -> str:
    """
    Generate a Markdown table comparing recall, weighted_recall, precision, miss_rate
    for all models (including SOTA baselines) across selected k.
    """
    cols = ["hierarchical_recall", "weighted_recall", "hierarchical_precision", "miss_rate"]
    col_titles = {
        "hierarchical_recall": "Recall",
        "weighted_recall": "Weighted Recall",
        "hierarchical_precision": "Precision",
        "miss_rate": "Miss Rate"
    }

    md = "## Recall / Weighted Recall / Precision / Miss Rate\n\n"
    for metric in cols:
        md += f"### {col_titles[metric]}\n\n"
        md += "| Model | " + " | ".join([f"k={k}" for k in k_values]) + " |\n"
        md += "| --- | " + " | ".join(["---" for _ in k_values]) + " |\n"

        # include all models present in the results (ensures SOTA present)
        # but preserve preferred order (ours first, then SOTA)
        present_models = list(dict.fromkeys(models_order + hierarchical_results["model"].unique().tolist()))
        for model in present_models:
            model_df = hierarchical_results[hierarchical_results["model"] == model]
            if model_df.empty:
                continue
            row = f"| {model} | "
            for k in k_values:
                k_df = model_df[model_df["k"] == k]
                if not k_df.empty and not pd.isna(k_df[metric].values[0]):
                    row += f"{k_df[metric].values[0]:.3f} | "
                else:
                    row += "N/A | "
            md += row + "\n"
        md += "\n"
    return md



# =========================
# RUN
# =========================


# Usage example:

if __name__ == "__main__":
    target_dataset = "chu50"
    top_k = 50
    alpha = 0.5
    threshold = 0.8
    euc_model = "base"
    hit_model = "syn"
    euc_model_rerank = "base"
    hit_model_rerank = "syn"
    normalization_mode = "global"

    # Parameters for relationship weighting
    relationship_alpha = 1.6  # Alpha for relationship weight calculation
    relationship_beta = 1.0  # Beta for relationship weight calculation

    params = {
        'target_dataset': "chu50",
        'top_k': 50,
        'alpha': 0.5,
        'threshold': 0.8,
        'euc_model': "base",
        'hit_model': "syn",
        'euc_model_rerank': "base",
        'hit_model_rerank': "syn",
        'normalization_mode': "global",
        'relationship_alpha': relationship_alpha,
        'relationship_beta': relationship_beta
    }
    import matplotlib.colors as mcolors

    palette = sns.color_palette("Paired", n_colors=8)
    print([mcolors.to_hex(color) for color in palette])

    # Load your results DataFrame and other necessary data
    input_path = project_root / f"output/output_rag_{target_dataset}_candidates_{top_k}_reranked_{alpha}_{threshold}_{euc_model_rerank}_{hit_model_rerank}_{normalization_mode}.csv"
    target_column = 'target_annotation'
    report_id_column = 'report_id'
    relationships_file = data_path / "hpo/hpo_relationships.json"

    print(f"Loading results from {input_path}...")
    df = pd.read_csv(input_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in CSV")
    if report_id_column not in df.columns:
        raise ValueError(f"Report ID column '{report_id_column}' not found in CSV")

    # Load HPO relationships
    print(f"Loading HPO relationships from {relationships_file}...")
    with open(relationships_file, 'r') as f:
        hpo_relationships = json.load(f)

    # Prétraiter les relations HPO pour optimiser les calculs
    processed_hpo = preprocess_hpo_relationships(hpo_relationships)

    # Find RAG candidate columns
    euclidean_col = [col for col in df.columns if 'euclidean' in col and 'top' in col and 'terms' not in col][0]
    # hyperbolic_col = [col for col in df.columns if 'hyperbolic' in col and 'top' in col and 'terms' not in col][0]
    hyperbolic_rerank_col = 'hyperbolic_reranking'
    hybrid_rerank_col = 'hybrid_reranking'
    # weighted_fallback_rerank_col = 'weighted_fallback_reranking'
    # ft_colbert_rerank_col = 'ft-colbert_reranking'
    # grpo_rerank_col = 'grpo_reranking'

    if not euclidean_col:
        raise ValueError("RAG candidate columns not found in CSV")

    # Prepare data
    target_values = df[target_column].tolist()

    # Prepare models data
    models_data = [
        ('Euclidean',
         [row[euclidean_col].split(',') if pd.notna(row[euclidean_col]) else [] for _, row in df.iterrows()]),
        # ('Hyperbolic',
        #  [row[hyperbolic_col].split(',') if pd.notna(row[hyperbolic_col]) else [] for _, row in df.iterrows()]),
        # ('HPO-ColBERT Rerank',
        #  [row[ft_colbert_rerank_col].split(',') if pd.notna(row[ft_colbert_rerank_col]) else [] for _, row in
        #   df.iterrows()]),
        # # ('GRPO Rerank',
        # #  [row[grpo_rerank_col].split(',') if pd.notna(row[grpo_rerank_col]) else [] for _, row in
        # #   df.iterrows()]),
        ('Hyperbolic Rerank',
         [row[hyperbolic_rerank_col].split(',') if pd.notna(row[hyperbolic_rerank_col]) else [] for _, row in
          df.iterrows()]),
        ('Hybrid Rerank',
         [row[hybrid_rerank_col].split(',') if pd.notna(row[hybrid_rerank_col]) else [] for _, row in df.iterrows()]),
    ]

    k_values = [1, 3, 5, 10, 15, 25, 30]
    output_dir = project_root / f"output/evaluation/{target_dataset}/axel"
    output_dir.mkdir(parents=True, exist_ok=True)

    for k in k_values:
        print(f"\nAnalyzing distributions for k={k}...")
        distributions = []
        for model_name, candidates in models_data:
            print(f"Processing {model_name}...")
            model_dist = {
                'exact_match': 0,
                'ancestor': defaultdict(int),
                'descendant': defaultdict(int),
                'cousin': defaultdict(int),
                'no_path': 0
            }

            for cands, target in zip(candidates, target_values):
                if pd.isna(target):
                    continue
                dist = analyze_hop_distribution(cands, target, k, hpo_relationships)
                for rel_type, counts in dist.items():
                    if rel_type in ['exact_match', 'no_path']:  # Simple counter types
                        model_dist[rel_type] += counts
                    else:  # Dictionary types (ancestor, descendant, cousin)
                        for hops, count in counts.items():
                            model_dist[rel_type][hops] += count

            distributions.append({
                'model': model_name,
                'exact_match_dist': model_dist['exact_match'],
                'ancestor_dist': dict(model_dist['ancestor']),
                'descendant_dist': dict(model_dist['descendant']),
                'cousin_dist': dict(model_dist['cousin']),
                'no_path_dist': model_dist['no_path']
            })

        results_df = pd.DataFrame(distributions)
        print(results_df)

        # Create visualizations
        print(f"Creating visualizations for k={k}...")
        plot_relationship_distributions(results_df, k, target_dataset, output_dir, params=params)

    # Create heatmaps and other visualizations
    plot_relationship_heatmaps(df, k_values, target_dataset, output_dir, params=params)

    # Add average hops heatmap
    average_hops = plot_average_hops_heatmap(models_data, target_values, k_values, hpo_relationships, output_dir,
                                             params=params)
    hops_results = plot_relationship_hops_heatmaps(
        models_data, target_values, k_values, hpo_relationships, output_dir, params=params)

    # Add hierarchical evaluation
    hierarchical_results = evaluate_hierarchical_performance(
        models_data, target_values, k_values, hpo_relationships,
        alpha=relationship_alpha, beta=relationship_beta)

    # --- Add SOTA baselines as horizontal lines (constant across k) ---
    sota_dataset_key = resolve_sota_dataset_key(target_dataset)
    sota_df = build_sota_dataframe(k_values, sota_dataset_key)
    if not sota_df.empty:
        hierarchical_results = pd.concat([hierarchical_results, sota_df], ignore_index=True)
        print(f"Added SOTA baselines for {sota_dataset_key}: {sota_df['model'].unique().tolist()}")


    # Plot hierarchical metrics separately
    plot_hierarchical_metrics(hierarchical_results, target_dataset, output_dir, params=params)

    # Plot dedicated weighted metrics comparison
    plot_weighted_metrics_comparison(hierarchical_results, target_dataset, output_dir, params=params)

    # Plot recall and miss rate comparison
    plot_recall_and_miss_rate(hierarchical_results, target_dataset, output_dir, params=params)

    # Define colors and markers for models
    model_colors = {
        'Euclidean': '#377EB8', # Blue
        # 'Hyperbolic': '#FF7F00',  # Orange
        # 'HPO-ColBERT Rerank': '#984EA3',  # Purple
        'Hyperbolic Rerank': '#E41A1C',  # Red
        'Hybrid Rerank': '#4DAF4A',  # Green
    }

    model_markers = {
        'Euclidean': 'o',
        # 'Hyperbolic': 's',
        # 'HPO-ColBERT Rerank': '^',
        'Hyperbolic Rerank': 'D',
        'Hybrid Rerank': 'P'
    }

    # Extend colors/markers for SOTA systems
    model_colors.update({
        'PhenoBERT': '#7f7f7f',         # grey
        'PhenoBERT_LLM': '#a6a6a6',     # lighter grey
        'String_matching': '#4d4d4d'    # darker grey
    })
    model_markers.update({
        'PhenoBERT': '^',
        'PhenoBERT_LLM': '^',
        'String_matching': 'o'
    })

    # Order: our models first, then SOTA
    sota_models = ['PhenoBERT', 'PhenoBERT_LLM', 'String_matching']
    models_order = [m for m, _ in models_data] + sota_models


    # Extract model order from models_data
    models_order = [model_name for model_name, _ in models_data]

    # Create compact combined plot
    plot_combined_recall_metrics(
        hierarchical_results,
        models_order=models_order,
        model_colors=model_colors,
        model_markers=model_markers,
        k_values=[1, 3, 5, 10, 15, 25, 30],
        output_dir=output_dir,
        target_dataset=target_dataset,
        figsize=(10, 4),  # Wider than tall for a compact layout
        dpi=300
    )

    # Create a table with standard and weighted metrics for k=1, 5, 10
    table_k_values = [1, 5, 10, 15]
    table_metrics = [
        'mrr', 'weighted_mrr',
        'ndcg', 'weighted_ndcg',
        'weighted_recall', 'weighted_precision'
    ]

    print("\n===== Standard and Weighted Metrics =====")
    for k in table_k_values:
        print(f"\nMetrics at k={k}:")
        k_results = hierarchical_results[hierarchical_results['k'] == k]
        for model in k_results['model'].unique():
            model_data = k_results[k_results['model'] == model]
            print(f"\n{model}:")
            for metric in table_metrics:
                value = model_data[metric].values[0]
                print(f"  {metric}: {value:.4f}")

    # Calculate and print relative gain
    calculate_relative_gain(hierarchical_results, table_k_values)

    # Plot branch coverage heatmap
    plot_branch_coverage_heatmap(hierarchical_results, target_dataset, output_dir, params=params)

    # Create compact combined plot for hops and branch coverage
    plot_combined_hops_coverage_metrics(
        hierarchical_results,
        models_order=models_order,
        model_colors=model_colors,
        model_markers=model_markers,
        k_values=[1, 3, 5, 10, 15, 25, 30],
        output_dir=output_dir,
        target_dataset=target_dataset,
        figsize=(8, 4),  # Adjust width/height ratio as needed
        dpi=300,
        linewidth=0.8,
        markersize=3
    )

    # Nouvelles visualisations pour la distribution des poids
    print("\nCréation des visualisations de distribution des poids...")
    detailed_k_values = [10, 15]  # Sélectionnez quelques valeurs de k spécifiques

    for k in detailed_k_values:
        print(f"Visualisation de la distribution des poids pour k={k}...")
        # Appel avec une valeur unique de k et processed_hpo
        plot_weight_distribution_by_position(models_data, target_values, k, processed_hpo, output_dir, params)
        plot_weight_histograms(models_data, target_values, k, processed_hpo, output_dir, params)
        plot_relationship_types_by_position(models_data, target_values, k, processed_hpo, output_dir, params)
        for threshold in [0.5, 0.7, 0.8, 0.9]:
            plot_relationship_types_comparison(models_data, target_values, k, processed_hpo, output_dir, params)
            plot_relationship_types_comparison_weights(models_data, target_values, k, processed_hpo, output_dir, params, weight_threshold=threshold)

    # Extract model order from models_data
    models_order = [model_name for model_name, _ in models_data]

    # Generate markdown table for weighted metrics
    selected_k_values = [1, 3, 5, 10, 15]  # Sélectionnez les valeurs de k qui vous intéressent
    markdown_table, latex_table = generate_weighted_metrics_tables(
        hierarchical_results,
        models_order=models_order,
        k_values=selected_k_values,
        output_dir=output_dir,
        target_dataset=target_dataset,
        single_column=True
    )

    # Print markdown to console
    print("\n" + markdown_table)

    # Save markdown to file
    markdown_file = output_dir / f"{target_dataset}_weighted_metrics_table.md"
    with open(markdown_file, 'w') as f:
        f.write(markdown_table)

    print(f"Markdown table saved to {markdown_file}")

