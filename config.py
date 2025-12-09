from pathlib import Path

project_root = Path(__file__).parent.resolve()
data_path = project_root / "data"
# data_path = "../data/"
models_path = project_root / "models/"

# Dataset
target_dataset = "id-68"  # ['id-68', 'chu50']

# RAG Settings
top_k = 50
mips = 'hyp-knn'  # mips strategy for hyperbolic RAG
euc_model = 'base'  # euclidean models type ['base', 'med']
hit_model = 'syn'  # hyperbolic models type ['syn', 'no-syn', 'snomed']

# Reranking Settings
alpha = 0.5  # hybrid factor
threshold = 0.8  # threshold for weighted fallback experiment
euc_model_rerank = "base"  # Model used for euclidean component reranking
hit_model_rerank = "syn"  # Model used for hyperbolic component reranking
normalization_mode = "global"  # Hyperbolic distances normalization mode ['none', 'global', 'local', 'hybrid']
hyperbolic_max_distance = 43.3674  # For normalization (value computed with utils.py script)

# ==== MODELS ====
euclidean_model = 'all-MiniLM-L12-v2'
# euclidean_model = 'abhinand/MedEmbed-small-v0.1'
hyperbolic_model = "HiT-all-MiniLM-L12-v2-hpo-hpo_datasets_syn_multi_random/final"
# hyperbolic_model = "HiT-all-MiniLM-L12-v2-hpo-hpo_datasets_multi_random/final"
# hyperbolic_model = "HiT-MiniLM-L12-SnomedCT-Hard"
late_interaction_model_base = 'colbert-ir/colbertv2.0'
late_interaction_model_ft = data_path / models_path / 'colbert_hpo_finetuned'
alternative_model = "bi-encoder/hpo_bi-encoder_from_hyperbolic"  # For additional Sentence Transformer model to experiment

# LLM Models
llm_span_detection = "openai/gpt-3.5-turbo"  # Spans detection
llm_model_hpo_sentences = "openai/gpt-4o-mini"  # For sentences generation from hpo and corresponding spans detection

# Useful files
HPO_RELATIONSHIPS_FILE = data_path / "hpo/hpo_relationships.json"
ORIGINAL_TERMS_FILE = data_path / 'hpo/original_terms.json'
SYNONYMS_MAPPING_FILE = data_path / 'hpo/synonym_mapping.json'
DISTANCE_MATRIX_FILE = data_path / "hpo/pairwise_distances_hyp_syn.npy"
# TERM_TO_INDEX_FILE = data_path / 'hpo/term_to_index_euclidean.json'  # mapping from HPO ID to row index in Euclidean distance matrix
TERM_TO_INDEX_FILE = data_path / "hpo/term_to_index_hyp_syn.json"  # mapping from HPO ID to row index in Hyperbolic distance matrix


# Late-interaction Training Settings
li_batch_size = 8
li_learning_rate = 1e-5
li_num_epochs = 2
li_max_query_length = 32
li_max_doc_length = 128
li_margin = 0.3
li_gradient_accumulation_steps = 2


