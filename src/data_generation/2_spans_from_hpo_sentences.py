'''This script extract relevant spans from each synthetic sentences'''
from hyperrag.config import *
import os
import json
import openai
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from hyperrag.src.cost_tracker import TokenCostTracker

key = os.environ.get("LLM_API_KEY")
base = os.environ.get("BASE_API_URL")

client = openai.OpenAI(api_key=key, base_url=base)

model = llm_model_hpo_sentences


def create_prompt(phenotype_data: Dict) -> str:
    """Create a prompt for the OpenAI API."""

    return f'''Extract the text span that best describes or indicates the phenotype "{phenotype_data['hpo_label']}" from this sentence.
        The span should:
            1. Capture the complete clinical observation or symptom
            2. Include relevant context that helps understand the phenotype
            3. Be specific to the actual medical condition
            4. Be concise while maintaining clinical accuracy
            5. Exclude patient names, temporal markers, or examination context
            6. Focus on the actual phenotypic finding

            Sentence: "{phenotype_data['sentence']}"

            Extract only the relevant span without any additional commentary.'''


def clean_span(span: str) -> str:
    """Clean and standardize the span format."""
    # Remove any surrounding whitespace
    span = span.strip()

    # Remove any escaped quotes
    span = span.replace('\\"', '')
    span = span.replace('"', '')

    return span


def process_sentence(client: openai.OpenAI, model: str, phenotype_data: Dict, tracker: TokenCostTracker) -> str:
    """Process a single sentence using the OpenAI API."""
    try:
        prompt = create_prompt(phenotype_data)

        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            # top_p=0.9,
            # frequency_penalty=1.2,
            messages=[{"role": "user", "content": prompt}],
        )
        output_text = response.choices[0].message.content.strip()

        # Track token usage
        tracker.add_usage(prompt, output_text)

        return clean_span(output_text)

    except Exception as e:
        print(f"Error processing sentence: {e}")
        return clean_span(phenotype_data['sentence'])


def process_entry(client: openai.OpenAI, model: str, entry: Dict, tracker: TokenCostTracker, max_workers: int = 5) -> Dict:
    """Process all sentences in an entry concurrently."""
    phenotype_data_list = [
        {
            'hpo_label': entry['hpo_label'],
            'hpo_id': entry['hpo_id'],
            'sentence': sentence
        }
        for sentence in entry['sentences']
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        process_func = partial(process_sentence, client, model, tracker=tracker)
        spans = list(executor.map(process_func, phenotype_data_list))

    # Get the token usage for this entry
    usage_stats = tracker.get_current_usage()

    entry['spans'] = spans
    entry['token_usage'] = usage_stats['total_stats']
    return entry


def main():
    # Configuration
    input_file = data_path / f'hpo/hpo-sentences_0_to_19483.json'
    output_file = data_path / 'hpo/hpo-sentences_0_to_19483_with_spans.json'

    # Initialize token tracker
    tracker = TokenCostTracker(
        model_name=model,
        log_file="hpo_token_usage_log.json"
    )

    # Load JSON data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    # Process all entries
    processed_data = []
    for entry in json_data:
        print(f"Processing {entry}")
        processed_entry = process_entry(client, model, entry, tracker)
        processed_data.append(processed_entry)

    # Get final usage statistics
    usage_summary = tracker.get_current_usage()

    # Prepare output data with usage summary
    output_data = {
        "entries": processed_data,
        "usage_summary": usage_summary
    }

    # Save results
    try:
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)

        # Save usage log
        tracker.save_usage_log()

        # Print summary
        print(f"\nResults saved to {output_file}")
        print(f"Processed {usage_summary['entries_processed']} entries")
        print(f"Total Cost: ${usage_summary['total_stats']['total_cost']:.4f}")
        print(f"Average Cost per Entry: ${usage_summary['average_per_entry']['total_cost']:.4f}")
    except Exception as e:
        print(f"Error saving results: {e}")


if __name__ == "__main__":
    main()


