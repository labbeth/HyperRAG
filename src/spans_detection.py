"""
Script for extracting span annotations from sentences using LLM.

This script:
- Loads a dataset of cleaned sentences.
- Uses a specified language model to detect spans in each sentence via API calls.
- Tracks token usage and associated costs for each API call.
- Saves the extracted spans alongside the original sentences to a CSV file.

Requirements:
- Environment variables LLM_API_KEY and BASE_API_URL must be set for API authentication.
- Configuration and prompt templates are imported from external modules.
"""


from hyperrag.config import *
import openai
import os
import json
import pandas as pd
from span_prompt_template import pipeline_prompt
from cost_tracker import TokenCostTracker
from datetime import datetime

key = os.environ.get("LLM_API_KEY")
base = os.environ.get("BASE_API_URL")

client = openai.OpenAI(api_key=key, base_url=base)

model = llm_span_detection
tracker = TokenCostTracker(model, log_file=f"{target_dataset}_spans_extraction_costs.json")

# open the sentences file
dataset_sentences = pd.read_csv(data_path / f'{target_dataset}/sentences_{target_dataset}_cleaned.csv')

prompt = pipeline_prompt


def openai_api(input_text):
    # Define the prompt
    input_prompt = pipeline_prompt.format(
        person="clinician",
        text="a sentence",
        input=input_text
    )
    print(input_prompt)

    response = client.chat.completions.create(
                model=model,
                temperature=0,
                # top_p=0.9,
                # frequency_penalty=1.2,
                # presence_penalty=1.2,
                messages=[{"role": "user", "content": input_prompt}],
            )
    print(response)

    output_text = response.choices[0].message.content.strip()

    # Track token usage
    tracker.add_usage(prompt, output_text)

    # Display current cost after each API call
    usage_stats = tracker.get_current_usage()
    print(f"Current total cost: ${usage_stats['total_stats']['total_cost']:.6f}")

    return response.choices[0].message.content


outputs = []
for s in dataset_sentences["Sentence_en"]:
    span = openai_api(s)
    outputs.append(span)

print(outputs)

print(f'Final token usage statistics:')
final_stats = tracker.get_current_usage()
print(f"Total tokens: {final_stats['total_stats']['total_tokens']}")
print(f"Total cost: ${final_stats['total_stats']['total_cost']:.6f}")

dataset_sentences['span'] = outputs

pd.DataFrame.to_csv(dataset_sentences, data_path / f'{target_dataset}/sentences_{target_dataset}_spans.csv', index=False)

