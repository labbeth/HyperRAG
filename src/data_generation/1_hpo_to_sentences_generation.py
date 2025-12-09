'''This script generates sentences from HPO terms'''
from hyperrag.config import *
import os
import json
import openai

key = os.environ.get("LLM_API_KEY")
base = os.environ.get("BASE_API_URL")

client = openai.OpenAI(api_key=key, base_url=base)

model = llm_model_hpo_sentences

# set the number of sentences per phenotype
sentence_number = "10"
# set the number of sentences with measurements per label
measures_number = "2"
# set the number of implicit sentences
implicit_count = "2"


# Load HPO data from JSON file
with open(data_path / 'hpo/hp_2025-01-16_format.json') as json_file:
    data = json.load(json_file)

results = []

# Define the absolute range of entries to consider (from index 0 to 19483) in several passes if needed to prevent API overload
start_index = 0
end_index = 19483

# Initialize total token counters
total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens = 0

# Prepare the input for the API call
for i in range(start_index, end_index + 1):  # Include end_index by using end_index + 1
    if i >= len(data["nodes"]):  # Ensure we don't go out of bounds
        break

    entry = data["nodes"][i]
    hpo_label = entry["lbl"]
    hpo_id = entry["id"]
    definition = ""
    if "meta" in entry and isinstance(entry["meta"], dict):
        definition = entry["meta"].get("definition", {}).get("val", "")

    # Extract synonyms safely
    synonyms = []
    if "meta" in entry and isinstance(entry["meta"], dict):
        synonyms = [
            synonym["val"] for synonym in entry["meta"].get("synonyms", [])
            if isinstance(synonym, dict) and synonym.get("pred") == "hasExactSynonym"
        ]

    # Join synonyms into a single string
    synonyms_str = ", ".join(synonyms)



    prompt = f'''For each HPO label, produce {sentence_number} purely observational sentences, referencing the phenotype explicitly or implicitly.

  Requirements:
    - Use both the main HPO label and all provided synonyms for explicit references. 
    - At least {implicit_count} sentences must be implicit references (avoid using the label or synonyms). 
    - Ensure diversity in perspective or detail across sentences (as if from different medical domains and contexts).
    - Ensure diversity in sentence openings.
    - Use first-person (“I”) or neutral (“we”) style; do not include any titles (e.g., “Dr. Smith”).
    - At least two sentences must use passive voice.
    - Vary sentence structure, wording and length (some short, some medium, some extended).
    - Avoid overuse of “the patient” (≤2 uses). Use pronouns (“he,” “she,” “they”) or a fictitious name (first name or “Mr./Mrs.” + last name) for diversity.
      * At least 1 sentence must use a fictitious name instead of “the patient” or a pronoun.
    - Occasionally include measurements/tests (e.g., mg/dL, <1st percentile), up to {measures_number} total.
    - No interpretive language (“suggesting,” “indicative of”); only factual observations.
    - No professorial explanations; just present observations.
    - Return a bulleted list (“- ”). Clinical shorthand is fine.

  Context:
    HPO label: {hpo_label}
    Definition: {definition}
    Synonyms: {synonyms_str}'''

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.4,
            top_p=0.9,
            frequency_penalty=1.2,
            # presence_penalty=1.2,
            messages=[{"role": "user", "content": prompt}],
        )

        generated_content = response.choices[0].message.content
        generated_sentences = [line.strip() for line in generated_content.split('\n') if line.strip().startswith('-')]
        generated_sentences = [sentence[1:].strip() for sentence in generated_sentences]  # Remove the hyphen character

        result_entry = {
            "hpo_label": hpo_label,
            "hpo_id": hpo_id,
            "sentences": generated_sentences,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        results.append(result_entry)

        # Accumulate total token counts
        total_prompt_tokens += response.usage.prompt_tokens
        total_completion_tokens += response.usage.completion_tokens
        total_tokens += response.usage.total_tokens

        # Print confirmation for the processed entry
        print(f"Processed entry {i}: HPO ID: {hpo_id}, HPO Label: {hpo_label}")

    except Exception as e:
        print(f"Error during API call for {hpo_label}: {e}")

# Define the output file name based on the range
output_file_name = data_path / f'hpo/hpo-sentences_{start_index}_to_{end_index}.json'

# Save results to a JSON file
with open(output_file_name, 'w', encoding="utf-8") as output_file:
    json.dump(results, output_file, indent=4)

# Print the total token counts after processing all entries
print(f"Total Prompt Tokens: {total_prompt_tokens}")
print(f"Total Completion Tokens: {total_completion_tokens}")
print(f"Total Tokens: {total_tokens}")

print("Generated sentences saved")
