import json
import random
import argparse
import os
from pathlib import Path

from together import Together
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_OUTPUT_DIR = "data"
REASONING_OUTPUT_FILE = "reasoning_traces.json"
SFT_OUTPUT_FILE = "reddit_preference_sft.json"

SYSTEM_MESSAGE = """You are an expert at analyzing Reddit responses. Compare two responses objectively and determine which one is better and why. Consider factors like helpfulness, accuracy, relevance, clarity, and overall usefulness to the person asking the question."""


# =============================================================================
# DATA LOADING
# =============================================================================

def load_shp_dataset(sample_size=None, split="train"):
    print(f"Loading SHP dataset (split: {split})...")
    dataset = load_dataset("stanfordnlp/SHP", split=split)
    
    baseline_data = []
    for i, example in enumerate(dataset):
        if sample_size and i >= sample_size:
            break
            
        baseline_data.append({
            'post_id': example['post_id'],
            'post': example['history'],  # The original post/question
            'response_A': example['human_ref_A'],
            'response_B': example['human_ref_B'],
            'true_label': example['labels'],  # 1 = A preferred, 0 = B preferred
        })
    
    print(f"Loaded {len(baseline_data)} examples")
    return baseline_data


def load_baseline_from_file(filepath):
    """Load baseline data from a JSON file if already processed."""
    print(f"Loading baseline data from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")
    return data


# =============================================================================
# REASONING TRACE GENERATION
# =============================================================================

def format_reasoning_prompt(post, response_a, response_b):
    """Format the prompt for reasoning trace generation."""
    prompt = f"""A Reddit user posted a question and received two different responses.
Analyze which response is better and explain why.

Original Post:
{post}

Response A:
{response_a}

Response B:
{response_b}

Think through this step-by-step:
1. First, I'll analyze the helpfulness and relevance...
2. Next, I'll examine the clarity and completeness...
3. Then, I'll evaluate the accuracy and usefulness...
4. Finally, I'll assess the overall quality...

**ANSWER IN THE FOLLOWING FORMAT:**
ANSWER: [A/B]
REASONING: Response [A/B] is better because..."""

    return prompt


def generate_reasoning_traces(baseline_data, client, model=DEFAULT_MODEL, sample_size=None):

    if sample_size and sample_size < len(baseline_data):
        baseline_data = random.sample(baseline_data, sample_size)
    
    reasoning_data = []
    agreements = 0
    total_processed = 0

    print(f"\nGenerating reasoning traces for {len(baseline_data)} examples...")
    print(f"Using model: {model}")
    print("-" * 60)

    for i, example in enumerate(baseline_data):
        print(f"Processing example {i+1}/{len(baseline_data)} (ID: {example['post_id']})")
        
        try:
            reasoning_prompt = format_reasoning_prompt(
                example['post'],
                example['response_A'],
                example['response_B']
            )
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": reasoning_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            reasoning_response = response.choices[0].message.content.strip()

            # Parse the model's answer
            model_choice = None
            model_prediction = None
            
            if "ANSWER: A" in reasoning_response:
                model_choice = 'A'
                model_prediction = 1
            elif "ANSWER: B" in reasoning_response:
                model_choice = 'B'
                model_prediction = 0
            else:
                print(f"  Could not parse answer, skipping...")
                continue

            true_label = example['true_label']
            agrees_with_truth = (model_prediction == true_label)
            agreements += int(agrees_with_truth)
            total_processed += 1

            status = "✓ AGREES" if agrees_with_truth else "✗ DISAGREES"
            true_choice = 'A' if true_label == 1 else 'B'
            print(f"  Model: {model_choice} | Truth: {true_choice} | {status}")

            # Only keep agreements (avoid sycophancy)
            if agrees_with_truth:
                reasoning_data.append({
                    'post_id': example['post_id'],
                    'post': example['post'],
                    'response_A': example['response_A'],
                    'response_B': example['response_B'],
                    'true_label': true_label,
                    'model_choice': model_choice,
                    'full_reasoning_response': reasoning_response,
                })
                print(f"  → Added to training data")
            else:
                print(f"  → Discarded (disagreement)")

            if total_processed > 0:
                print(f"  Agreement rate: {agreements/total_processed:.1%} ({agreements}/{total_processed})")
            print()

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print("=" * 60)
    print("REASONING GENERATION COMPLETE")
    print(f"  Total Processed: {total_processed}")
    print(f"  Agreements: {agreements} ({agreements/total_processed:.1%})" if total_processed > 0 else "  No examples processed")
    print(f"  Training Examples Created: {len(reasoning_data)}")
    print("=" * 60)

    return reasoning_data


# =============================================================================
# SFT DATASET CREATION
# =============================================================================

def extract_reasoning_from_response(reasoning_response, correct_choice):
    try:
        if "REASONING:" in reasoning_response:
            reasoning_part = reasoning_response.split("REASONING:", 1)[1].strip()
            
            # Clean up the reasoning text
            reasoning_part = reasoning_part.replace(
                f"Response {correct_choice} is better because", ""
            ).strip()
            reasoning_part = reasoning_part.replace(
                f"Response {correct_choice} is better", ""
            ).strip()
            
            # Remove any leading punctuation
            reasoning_part = reasoning_part.lstrip(":.,- ")
            
            return reasoning_part
        else:
            return "it provides a more helpful, accurate, and relevant response to the question asked."
    except Exception:
        return "it better addresses the user's needs and provides more valuable information."


def generate_sft_dataset(reasoning_data, output_file):
    sft_dataset = []

    print(f"\nGenerating SFT dataset from {len(reasoning_data)} reasoning traces...")

    for i, example in enumerate(reasoning_data):
        if (i + 1) % 100 == 0:
            print(f"  Processing example {i+1}/{len(reasoning_data)}")

        # Extract data
        post = example['post']
        response_A = example['response_A']
        response_B = example['response_B']
        true_label = example['true_label']  # 1 = A preferred, 0 = B preferred
        reasoning_response = example['full_reasoning_response']

        # Determine the correct answer based on true human preference
        correct_choice = 'A' if true_label == 1 else 'B'

        # Extract reasoning from the model's response
        reasoning_text = extract_reasoning_from_response(reasoning_response, correct_choice)

        # Randomly flip A/B positions to avoid positional bias
        flip_positions = random.choice([True, False])

        if flip_positions:
            # Swap A and B
            display_response_A = response_B
            display_response_B = response_A
            display_correct_choice = 'B' if correct_choice == 'A' else 'A'
        else:
            # Keep original order
            display_response_A = response_A
            display_response_B = response_B
            display_correct_choice = correct_choice

        # Create the user prompt
        user_prompt = f"""Which response is better? Analyze the differences between these two responses.

Original Post:
{post}

Response A:
{display_response_A}

Response B:
{display_response_B}

Think through this step-by-step:
1. First, I'll analyze the helpfulness and relevance...
2. Next, I'll examine the clarity and completeness...
3. Then, I'll evaluate the accuracy and usefulness...
4. Finally, I'll assess the overall quality...

**ANSWER IN THE FOLLOWING FORMAT:**
Response [A/B] is better because..."""

        # Create the assistant response
        assistant_response = f"Response {display_correct_choice} is better because {reasoning_text}"

        # Create SFT training example
        sft_example = {
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
                {
                    "role": "assistant",
                    "content": assistant_response
                }
            ],
            "metadata": {
                "post_id": example['post_id'],
                "true_label": true_label,
                "flipped": flip_positions,
                "original_model_choice": example['model_choice']
            }
        }

        sft_dataset.append(sft_example)

    # Save the dataset
    with open(output_file, 'w') as f:
        json.dump(sft_dataset, f, indent=2)

    # Print statistics
    flipped_count = sum(1 for ex in sft_dataset if ex['metadata']['flipped'])
    
    print(f"\n{'=' * 60}")
    print("SFT DATASET GENERATED")
    print(f"  Total examples: {len(sft_dataset)}")
    print(f"  Position flips: {flipped_count}/{len(sft_dataset)} ({flipped_count/len(sft_dataset)*100:.1f}%)")
    print(f"  Saved to: {output_file}")
    print("=" * 60)

    return sft_dataset


def preview_sft_examples(sft_dataset, num_examples=2):
    print(f"\nPREVIEW OF SFT DATASET (showing {num_examples} examples):")
    print("=" * 80)

    for i in range(min(num_examples, len(sft_dataset))):
        example = sft_dataset[i]
        print(f"\nEXAMPLE {i+1}:")
        print(f"  Post ID: {example['metadata']['post_id']}")
        print(f"  Flipped: {example['metadata']['flipped']}")
        print(f"\n  USER PROMPT (truncated):")
        print(f"  {example['messages'][1]['content'][:300]}...")
        print(f"\n  ASSISTANT RESPONSE:")
        print(f"  {example['messages'][2]['content'][:500]}")
        print("-" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate reasoning traces and SFT dataset for preference learning"
    )
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of examples to process (default: {DEFAULT_SAMPLE_SIZE})"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Together AI model to use for reasoning (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save outputs (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--baseline-file", type=str, default=None,
        help="Load baseline data from file instead of HuggingFace"
    )
    parser.add_argument(
        "--reasoning-file", type=str, default=None,
        help="Load existing reasoning traces instead of generating new ones"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Preview SFT examples after generation"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Initialize Together AI client
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    reasoning_output = output_dir / REASONING_OUTPUT_FILE
    sft_output = output_dir / SFT_OUTPUT_FILE
    
    # Step 1: Load or generate reasoning traces
    if args.reasoning_file:
        # Load existing reasoning traces
        print(f"Loading existing reasoning traces from {args.reasoning_file}...")
        with open(args.reasoning_file, 'r') as f:
            reasoning_data = json.load(f)
        print(f"Loaded {len(reasoning_data)} reasoning traces")
    else:
        # Load baseline data
        if args.baseline_file:
            baseline_data = load_baseline_from_file(args.baseline_file)
        else:
            baseline_data = load_shp_dataset(sample_size=args.sample_size)
        
        # Generate reasoning traces
        reasoning_data = generate_reasoning_traces(
            baseline_data,
            client=client,
            model=args.model,
            sample_size=args.sample_size
        )
        
        # Save reasoning traces
        with open(reasoning_output, 'w') as f:
            json.dump(reasoning_data, f, indent=2)
        print(f"\nReasoning traces saved to: {reasoning_output}")
    
    # Step 2: Generate SFT dataset
    sft_dataset = generate_sft_dataset(reasoning_data, sft_output)
    
    # Optional: Preview examples
    if args.preview:
        preview_sft_examples(sft_dataset)
    
    print(f"\n Done! SFT dataset ready at: {sft_output}")
    print(f"  Next step: Run finetune.py to train the expert model")


if __name__ == "__main__":
    main()
