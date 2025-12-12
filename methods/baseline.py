from datasets import load_dataset
import json
from typing import Dict, List
import random
import os
from together import Together
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def format_baseline_prompt(post, response_a, response_b):
    prompt = f"""Given the following post and two responses, determine which response is better.

POST: {post}

RESPONSE A: {response_a}

RESPONSE B: {response_b}

Which response is better? Respond with only "A" or "B"."""

    return prompt


def prepare_baseline_data(dataset, split='train'):
    baseline_data = []
    for example in dataset[split]:
        formatted_example = {
            'post_id': example['post_id'],
            'post': example['history'],
            'response_A': example['human_ref_A'],
            'response_B': example['human_ref_B'],
            'true_label': example['labels'],
            'prompt': format_baseline_prompt(
                example['history'],
                example['human_ref_A'],
                example['human_ref_B']
            )
        }
        baseline_data.append(formatted_example)
    return baseline_data


def run_baseline_evaluation(baseline_data, client, sample_size=None):
    if sample_size:
        baseline_data = random.sample(baseline_data, sample_size)

    correct = 0
    total = 0
    results = []

    for example in baseline_data:
        try:
            response = client.chat.completions.create(
                model='meta-llama/Llama-3.3-70B-Instruct-Turbo',
                messages=[
                    {"role": "user", "content": example['prompt']}
                ],
                max_tokens=10,
                temperature=0
            )

            model_answer = response.choices[0].message.content.strip().upper()

            if model_answer == 'A':
                model_prediction = 1
            elif model_answer == 'B':
                model_prediction = 0
            else:
                print(f"Unexpected model response: {model_answer}")
                continue

            # Check if correct
            is_correct = (model_prediction == example['true_label'])
            correct += is_correct
            total += 1

            results.append({
                'post_id': example['post_id'],
                'model_prediction': model_prediction,
                'true_label': example['true_label'],
                'correct': is_correct,
                'model_response': model_answer
            })

        except Exception as e:
            print(f"Error processing example {example['post_id']}: {e}")
            continue

    accuracy = correct / total if total > 0 else 0
    print(f"Baseline Accuracy: {accuracy:.3f} ({correct}/{total})")

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results
    }


def main():
    # Initialize Together AI client
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    
    # Load dataset
    ds = load_dataset("stanfordnlp/SHP")
    print(f"Dataset size - Train: {len(ds['train'])}, Test: {len(ds['test'])}")

    # Prepare data
    train_baseline = prepare_baseline_data(ds, 'train')
    test_baseline = prepare_baseline_data(ds, 'test')

    # Run evaluation
    aggregate_accuracy = 0.0
    num_runs = 10
    
    for i in range(num_runs):
        print(f"\nRun {i+1}/{num_runs}")
        output = run_baseline_evaluation(test_baseline, client, sample_size=100)
        aggregate_accuracy += output['accuracy']
    
    print(f"\nAverage accuracy is {aggregate_accuracy/num_runs:.3f}")


if __name__ == "__main__":
    main()
