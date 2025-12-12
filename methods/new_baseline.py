from datasets import load_dataset
import json
from typing import Dict, List
import random
import os
import argparse
from together import Together
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Model configurations
MODELS = {
    'llama-70b': {
        'provider': 'together',
        'model_id': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
        'description': 'Llama 3.3 70B (Together AI)'
    },
    'gpt-4o-mini': {
        'provider': 'openai',
        'model_id': 'gpt-4o-mini-2024-07-18',
        'description': 'GPT-4o-mini base (OpenAI)'
    },
    'gpt-4o': {
        'provider': 'openai',
        'model_id': 'gpt-4o',
        'description': 'GPT-4o (OpenAI)'
    }
}


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


def run_baseline_evaluation(baseline_data, client, model_id, provider, sample_size=None, seed=42):

    random.seed(seed)
    
    if sample_size:
        baseline_data = random.sample(baseline_data, sample_size)

    correct = 0
    total = 0
    results = []

    print(f"\nEvaluating {model_id}...")
    print(f"Examples: {len(baseline_data)}")
    print("-" * 50)

    for i, example in enumerate(baseline_data):
        if (i + 1) % 20 == 0:
            acc_str = f" (acc: {correct/total:.3f})" if total > 0 else ""
            print(f"  Processing {i+1}/{len(baseline_data)}...{acc_str}")
        
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "user", "content": example['prompt']}
                ],
                max_tokens=10,
                temperature=0
            )

            model_answer = response.choices[0].message.content.strip().upper()

            # Parse response
            if model_answer == 'A' or (model_answer.startswith('A') and 'B' not in model_answer):
                model_prediction = 1
            elif model_answer == 'B' or (model_answer.startswith('B') and 'A' not in model_answer):
                model_prediction = 0
            elif 'A' in model_answer and 'B' not in model_answer:
                model_prediction = 1
            elif 'B' in model_answer and 'A' not in model_answer:
                model_prediction = 0
            else:
                print(f"  Could not parse: {model_answer}")
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
            print(f"  Error: {e}")
            continue

    accuracy = correct / total if total > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"RESULTS: {model_id}")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"{'='*50}")

    return {
        'model_id': model_id,
        'provider': provider,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results
    }


def get_client(provider: str):
    if provider == 'together':
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set")
        return Together(api_key=api_key)
    elif provider == 'openai':
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAI(api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline evaluation on SHP dataset with different models"
    )
    parser.add_argument(
        "--model", type=str, default="llama-70b",
        choices=list(MODELS.keys()) + ['all', 'custom'],
        help="Model to evaluate (default: llama-70b)"
    )
    parser.add_argument(
        "--custom-model", type=str, default=None,
        help="Custom OpenAI model ID (e.g., ft:gpt-4o-mini-2024-07-18:...)"
    )
    parser.add_argument(
        "--sample-size", type=int, default=100,
        help="Number of examples per run (default: 100)"
    )
    parser.add_argument(
        "--num-runs", type=int, default=1,
        help="Number of evaluation runs to average (default: 1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "test"],
        help="Dataset split to evaluate on (default: test)"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print("Loading SHP dataset...")
    ds = load_dataset("stanfordnlp/SHP")
    print(f"Dataset size - Train: {len(ds['train'])}, Test: {len(ds['test'])}")
    
    # Prepare data
    baseline_data = prepare_baseline_data(ds, args.split)
    print(f"Prepared {len(baseline_data)} examples from {args.split} split")
    
    # Determine which models to run
    if args.model == 'all':
        models_to_run = list(MODELS.keys())
    elif args.model == 'custom':
        if not args.custom_model:
            raise ValueError("--custom-model required when using --model custom")
        models_to_run = ['custom']
    else:
        models_to_run = [args.model]
    
    # Run evaluations
    all_results = {}
    
    for model_key in models_to_run:
        if model_key == 'custom':
            model_id = args.custom_model
            provider = 'openai'
            description = f"Custom model: {model_id}"
        else:
            model_config = MODELS[model_key]
            model_id = model_config['model_id']
            provider = model_config['provider']
            description = model_config['description']
        
        print(f"\n{'='*60}")
        print(f"EVALUATING: {description}")
        print(f"{'='*60}")
        
        # Get client
        client = get_client(provider)
        
        # Run multiple times if requested
        aggregate_accuracy = 0.0
        
        for run in range(args.num_runs):
            if args.num_runs > 1:
                print(f"\n--- Run {run+1}/{args.num_runs} ---")
            
            # Use different seed for each run
            run_seed = args.seed + run
            
            output = run_baseline_evaluation(
                baseline_data,
                client,
                model_id=model_id,
                provider=provider,
                sample_size=args.sample_size,
                seed=run_seed
            )
            aggregate_accuracy += output['accuracy']
        
        avg_accuracy = aggregate_accuracy / args.num_runs
        all_results[model_key if model_key != 'custom' else model_id] = {
            'description': description,
            'avg_accuracy': avg_accuracy,
            'num_runs': args.num_runs
        }
        
        if args.num_runs > 1:
            print(f"\nAverage accuracy over {args.num_runs} runs: {avg_accuracy:.4f}")
    
    # Print comparison if multiple models
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['avg_accuracy'], reverse=True)
        
        for i, (model_key, result) in enumerate(sorted_results):
            rank = i + 1
            print(f"  {rank}. {result['description']}: {result['avg_accuracy']:.4f}")
        
        print(f"{'='*60}")
    
    # Save results
    results_file = f"baseline_results_{args.model}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
