import json
import argparse
import time
import random
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from openai import OpenAI
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# === CONFIGURATION ===
DEFAULT_BASE_MODEL = "gpt-4o-mini-2024-07-18"
DEFAULT_EPOCHS = 2
DEFAULT_SFT_FILE = "data/reddit_preference_sft.json"
DEFAULT_JSONL_FILE = "data/shp_openai_format.jsonl"

SYSTEM_MESSAGE = """You are an expert at analyzing Reddit responses. Compare two responses objectively and determine which one is better and why. Consider factors like helpfulness, accuracy, relevance, clarity, and overall usefulness to the person asking the question."""


def load_sft_data(filepath: str) -> List[Dict]:
    print(f"\nLoading SFT data from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} examples")
    return data


def convert_to_openai_jsonl(sft_data: List[Dict], output_file: str) -> int:    
    print(f"\nConverting to OpenAI JSONL format...")
    
    with open(output_file, 'w') as f:
        for example in sft_data:
            # SFT data already has the correct messages format
            openai_example = {
                "messages": example["messages"]
            }
            json.dump(openai_example, f)
            f.write('\n')
    
    print(f"  Converted {len(sft_data)} examples")
    print(f"  Saved to: {output_file}")
    
    return len(sft_data)


def upload_training_file(client: OpenAI, jsonl_file: str) -> str:
    print(f"\nUploading {jsonl_file} to OpenAI...")
    
    with open(jsonl_file, 'rb') as f:
        response = client.files.create(
            file=f,
            purpose='fine-tune'
        )
    
    file_id = response.id
    print(f"  File ID: {file_id}")
    
    # Wait for file to be processed
    print("  Waiting for file processing...")
    while True:
        file_status = client.files.retrieve(file_id)
        if file_status.status == 'processed':
            print("  File processed successfully")
            break
        elif file_status.status == 'error':
            raise Exception(f"File processing failed: {file_status.status_details}")
        time.sleep(2)
    
    return file_id


def launch_finetuning(
    client: OpenAI,
    file_id: str,
    base_model: str = DEFAULT_BASE_MODEL,
    n_epochs: int = DEFAULT_EPOCHS,
    suffix: Optional[str] = None
) -> str:

    print(f"\n{'='*50}")
    print("LAUNCHING FINE-TUNING JOB")
    print(f"{'='*50}")
    print(f"Base model: {base_model}")
    print(f"Epochs: {n_epochs}")
    print(f"Suffix: {suffix or 'shp-preference-expert'}")
    
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=base_model,
        hyperparameters={
            "n_epochs": n_epochs
        },
        suffix=suffix or "shp-preference-expert"
    )
    
    job_id = job.id
    print(f"\nJob ID: {job_id}")
    print(f"Status: {job.status}")
    
    return job_id


def check_status(client: OpenAI, job_id: str) -> Tuple[Optional[str], str]:
    job = client.fine_tuning.jobs.retrieve(job_id)
    
    print(f"Status: {job.status}")
    
    if job.status == "succeeded":
        model_name = job.fine_tuned_model
        print(f"✓ Model ready: {model_name}")
        return model_name, "succeeded"
        
    elif job.status == "failed":
        error_msg = job.error.message if job.error else "Unknown error"
        print(f"✗ Failed: {error_msg}")
        return None, "failed"
        
    elif job.status == "cancelled":
        print(f"✗ Cancelled")
        return None, "cancelled"
    
    if job.trained_tokens:
        print(f"  Trained tokens: {job.trained_tokens}")
    
    return None, job.status


def wait_for_completion(client: OpenAI, job_id: str, check_interval: int = 30) -> Optional[str]:

    print(f"\nWaiting for fine-tuning to complete...")
    print(f"Checking every {check_interval} seconds")
    print("-" * 40)
    
    iteration = 0
    while True:
        iteration += 1
        print(f"\n[Check #{iteration}]")
        
        model_id, status = check_status(client, job_id)
        
        if status == "succeeded":
            return model_id
        elif status in ["failed", "cancelled"]:
            return None
        
        print(f"Waiting {check_interval}s...")
        time.sleep(check_interval)


def list_jobs(client: OpenAI, limit: int = 10) -> List:

    print(f"\nRecent fine-tuning jobs (limit {limit}):")
    print("-" * 60)
    
    jobs = client.fine_tuning.jobs.list(limit=limit)
    
    for job in jobs.data:
        model_str = f" → {job.fine_tuned_model}" if job.fine_tuned_model else ""
        print(f"  {job.id}: {job.status} ({job.model}){model_str}")
    
    return jobs.data


def cancel_job(client: OpenAI, job_id: str):
    print(f"Cancelling job {job_id}...")
    client.fine_tuning.jobs.cancel(job_id)
    print("  Job cancelled")


def list_events(client: OpenAI, job_id: str, limit: int = 20):
    print(f"\nRecent events for job {job_id}:")
    print("-" * 60)
    
    events = client.fine_tuning.jobs.list_events(job_id, limit=limit)
    
    for event in reversed(events.data):
        print(f"  [{event.created_at}] {event.message}")


def load_test_data(sample_size: Optional[int] = None, seed: int = 42) -> List[Dict]:
    print(f"\nLoading test data from HuggingFace SHP...")
    
    ds = load_dataset("stanfordnlp/SHP", split='test')
    
    test_data = []
    for example in ds:
        test_data.append({
            'post_id': example['post_id'],
            'post': example['history'],
            'response_A': example['human_ref_A'],
            'response_B': example['human_ref_B'],
            'true_label': example['labels'],  # 1 = A better, 0 = B better
        })
    
    if sample_size and sample_size < len(test_data):
        random.seed(seed)
        test_data = random.sample(test_data, sample_size)
    
    print(f"  Loaded {len(test_data)} test examples")
    return test_data


def format_evaluation_prompt(post: str, response_a: str, response_b: str) -> str:

    return f"""Which response is better? Analyze the differences between these two responses.

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
Response [A/B] is better because..."""


def evaluate_finetuned_model(
    client: OpenAI,
    model_id: str,
    test_data: List[Dict],
    sample_size: int = 100,
    seed: int = 42,
    verbose: bool = True
) -> Dict:

    random.seed(seed)
    
    print(f"\n{'='*50}")
    print("FINE-TUNED MODEL EVALUATION")
    print(f"{'='*50}")
    print(f"Model: {model_id}")
    print(f"Examples: {min(sample_size, len(test_data))}")
    print("-" * 50)
    
    if sample_size < len(test_data):
        test_data = random.sample(test_data, sample_size)
    
    correct = 0
    total = 0
    results = []
    
    for i, example in enumerate(test_data):
        if verbose and (i + 1) % 10 == 0:
            acc_str = f" (acc: {correct/total:.3f})" if total > 0 else ""
            print(f"  Processing {i+1}/{len(test_data)}...{acc_str}")
        
        try:
            prompt = format_evaluation_prompt(
                example['post'],
                example['response_A'],
                example['response_B']
            )
            
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0
            )
            
            full_response = response.choices[0].message.content.strip()
            
            # Parse the answer - matches training format "Response [A/B] is better"
            model_choice = None
            if full_response.startswith("Response A is better") or "Response A is better" in full_response[:50]:
                model_choice = "A"
            elif full_response.startswith("Response B is better") or "Response B is better" in full_response[:50]:
                model_choice = "B"
            else:
                # Fallback parsing
                lower_response = full_response.lower()[:100]
                if "response a" in lower_response and "better" in lower_response:
                    model_choice = "A"
                elif "response b" in lower_response and "better" in lower_response:
                    model_choice = "B"
                else:
                    if verbose:
                        print(f"    Could not parse: {full_response[:80]}...")
                    continue
            
            # Convert to numeric for comparison (1 = A better, 0 = B better)
            model_prediction = 1 if model_choice == 'A' else 0
            is_correct = (model_prediction == example['true_label'])
            correct += int(is_correct)
            total += 1
            
            results.append({
                'post_id': example['post_id'],
                'model_choice': model_choice,
                'true_label': example['true_label'],
                'correct': is_correct,
                'reasoning': full_response[:300]
            })
            
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n{'='*50}")
    print("FINE-TUNED MODEL RESULTS")
    print(f"{'='*50}")
    print(f"Model: {model_id}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"{'='*50}")
    
    return {
        'model_id': model_id,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results
    }


def model_id_to_filename(model_id: str) -> str:
    return model_id.replace(":", "_").replace("/", "_")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune and evaluate SHP preference expert model on OpenAI"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Launch fine-tuning")
    train_parser.add_argument(
        "--input", type=str, default=DEFAULT_SFT_FILE,
        help=f"Input SFT JSON file (default: {DEFAULT_SFT_FILE})"
    )
    train_parser.add_argument(
        "--jsonl", type=str, default=DEFAULT_JSONL_FILE,
        help=f"Output JSONL file path (default: {DEFAULT_JSONL_FILE})"
    )
    train_parser.add_argument(
        "--model", type=str, default=DEFAULT_BASE_MODEL,
        help=f"Base model to fine-tune (default: {DEFAULT_BASE_MODEL})"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Number of epochs (default: {DEFAULT_EPOCHS})"
    )
    train_parser.add_argument(
        "--suffix", type=str, default=None,
        help="Model name suffix"
    )
    train_parser.add_argument(
        "--wait", action="store_true",
        help="Wait for fine-tuning to complete"
    )
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check job status")
    status_parser.add_argument("job_id", type=str, help="Fine-tuning job ID")
    status_parser.add_argument("--wait", action="store_true", help="Wait for completion")
    status_parser.add_argument("--events", action="store_true", help="Show job events")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List fine-tuning jobs")
    list_parser.add_argument(
        "--limit", type=int, default=10,
        help="Number of jobs to list (default: 10)"
    )
    
    # Cancel command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a job")
    cancel_parser.add_argument("job_id", type=str, help="Job ID to cancel")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate fine-tuned model")
    eval_parser.add_argument("model_id", type=str, help="Fine-tuned model ID")
    eval_parser.add_argument(
        "--num-examples", type=int, default=100,
        help="Number of test examples (default: 100)"
    )
    eval_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key)
    
    if args.command == "train":
        # Step 1: Load SFT data
        sft_data = load_sft_data(args.input)
        
        # Step 2: Convert to JSONL
        convert_to_openai_jsonl(sft_data, args.jsonl)
        
        # Step 3: Upload file
        file_id = upload_training_file(client, args.jsonl)
        
        # Step 4: Launch fine-tuning
        job_id = launch_finetuning(
            client=client,
            file_id=file_id,
            base_model=args.model,
            n_epochs=args.epochs,
            suffix=args.suffix
        )
        
        # Step 5: Wait if requested
        if args.wait:
            model_id = wait_for_completion(client, job_id)
            if model_id:
                print(f"\n Fine-tuning complete!")
                print(f"  Model ID: {model_id}")
                print(f"\n  To evaluate:")
                print(f"    python finetune_shp.py evaluate {model_id}")
            else:
                print(f"\n✗ Fine-tuning failed or was cancelled")
        else:
            print(f"\n Fine-tuning job launched!")
            print(f"  Job ID: {job_id}")
            print(f"\n  To check status:")
            print(f"    python finetune_shp.py status {job_id}")
            print(f"\n  To wait for completion:")
            print(f"    python finetune_shp.py status {job_id} --wait")
    
    elif args.command == "status":
        if args.events:
            list_events(client, args.job_id)
        
        if args.wait:
            model_id = wait_for_completion(client, args.job_id)
            if model_id:
                print(f"\n Model ready: {model_id}")
        else:
            check_status(client, args.job_id)
    
    elif args.command == "list":
        list_jobs(client, args.limit)
    
    elif args.command == "cancel":
        cancel_job(client, args.job_id)
    
    elif args.command == "evaluate":
        # Load test data from HuggingFace
        test_data = load_test_data(sample_size=args.num_examples, seed=args.seed)
        
        # Evaluate fine-tuned model
        results = evaluate_finetuned_model(
            client=client,
            model_id=args.model_id,
            test_data=test_data,
            sample_size=args.num_examples,
            seed=args.seed
        )
        
        # Save results
        results_file = f"eval_results_{model_id_to_filename(args.model_id)}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
