import json
import random
import math
import argparse
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

from openai import OpenAI
from together import Together
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Generator models - diverse set for rollout generation
GENERATOR_MODELS = {
    'llama-3.3-70b': {
        'provider': 'together',
        'model_id': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
    },
    'llama-3.1-70b': {
        'provider': 'together',
        'model_id': 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
    },
    'mistral-7b': {
        'provider': 'together',
        'model_id': 'mistralai/Mistral-7B-Instruct-v0.3',
    },
    'mixtral-8x7b': {
        'provider': 'together',
        'model_id': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    },
    'qwen-72b': {
        'provider': 'together',
        'model_id': 'Qwen/Qwen2.5-72B-Instruct-Turbo',
    },
    'qwen-7b': {
        'provider': 'together',
        'model_id': 'Qwen/Qwen2.5-7B-Instruct-Turbo',
    },
    'gpt-4o-mini': {
        'provider': 'openai',
        'model_id': 'gpt-4o-mini-2024-07-18',
    },
}

# Judge models for multi-model position-debiased ensemble
JUDGE_MODELS = {
    'gpt-4o-mini': {
        'provider': 'openai',
        'model_id': 'gpt-4o-mini-2024-07-18',
    },
    'gpt-4o': {
        'provider': 'openai',
        'model_id': 'gpt-4o',
    },
    'llama-70b': {
        'provider': 'together',
        'model_id': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
    },
}

DEFAULT_OUTPUT_DIR = "data"
REWARDS_OUTPUT_FILE = "shp_rollout_rewards.json"
TRAINING_DATA_FILE = "shp_grpo_training_data.json"
DEFAULT_ELO_K = 32
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RolloutResult:
    prompt_id: str
    prompt: str
    rollouts: List[str]
    rollout_sources: List[str]  # Which model generated each rollout
    elo_ratings: List[float]
    rewards: List[float]
    num_comparisons: int
    

@dataclass  
class TrainingExample:
    prompt_id: str
    prompt: str
    response: str
    source_model: str
    reward: float
    rank: int


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def call_with_retry(func, *args, **kwargs):
    last_exception = None
    
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            if "rate_limit" in error_str or "429" in error_str or "overloaded" in error_str:
                last_exception = e
                delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                print(f"      Retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(delay)
            else:
                raise e
    
    raise last_exception


def get_client(provider: str, clients: Dict):
    """Get the appropriate client for a provider."""
    return clients.get(provider)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_shp_prompts(
    sample_size: Optional[int] = None,
    seed: int = 42,
    split: str = 'test'
) -> List[Dict]:    
    print(f"Loading SHP dataset (split: {split})...")
    ds = load_dataset("stanfordnlp/SHP", split=split)
    
    prompts = []
    for i, example in enumerate(ds):
        prompts.append({
            'prompt_id': example['post_id'],
            'prompt': example['history'],
            'original_response_a': example['human_ref_A'],
            'original_response_b': example['human_ref_B'],
            'original_label': example['labels'],
        })
    
    print(f"  Found {len(prompts)} examples")
    
    if sample_size and sample_size < len(prompts):
        random.seed(seed)
        prompts = random.sample(prompts, sample_size)
        print(f"  Sampled {len(prompts)} prompts")
    
    return prompts


# =============================================================================
# ROLLOUT GENERATION
# =============================================================================

def generate_single_rollout(
    clients: Dict,
    prompt: str,
    model_key: str,
    max_tokens: int = 300
) -> Tuple[Optional[str], str]:

    model_config = GENERATOR_MODELS[model_key]
    provider = model_config['provider']
    model_id = model_config['model_id']
    client = clients[provider]
    
    try:
        def make_request():
            return client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7  # Some temperature for variety
            )
        
        response = call_with_retry(make_request)
        rollout = response.choices[0].message.content.strip()
        return rollout, model_key
        
    except Exception as e:
        print(f"    Error generating from {model_key}: {e}")
        return None, model_key


def generate_diverse_rollouts(
    clients: Dict,
    prompt: str,
    model_keys: List[str],
    max_tokens: int = 300
) -> Tuple[List[str], List[str]]:
    rollouts = []
    sources = []
    
    for model_key in model_keys:
        result, source = generate_single_rollout(
            clients, prompt, model_key, max_tokens
        )
        if result is not None:
            rollouts.append(result)
            sources.append(source)
    
    return rollouts, sources


# =============================================================================
# JUDGING WITH MULTI-MODEL POSITION-DEBIASED ENSEMBLE
# =============================================================================

def format_judge_prompt(post: str, response_a: str, response_b: str) -> str:
    return f"""Given the following post and two responses, determine which response is better.

POST: {post}

RESPONSE A: {response_a}

RESPONSE B: {response_b}

Which response is better? Respond with only "A" or "B"."""


def get_single_judgment(
    client,
    model_id: str,
    prompt: str,
) -> Optional[int]:

    try:
        def make_request():
            return client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )
        
        response = call_with_retry(make_request)
        answer = response.choices[0].message.content.strip().upper()
        
        if 'A' in answer and 'B' not in answer:
            return 1
        elif 'B' in answer and 'A' not in answer:
            return 0
        return None
        
    except Exception as e:
        print(f"      Judge error: {e}")
        return None


def position_debiased_judgment(
    client,
    model_id: str,
    post: str,
    response_a: str,
    response_b: str
) -> Optional[int]:
    
    # Original order
    prompt1 = format_judge_prompt(post, response_a, response_b)
    pred1 = get_single_judgment(client, model_id, prompt1)
    
    # Swapped order
    prompt2 = format_judge_prompt(post, response_b, response_a)
    pred2 = get_single_judgment(client, model_id, prompt2)
    
    if pred1 is None or pred2 is None:
        return None
    
    # Adjust pred2 for swap
    pred2_adjusted = 1 - pred2
    
    # Only return if consistent
    if pred1 == pred2_adjusted:
        return pred1
    return None


def multi_model_position_debiased_judge(
    clients: Dict,
    post: str,
    response_a: str,
    response_b: str,
    judge_models: Dict = JUDGE_MODELS
) -> Optional[str]:
 
    votes = {0: 0, 1: 0}
    
    for model_key, model_config in judge_models.items():
        client = clients[model_config['provider']]
        model_id = model_config['model_id']
        
        pred = position_debiased_judgment(
            client, model_id, post, response_a, response_b
        )
        
        if pred is not None:
            votes[pred] += 1
    
    # Need at least one vote
    if votes[0] == 0 and votes[1] == 0:
        return None
    
    # Majority vote
    if votes[1] > votes[0]:
        return 'A'
    elif votes[0] > votes[1]:
        return 'B'
    else:
        # Tie - could return None or random
        return None


# =============================================================================
# ELO TOURNAMENT
# =============================================================================

def update_elo(
    rating_a: float, 
    rating_b: float, 
    winner: str, 
    k: float = DEFAULT_ELO_K
) -> Tuple[float, float]:

    expected_a = 1 / (1 + 10**((rating_b - rating_a) / 400))
    expected_b = 1 - expected_a
    
    if winner == "A":
        score_a, score_b = 1.0, 0.0
    elif winner == "B":
        score_a, score_b = 0.0, 1.0
    else:
        score_a, score_b = 0.5, 0.5
    
    new_rating_a = rating_a + k * (score_a - expected_a)
    new_rating_b = rating_b + k * (score_b - expected_b)
    
    return new_rating_a, new_rating_b


def elo_to_rewards(elo_ratings: List[float]) -> List[float]:

    if len(elo_ratings) == 0:
        return []
    if len(elo_ratings) == 1:
        return [0.5]
    
    min_r = min(elo_ratings)
    max_r = max(elo_ratings)
    spread = max_r - min_r
    
    if spread == 0:
        return [0.5] * len(elo_ratings)
    
    return [(r - min_r) / spread for r in elo_ratings]


def run_elo_tournament(
    clients: Dict,
    post: str,
    rollouts: List[str],
    verbose: bool = True
) -> Tuple[List[float], List[float], int]:
 
    n = len(rollouts)
    elo_ratings = [1000.0] * n  # Start at 1000
    
    # Generate all pairs
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(pairs)
    
    total_comparisons = len(pairs)
    valid_comparisons = 0
    
    if verbose:
        print(f"    Running {total_comparisons} pairwise comparisons...")
    
    for pair_idx, (i, j) in enumerate(pairs):
        # Random position to avoid bias
        if random.random() < 0.5:
            response_a, response_b = rollouts[i], rollouts[j]
            idx_a, idx_b = i, j
        else:
            response_a, response_b = rollouts[j], rollouts[i]
            idx_a, idx_b = j, i
        
        winner = multi_model_position_debiased_judge(
            clients, post, response_a, response_b
        )
        
        if winner is not None:
            valid_comparisons += 1
            
            if winner == "A":
                winner_idx, loser_idx = idx_a, idx_b
            else:
                winner_idx, loser_idx = idx_b, idx_a
            
            new_winner, new_loser = update_elo(
                elo_ratings[winner_idx],
                elo_ratings[loser_idx],
                "A"
            )
            elo_ratings[winner_idx] = new_winner
            elo_ratings[loser_idx] = new_loser
        
        if verbose and (pair_idx + 1) % 5 == 0:
            print(f"      Completed {pair_idx + 1}/{total_comparisons} comparisons")
    
    rewards = elo_to_rewards(elo_ratings)
    
    return elo_ratings, rewards, valid_comparisons


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_single_prompt(
    clients: Dict,
    prompt_data: Dict,
    generator_models: List[str],
    verbose: bool = True
) -> Optional[RolloutResult]:

    prompt_id = prompt_data['prompt_id']
    post = prompt_data['prompt']
    
    if verbose:
        print(f"\n  Prompt ID: {prompt_id}")
        print(f"  Post: {post[:100]}...")
    
    # Generate diverse rollouts
    if verbose:
        print(f"  Generating rollouts from {len(generator_models)} models...")
    
    rollouts, sources = generate_diverse_rollouts(
        clients, post, generator_models
    )
    
    if len(rollouts) < 2:
        print(f"  ERROR: Only generated {len(rollouts)} rollouts")
        return None
    
    if verbose:
        print(f"  Generated {len(rollouts)} rollouts: {sources}")
    
    # Run tournament
    elo_ratings, rewards, num_comparisons = run_elo_tournament(
        clients, post, rollouts, verbose
    )
    
    if verbose:
        print(f"  Tournament: {num_comparisons} valid comparisons")
        print(f"  Reward range: [{min(rewards):.3f}, {max(rewards):.3f}]")
    
    return RolloutResult(
        prompt_id=prompt_id,
        prompt=post,
        rollouts=rollouts,
        rollout_sources=sources,
        elo_ratings=elo_ratings,
        rewards=rewards,
        num_comparisons=num_comparisons
    )


def run_full_pipeline(
    clients: Dict,
    prompts: List[Dict],
    generator_models: List[str],
    verbose: bool = True
) -> Tuple[List[RolloutResult], List[TrainingExample]]:

    all_results = []
    training_examples = []
    
    n_models = len(generator_models)
    n_comparisons = n_models * (n_models - 1) // 2
    
    print(f"\n{'='*60}")
    print(f"ROLLOUT GENERATION AND TOURNAMENT PIPELINE")
    print(f"{'='*60}")
    print(f"Generator models: {generator_models}")
    print(f"Judge ensemble: {list(JUDGE_MODELS.keys())}")
    print(f"Num prompts: {len(prompts)}")
    print(f"Rollouts per prompt: {n_models}")
    print(f"Comparisons per prompt: {n_comparisons}")
    print(f"{'='*60}")
    
    for i, prompt_data in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Processing prompt...")
        
        result = process_single_prompt(
            clients, prompt_data, generator_models, verbose
        )
        
        if result is not None:
            all_results.append(result)
            
            # Create training examples
            indexed_rewards = list(enumerate(result.rewards))
            sorted_by_reward = sorted(indexed_rewards, key=lambda x: x[1], reverse=True)
            
            for rank, (orig_idx, reward) in enumerate(sorted_by_reward, start=1):
                example = TrainingExample(
                    prompt_id=result.prompt_id,
                    prompt=result.prompt,
                    response=result.rollouts[orig_idx],
                    source_model=result.rollout_sources[orig_idx],
                    reward=reward,
                    rank=rank
                )
                training_examples.append(example)
    
    print_summary(all_results, training_examples)
    
    return all_results, training_examples


def print_summary(results: List[RolloutResult], training_examples: List[TrainingExample]):

    if not results:
        print("\nNo results.")
        return
    
    all_rewards = [ex.reward for ex in training_examples]
    
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Prompts processed: {len(results)}")
    print(f"Total training examples: {len(training_examples)}")
    print(f"\nReward Statistics:")
    print(f"  Min:  {min(all_rewards):.4f}")
    print(f"  Max:  {max(all_rewards):.4f}")
    print(f"  Mean: {sum(all_rewards)/len(all_rewards):.4f}")
    
    # Model performance
    model_rewards = {}
    for ex in training_examples:
        if ex.source_model not in model_rewards:
            model_rewards[ex.source_model] = []
        model_rewards[ex.source_model].append(ex.reward)
    
    print(f"\nAverage Reward by Generator Model:")
    for model, rewards in sorted(model_rewards.items(), key=lambda x: -sum(x[1])/len(x[1])):
        avg = sum(rewards) / len(rewards)
        print(f"  {model}: {avg:.4f}")
    
    # Comparison success
    total_expected = sum(
        len(r.rollouts) * (len(r.rollouts) - 1) // 2 for r in results
    )
    total_actual = sum(r.num_comparisons for r in results)
    print(f"\nComparison Success: {total_actual}/{total_expected} ({total_actual/total_expected*100:.1f}%)")
    print(f"{'='*60}")


def save_results(
    results: List[RolloutResult],
    training_examples: List[TrainingExample],
    output_dir: str = DEFAULT_OUTPUT_DIR
):
    """Save results to JSON files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Full results
    results_file = output_path / REWARDS_OUTPUT_FILE
    with open(results_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nSaved results to: {results_file}")
    
    # Training data
    training_file = output_path / TRAINING_DATA_FILE
    with open(training_file, 'w') as f:
        json.dump([asdict(ex) for ex in training_examples], f, indent=2)
    print(f"Saved training data to: {training_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate rollouts and run ELO tournament for SHP dataset"
    )
    
    parser.add_argument(
        "--num-prompts", type=int, default=10,
        help="Number of prompts to process (default: 10)"
    )
    parser.add_argument(
        "--generators", type=str, nargs='+',
        default=['llama-3.3-70b', 'llama-3.1-70b', 'qwen-72b', 'qwen-7b', 
                 'mixtral-8x7b', 'mistral-7b', 'gpt-4o-mini'],
        help="Generator models to use"
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce verbosity"
    )
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # Initialize clients
    clients = {
        'openai': OpenAI(api_key=os.environ.get("OPENAI_API_KEY")),
        'together': Together(api_key=os.environ.get("TOGETHER_API_KEY")),
    }
    
    # Load prompts
    prompts = load_shp_prompts(sample_size=args.num_prompts, seed=args.seed)
    
    # Validate generator models
    valid_generators = [g for g in args.generators if g in GENERATOR_MODELS]
    if len(valid_generators) < 2:
        raise ValueError(f"Need at least 2 valid generators. Available: {list(GENERATOR_MODELS.keys())}")
    
    print(f"\nUsing generators: {valid_generators}")
    
    # Run pipeline
    results, training_examples = run_full_pipeline(
        clients, prompts, valid_generators, verbose=not args.quiet
    )
    
    # Save
    save_results(results, training_examples, args.output_dir)
    
    print(f"\n Pipeline complete!")
    print(f"  Training examples: {len(training_examples)}")


if __name__ == "__main__":
    main()