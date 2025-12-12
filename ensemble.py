import json
import random
import os
import argparse
from collections import Counter
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from datasets import load_dataset
from together import Together
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS = {
    'llama-70b': {
        'provider': 'together',
        'model_id': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
    },
    'gpt-4o-mini': {
        'provider': 'openai',
        'model_id': 'gpt-4o-mini-2024-07-18',
    },
    'gpt-4o-mini-ft': {
        'provider': 'openai',
        'model_id': 'ft:gpt-4o-mini-2024-07-18:nimbic-ai:shp-preference-expert:Ckd5lDAe',
    },
    'gpt-4o': {
        'provider': 'openai',
        'model_id': 'gpt-4o',
    },
}

# =============================================================================
# CLIENTS
# =============================================================================

def get_client(provider: str):
    if provider == 'together':
        return Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    elif provider == 'openai':
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    else:
        raise ValueError(f"Unknown provider: {provider}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_test_data(sample_size: Optional[int] = None, seed: int = 42) -> List[Dict]:
    """Load test data from SHP dataset."""
    print("Loading SHP dataset...")
    ds = load_dataset("stanfordnlp/SHP", split='test')
    
    test_data = []
    for example in ds:
        test_data.append({
            'post_id': example['post_id'],
            'post': example['history'],
            'response_A': example['human_ref_A'],
            'response_B': example['human_ref_B'],
            'true_label': example['labels'],
        })
    
    if sample_size and sample_size < len(test_data):
        random.seed(seed)
        test_data = random.sample(test_data, sample_size)
    
    print(f"Loaded {len(test_data)} test examples")
    return test_data


def format_prompt(post: str, response_a: str, response_b: str) -> str:
    """Simple baseline prompt."""
    return f"""Given the following post and two responses, determine which response is better.

POST: {post}

RESPONSE A: {response_a}

RESPONSE B: {response_b}

Which response is better? Respond with only "A" or "B"."""


# =============================================================================
# SINGLE MODEL INFERENCE
# =============================================================================

def get_prediction(
    client,
    model_id: str,
    prompt: str,
    temperature: float = 0,
    return_logprobs: bool = False,
    max_tokens: int = 10
) -> Tuple[Optional[int], Optional[float]]:
    """
    Get a single prediction from a model.
    
    Returns:
        (prediction, confidence) where prediction is 1 for A, 0 for B
    """
    try:
        kwargs = {
            'model': model_id,
            'messages': [{"role": "user", "content": prompt}],
            'max_tokens': max_tokens,
            'temperature': temperature,
        }
        
        # OpenAI supports logprobs
        if return_logprobs and 'gpt' in model_id:
            kwargs['logprobs'] = True
            kwargs['top_logprobs'] = 5
        
        response = client.chat.completions.create(**kwargs)
        
        answer = response.choices[0].message.content.strip().upper()
        
        # Parse answer
        if 'A' in answer and 'B' not in answer:
            prediction = 1
        elif 'B' in answer and 'A' not in answer:
            prediction = 0
        else:
            return None, None
        
        # Extract confidence if available
        confidence = None
        if return_logprobs and hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
            # Get probability of the chosen token
            try:
                logprob = response.choices[0].logprobs.content[0].logprob
                confidence = 2.718281828 ** logprob  # e^logprob
            except:
                pass
        
        return prediction, confidence
        
    except Exception as e:
        print(f"    Error: {e}")
        return None, None


# =============================================================================
# ENSEMBLE METHODS
# =============================================================================

def self_consistency(
    client,
    model_id: str,
    prompt: str,
    n_samples: int = 5,
    temperature: float = 0.7
) -> Tuple[Optional[int], float]:

    predictions = []
    
    for _ in range(n_samples):
        pred, _ = get_prediction(client, model_id, prompt, temperature=temperature)
        if pred is not None:
            predictions.append(pred)
    
    if not predictions:
        return None, 0.0
    
    # Majority vote
    counter = Counter(predictions)
    majority_pred, majority_count = counter.most_common(1)[0]
    agreement = majority_count / len(predictions)
    
    return majority_pred, agreement


def multi_model_ensemble(
    clients: Dict[str, any],
    models: List[str],
    prompt: str,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[Optional[int], Dict[str, int]]:

    predictions = {}
    weighted_votes = {0: 0.0, 1: 0.0}
    
    for model_key in models:
        model_config = MODELS[model_key]
        client = clients[model_config['provider']]
        
        pred, _ = get_prediction(client, model_config['model_id'], prompt, temperature=0)
        
        if pred is not None:
            predictions[model_key] = pred
            weight = weights.get(model_key, 1.0) if weights else 1.0
            weighted_votes[pred] += weight
    
    if not predictions:
        return None, {}
    
    # Weighted majority
    final_pred = 1 if weighted_votes[1] > weighted_votes[0] else 0
    
    return final_pred, predictions


def confidence_weighted_ensemble(
    clients: Dict[str, any],
    models: List[str],
    prompt: str
) -> Tuple[Optional[int], Dict[str, Tuple[int, float]]]:

    predictions = {}
    weighted_votes = {0: 0.0, 1: 0.0}
    
    for model_key in models:
        model_config = MODELS[model_key]
        client = clients[model_config['provider']]
        
        pred, conf = get_prediction(
            client, 
            model_config['model_id'], 
            prompt, 
            temperature=0,
            return_logprobs=True
        )
        
        if pred is not None:
            conf = conf if conf is not None else 0.5
            predictions[model_key] = (pred, conf)
            weighted_votes[pred] += conf
    
    if not predictions:
        return None, {}
    
    final_pred = 1 if weighted_votes[1] > weighted_votes[0] else 0
    
    return final_pred, predictions


def debate_deliberation(
    clients: Dict[str, any],
    post: str,
    response_a: str,
    response_b: str,
    advocate_model: str = 'gpt-4o-mini',
    judge_model: str = 'gpt-4o'
) -> Tuple[Optional[int], Dict]:
 
    # Advocate prompts
    advocate_a_prompt = f"""You are arguing that Response A is better. Make a compelling case.

POST: {post}

RESPONSE A: {response_a}

RESPONSE B: {response_b}

Argue why Response A is better in 2-3 sentences."""

    advocate_b_prompt = f"""You are arguing that Response B is better. Make a compelling case.

POST: {post}

RESPONSE A: {response_a}

RESPONSE B: {response_b}

Argue why Response B is better in 2-3 sentences."""

    # Get advocate arguments
    advocate_config = MODELS[advocate_model]
    advocate_client = clients[advocate_config['provider']]
    
    try:
        arg_a_response = advocate_client.chat.completions.create(
            model=advocate_config['model_id'],
            messages=[{"role": "user", "content": advocate_a_prompt}],
            max_tokens=150,
            temperature=0.7
        )
        argument_a = arg_a_response.choices[0].message.content.strip()
        
        arg_b_response = advocate_client.chat.completions.create(
            model=advocate_config['model_id'],
            messages=[{"role": "user", "content": advocate_b_prompt}],
            max_tokens=150,
            temperature=0.7
        )
        argument_b = arg_b_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    Advocate error: {e}")
        return None, {}
    
    # Judge prompt
    judge_prompt = f"""You are a judge evaluating which response is better based on two arguments.

POST: {post}

RESPONSE A: {response_a}

RESPONSE B: {response_b}

ARGUMENT FOR A: {argument_a}

ARGUMENT FOR B: {argument_b}

Based on the arguments and the original responses, which response is actually better? 
Respond with only "A" or "B"."""

    # Get judge decision
    judge_config = MODELS[judge_model]
    judge_client = clients[judge_config['provider']]
    
    pred, _ = get_prediction(judge_client, judge_config['model_id'], judge_prompt, temperature=0)
    
    return pred, {
        'argument_a': argument_a,
        'argument_b': argument_b,
        'judge_model': judge_model
    }


# =============================================================================
#  Position Debiasing, Chain-of-Thought, Aspect-Based
# =============================================================================

def position_debiased_prediction(
    client,
    model_id: str,
    post: str,
    response_a: str,
    response_b: str
) -> Tuple[Optional[int], str]:

    # Original order: A first
    prompt1 = format_prompt(post, response_a, response_b)
    pred1, _ = get_prediction(client, model_id, prompt1, temperature=0)
    
    # Swapped order: B first (presented as A)
    prompt2 = format_prompt(post, response_b, response_a)
    pred2, _ = get_prediction(client, model_id, prompt2, temperature=0)
    
    if pred1 is None or pred2 is None:
        return None, 'error'
    
    # Adjust pred2: if model said "A" with swapped order, it means original B
    pred2_adjusted = 1 - pred2
    
    if pred1 == pred2_adjusted:
        return pred1, 'consistent'
    else:
        return None, 'inconsistent'


def chain_of_thought_prediction(
    client,
    model_id: str,
    post: str,
    response_a: str,
    response_b: str
) -> Tuple[Optional[int], str]:

    prompt = f"""Given the following post and two responses, determine which response is better.

POST: {post}

RESPONSE A: {response_a}

RESPONSE B: {response_b}

Think step by step:
1. What is the post asking for or expressing?
2. How well does Response A address this?
3. How well does Response B address this?
4. Which response is more helpful, relevant, and well-written?

Based on your analysis, which response is better? End your response with exactly "ANSWER: A" or "ANSWER: B"."""

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0
        )
        
        full_response = response.choices[0].message.content.strip()
        
        # Parse answer from response
        pred = None
        if "ANSWER: A" in full_response or "ANSWER:A" in full_response:
            pred = 1
        elif "ANSWER: B" in full_response or "ANSWER:B" in full_response:
            pred = 0
        else:
            # Fallback: check last line
            last_line = full_response.split('\n')[-1].upper()
            if 'A' in last_line and 'B' not in last_line:
                pred = 1
            elif 'B' in last_line and 'A' not in last_line:
                pred = 0
        
        return pred, full_response
        
    except Exception as e:
        print(f"    CoT Error: {e}")
        return None, ""


def aspect_based_prediction(
    client,
    model_id: str,
    post: str,
    response_a: str,
    response_b: str
) -> Tuple[Optional[int], Dict[str, str]]:
 
    ASPECTS = [
        "Which response is more helpful to the original poster?",
        "Which response is more relevant to what was asked?",
        "Which response is better written and clearer?",
        "Which response would you recommend to a friend?",
    ]
    
    votes = []
    aspect_results = {}
    
    for aspect in ASPECTS:
        prompt = f"""POST: {post}

RESPONSE A: {response_a}

RESPONSE B: {response_b}

{aspect} Answer with only "A" or "B"."""

        pred, _ = get_prediction(client, model_id, prompt, temperature=0)
        
        if pred is not None:
            votes.append(pred)
            aspect_results[aspect] = 'A' if pred == 1 else 'B'
    
    if len(votes) < 2:
        return None, aspect_results
    
    # Majority vote
    final_pred = 1 if sum(votes) > len(votes) / 2 else 0
    
    return final_pred, aspect_results


def position_debiased_with_fallback(
    client,
    model_id: str,
    post: str,
    response_a: str,
    response_b: str,
    fallback: str = 'first'
) -> Tuple[Optional[int], str]:
 
    # Original order
    prompt1 = format_prompt(post, response_a, response_b)
    pred1, _ = get_prediction(client, model_id, prompt1, temperature=0)
    
    # Swapped order
    prompt2 = format_prompt(post, response_b, response_a)
    pred2, _ = get_prediction(client, model_id, prompt2, temperature=0)
    
    if pred1 is None or pred2 is None:
        return None, 'error'
    
    pred2_adjusted = 1 - pred2
    
    if pred1 == pred2_adjusted:
        return pred1, 'consistent'
    else:
        if fallback == 'first':
            return pred1, 'inconsistent_fallback_first'
        elif fallback == 'random':
            return random.choice([pred1, pred2_adjusted]), 'inconsistent_fallback_random'
        else:  # skip
            return None, 'inconsistent_skip'


# =============================================================================
# COMBINED METHODS
# =============================================================================

def multi_model_position_debiased(
    clients: Dict[str, any],
    models: List[str],
    post: str,
    response_a: str,
    response_b: str
) -> Tuple[Optional[int], Dict]:
 
    predictions = {}
    votes = {0: 0, 1: 0}
    
    for model_key in models:
        model_config = MODELS[model_key]
        client = clients[model_config['provider']]
        
        pred, status = position_debiased_prediction(
            client, model_config['model_id'],
            post, response_a, response_b
        )
        
        predictions[model_key] = {'prediction': pred, 'status': status}
        
        if pred is not None:
            votes[pred] += 1
    
    if votes[0] == 0 and votes[1] == 0:
        return None, predictions
    
    final_pred = 1 if votes[1] > votes[0] else 0
    
    return final_pred, predictions


def self_consistency_with_cot(
    client,
    model_id: str,
    post: str,
    response_a: str,
    response_b: str,
    n_samples: int = 5,
    temperature: float = 0.7
) -> Tuple[Optional[int], float, List[str]]:

    predictions = []
    reasonings = []
    
    for _ in range(n_samples):
        pred, reasoning = chain_of_thought_prediction(
            client, model_id, post, response_a, response_b
        )
        # Need to use temperature > 0 for diversity
        # Modify the function call to use temperature
        prompt = f"""Given the following post and two responses, determine which response is better.

POST: {post}

RESPONSE A: {response_a}

RESPONSE B: {response_b}

Think step by step:
1. What is the post asking for or expressing?
2. How well does Response A address this?
3. How well does Response B address this?
4. Which response is more helpful, relevant, and well-written?

Based on your analysis, which response is better? End your response with exactly "ANSWER: A" or "ANSWER: B"."""

        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=temperature
            )
            
            full_response = response.choices[0].message.content.strip()
            
            pred = None
            if "ANSWER: A" in full_response or "ANSWER:A" in full_response:
                pred = 1
            elif "ANSWER: B" in full_response or "ANSWER:B" in full_response:
                pred = 0
            else:
                last_line = full_response.split('\n')[-1].upper()
                if 'A' in last_line and 'B' not in last_line:
                    pred = 1
                elif 'B' in last_line and 'A' not in last_line:
                    pred = 0
            
            if pred is not None:
                predictions.append(pred)
                reasonings.append(full_response[:200])
                
        except Exception as e:
            continue
    
    if not predictions:
        return None, 0.0, []
    
    counter = Counter(predictions)
    majority_pred, majority_count = counter.most_common(1)[0]
    agreement = majority_count / len(predictions)
    
    return majority_pred, agreement, reasonings


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_method(
    test_data: List[Dict],
    method: str,
    clients: Dict[str, any],
    **kwargs
) -> Dict:
  
    correct = 0
    total = 0
    results = []
    
    # Track extra stats for some methods
    consistent_count = 0
    
    print(f"\n{'='*60}")
    print(f"EVALUATING: {method.upper()}")
    print(f"{'='*60}")
    
    for i, example in enumerate(test_data):
        if (i + 1) % 10 == 0:
            acc_str = f" (acc: {correct/total:.3f})" if total > 0 else ""
            extra_str = ""
            if method == 'position_debiased':
                extra_str = f", consistent: {consistent_count}/{i+1}"
            print(f"  Processing {i+1}/{len(test_data)}...{acc_str}{extra_str}")
        
        prompt = format_prompt(
            example['post'],
            example['response_A'],
            example['response_B']
        )
        
        pred = None
        extra_info = {}
        
        # === BASELINE ===
        if method == 'baseline':
            model_key = kwargs.get('model', 'gpt-4o-mini')
            model_config = MODELS[model_key]
            client = clients[model_config['provider']]
            pred, _ = get_prediction(client, model_config['model_id'], prompt, temperature=0)
        
        # === SELF-CONSISTENCY ===
        elif method == 'self_consistency':
            model_key = kwargs.get('model', 'gpt-4o-mini')
            model_config = MODELS[model_key]
            client = clients[model_config['provider']]
            n_samples = kwargs.get('n_samples', 5)
            temperature = kwargs.get('temperature', 0.7)
            pred, agreement = self_consistency(
                client, model_config['model_id'], prompt, 
                n_samples=n_samples, temperature=temperature
            )
            extra_info['agreement'] = agreement
        
        # === MULTI-MODEL ===
        elif method == 'multi_model':
            models = kwargs.get('models', ['gpt-4o-mini', 'llama-70b'])
            weights = kwargs.get('weights', None)
            pred, individual = multi_model_ensemble(clients, models, prompt, weights)
            extra_info['individual_predictions'] = individual
        
        # === CONFIDENCE WEIGHTED ===
        elif method == 'confidence_weighted':
            models = kwargs.get('models', ['gpt-4o-mini', 'gpt-4o'])
            pred, individual = confidence_weighted_ensemble(clients, models, prompt)
            extra_info['individual_predictions'] = {k: {'pred': v[0], 'conf': v[1]} for k, v in individual.items()}
        
        # === DEBATE ===
        elif method == 'debate':
            advocate = kwargs.get('advocate_model', 'gpt-4o-mini')
            judge = kwargs.get('judge_model', 'gpt-4o')
            pred, debate_info = debate_deliberation(
                clients,
                example['post'],
                example['response_A'],
                example['response_B'],
                advocate_model=advocate,
                judge_model=judge
            )
            extra_info['debate'] = debate_info
        
        # === POSITION DEBIASED ===
        elif method == 'position_debiased':
            model_key = kwargs.get('model', 'gpt-4o-mini')
            model_config = MODELS[model_key]
            client = clients[model_config['provider']]
            fallback = kwargs.get('fallback', 'skip')
            
            pred, status = position_debiased_with_fallback(
                client, model_config['model_id'],
                example['post'], example['response_A'], example['response_B'],
                fallback=fallback
            )
            extra_info['status'] = status
            if 'consistent' in status:
                consistent_count += 1
        
        # === CHAIN OF THOUGHT ===
        elif method == 'cot':
            model_key = kwargs.get('model', 'gpt-4o-mini')
            model_config = MODELS[model_key]
            client = clients[model_config['provider']]
            pred, reasoning = chain_of_thought_prediction(
                client, model_config['model_id'],
                example['post'], example['response_A'], example['response_B']
            )
            extra_info['reasoning'] = reasoning[:300] if reasoning else ""
        
        # === ASPECT-BASED ===
        elif method == 'aspect_based':
            model_key = kwargs.get('model', 'gpt-4o-mini')
            model_config = MODELS[model_key]
            client = clients[model_config['provider']]
            pred, aspect_votes = aspect_based_prediction(
                client, model_config['model_id'],
                example['post'], example['response_A'], example['response_B']
            )
            extra_info['aspect_votes'] = aspect_votes
        
        # === MULTI-MODEL POSITION DEBIASED ===
        elif method == 'multi_model_debiased':
            models = kwargs.get('models', ['gpt-4o-mini', 'llama-70b'])
            pred, model_results = multi_model_position_debiased(
                clients, models,
                example['post'], example['response_A'], example['response_B']
            )
            extra_info['model_results'] = model_results
        
        # === SELF-CONSISTENCY WITH COT ===
        elif method == 'self_consistency_cot':
            model_key = kwargs.get('model', 'gpt-4o-mini')
            model_config = MODELS[model_key]
            client = clients[model_config['provider']]
            n_samples = kwargs.get('n_samples', 5)
            temperature = kwargs.get('temperature', 0.7)
            pred, agreement, reasonings = self_consistency_with_cot(
                client, model_config['model_id'],
                example['post'], example['response_A'], example['response_B'],
                n_samples=n_samples, temperature=temperature
            )
            extra_info['agreement'] = agreement
            extra_info['num_reasonings'] = len(reasonings)
        
        # Record result
        if pred is not None:
            is_correct = (pred == example['true_label'])
            correct += int(is_correct)
            total += 1
            
            results.append({
                'post_id': example['post_id'],
                'prediction': pred,
                'true_label': example['true_label'],
                'correct': is_correct,
                **extra_info
            })
        else:
            results.append({
                'post_id': example['post_id'],
                'prediction': None,
                'true_label': example['true_label'],
                'correct': None,
                **extra_info
            })
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {method.upper()}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    if method == 'position_debiased':
        print(f"Consistency rate: {consistent_count/len(test_data):.4f} ({consistent_count}/{len(test_data)})")
    print(f"{'='*60}")
    
    output = {
        'method': method,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'params': {k: v for k, v in kwargs.items() if not callable(v)},
        'results': results
    }
    
    if method == 'position_debiased':
        output['consistency_rate'] = consistent_count / len(test_data)
        output['consistent_count'] = consistent_count
    
    return output


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ensemble methods for preference prediction")
    
    parser.add_argument(
        "--method", type=str, default="baseline",
        choices=[
            'baseline', 'self_consistency', 'multi_model', 'confidence_weighted', 
            'debate', 'position_debiased', 'cot', 'aspect_based',
            'multi_model_debiased', 'self_consistency_cot', 'all'
        ],
        help="Ensemble method to use"
    )
    parser.add_argument("--sample-size", type=int, default=100, help="Number of test examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Self-consistency params
    parser.add_argument("--n-samples", type=int, default=5, help="Samples for self-consistency")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    
    # Model selection
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model for single-model methods")
    parser.add_argument(
        "--models", type=str, nargs='+', 
        default=['gpt-4o-mini', 'llama-70b'],
        help="Models for ensemble methods"
    )
    
    # Position debiasing
    parser.add_argument(
        "--fallback", type=str, default="skip",
        choices=['skip', 'first', 'random'],
        help="Fallback strategy for inconsistent position-debiased predictions"
    )
    
    args = parser.parse_args()
    
    # Load data
    test_data = load_test_data(sample_size=args.sample_size, seed=args.seed)
    
    # Initialize clients
    clients = {
        'openai': get_client('openai'),
        'together': get_client('together')
    }
    
    # Determine which methods to run
    if args.method == 'all':
        methods_to_run = [
            'baseline', 'self_consistency', 'multi_model', 'debate',
            'position_debiased', 'cot', 'aspect_based'
        ]
    else:
        methods_to_run = [args.method]
    
    # Run evaluations
    all_results = {}
    
    for method in methods_to_run:
        kwargs = {
            'model': args.model,
            'models': args.models,
            'n_samples': args.n_samples,
            'temperature': args.temperature,
            'fallback': args.fallback,
        }
        
        # Method-specific defaults
        if method == 'debate':
            kwargs['advocate_model'] = 'gpt-4o-mini'
            kwargs['judge_model'] = 'gpt-4o'
        
        result = evaluate_method(test_data, method, clients, **kwargs)
        all_results[method] = result
    
    # Print comparison
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        for method, result in sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            extra = ""
            if 'consistency_rate' in result:
                extra = f" (consistency: {result['consistency_rate']:.2f})"
            print(f"  {method}: {result['accuracy']:.4f}{extra}")
    
    # Save results
    output_file = f"ensemble_results_{args.method}.json"
    with open(output_file, 'w') as f:
        # Remove detailed results for readability
        summary = {k: {kk: vv for kk, vv in v.items() if kk != 'results'} for k, v in all_results.items()}
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()