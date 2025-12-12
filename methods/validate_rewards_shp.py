import json
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import List, Dict


# =============================================================================
# QUALITY SIGNALS
# =============================================================================

def compute_quality_signals(text: str) -> Dict:

    sentences = re.split(r'[.!?]+', text)
    words = text.split()
    
    return {
        'length': len(text),
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_sentence_length': len(words) / max(len(sentences), 1),
        'has_code': 1 if '```' in text or 'def ' in text or 'import ' in text else 0,
        'has_explanation': 1 if any(w in text.lower() for w in 
            ['because', 'therefore', 'thus', 'since', 'this means', 'in other words']) else 0,
        'has_steps': 1 if any(p in text for p in 
            ['1.', '1)', 'Step 1', 'First,', 'Firstly', '- First']) else 0,
        'has_list': 1 if any(p in text for p in ['•', '-', '*', '1.', '2.', '3.']) else 0,
        'question_count': text.count('?'),
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_rewards_data(filepath: str) -> List[Dict]:
    with open(filepath) as f:
        return json.load(f)


def load_training_data(filepath: str) -> List[Dict]:
    with open(filepath) as f:
        return json.load(f)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_model_performance(data: List[Dict]) -> Dict:

    model_rewards = defaultdict(list)
    model_ranks = defaultdict(list)
    model_elos = defaultdict(list)
    
    for item in data:
        sources = item.get('rollout_sources', [])
        rewards = item['rewards']
        elos = item['elo_ratings']
        
        # Calculate ranks for this prompt
        indexed_rewards = list(enumerate(rewards))
        sorted_by_reward = sorted(indexed_rewards, key=lambda x: x[1], reverse=True)
        ranks = [0] * len(rewards)
        for rank, (idx, _) in enumerate(sorted_by_reward, start=1):
            ranks[idx] = rank
        
        for i, source in enumerate(sources):
            model_rewards[source].append(rewards[i])
            model_ranks[source].append(ranks[i])
            model_elos[source].append(elos[i])
    
    # Compute statistics
    model_stats = {}
    for model in model_rewards.keys():
        rewards = model_rewards[model]
        ranks = model_ranks[model]
        elos = model_elos[model]
        
        model_stats[model] = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_rank': np.mean(ranks),
            'avg_elo': np.mean(elos),
            'win_rate': sum(1 for r in ranks if r == 1) / len(ranks),
            'top3_rate': sum(1 for r in ranks if r <= 3) / len(ranks),
            'count': len(rewards),
        }
    
    return model_stats


def analyze_quality_correlations(data: List[Dict]) -> Dict:

    all_rollouts = []
    
    for item in data:
        sources = item.get('rollout_sources', [f'model_{i}' for i in range(len(item['rollouts']))])
        
        for i, (rollout, reward) in enumerate(zip(item['rollouts'], item['rewards'])):
            signals = compute_quality_signals(rollout)
            all_rollouts.append({
                'reward': reward,
                'source': sources[i] if i < len(sources) else 'unknown',
                **signals
            })
    
    rewards = np.array([r['reward'] for r in all_rollouts])
    
    correlations = {}
    signal_names = ['length', 'word_count', 'sentence_count', 'avg_sentence_length',
                    'has_explanation', 'has_steps', 'has_list']
    
    for signal in signal_names:
        values = np.array([r[signal] for r in all_rollouts])
        if np.std(values) > 0:
            corr = np.corrcoef(rewards, values)[0, 1]
        else:
            corr = 0.0
        correlations[signal] = corr
    
    return correlations, all_rollouts


def get_reward_distribution_stats(data: List[Dict]) -> Dict:

    all_rewards = []
    all_elos = []
    
    for item in data:
        all_rewards.extend(item['rewards'])
        all_elos.extend(item['elo_ratings'])
    
    rewards = np.array(all_rewards)
    elos = np.array(all_elos)
    
    return {
        'total_rollouts': len(rewards),
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'median_reward': np.median(rewards),
        'high_count': sum(rewards > 0.7),
        'medium_count': sum((rewards >= 0.3) & (rewards <= 0.7)),
        'low_count': sum(rewards < 0.3),
        'elo_reward_corr': np.corrcoef(elos, rewards)[0, 1],
        'all_rewards': rewards,
        'all_elos': elos,
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualizations(data: List[Dict], model_stats: Dict, dist_stats: Dict, 
                          output_prefix: str = "shp_reward_analysis"):
    
    # Figure 1: Model Performance Comparison
    fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4))
    
    # Sort models by average reward
    sorted_models = sorted(model_stats.keys(), key=lambda x: model_stats[x]['avg_reward'], reverse=True)
    
    # Plot 1a: Average Reward by Model
    ax = axes1[0]
    models = sorted_models
    avg_rewards = [model_stats[m]['avg_reward'] for m in models]
    std_rewards = [model_stats[m]['std_reward'] for m in models]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    bars = ax.bar(range(len(models)), avg_rewards, yerr=std_rewards, 
                  color=colors, edgecolor='black', capsize=3)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace('-', '\n') for m in models], fontsize=8)
    ax.set_ylabel('Average Reward')
    ax.set_title('Average Reward by Generator Model')
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline (0.5)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 1b: Win Rate by Model
    ax = axes1[1]
    win_rates = [model_stats[m]['win_rate'] * 100 for m in models]
    top3_rates = [model_stats[m]['top3_rate'] * 100 for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, win_rates, width, label='Win Rate (Rank 1)', color='gold', edgecolor='black')
    ax.bar(x + width/2, top3_rates, width, label='Top 3 Rate', color='silver', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('-', '\n') for m in models], fontsize=8)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Win Rate & Top 3 Rate by Model')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 1c: Average Rank by Model (lower is better)
    ax = axes1[2]
    avg_ranks = [model_stats[m]['avg_rank'] for m in models]
    
    bars = ax.bar(range(len(models)), avg_ranks, color=colors, edgecolor='black')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace('-', '\n') for m in models], fontsize=8)
    ax.set_ylabel('Average Rank (lower = better)')
    ax.set_title('Average Rank by Generator Model')
    ax.invert_yaxis()  # Lower rank is better
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig1.savefig(f"{output_prefix}_models.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_models.png")
    
    # Figure 2: Reward Distribution Analysis
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
    
    all_rewards = dist_stats['all_rewards']
    all_elos = dist_stats['all_elos']
    
    # Plot 2a: Reward Distribution Histogram
    ax = axes2[0]
    ax.hist(all_rewards, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(all_rewards), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(all_rewards):.2f}')
    ax.axvline(np.median(all_rewards), color='orange', linestyle=':', linewidth=2,
               label=f'Median: {np.median(all_rewards):.2f}')
    ax.set_xlabel('Normalized Reward')
    ax.set_ylabel('Count')
    ax.set_title('Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2b: ELO vs Reward Correlation
    ax = axes2[1]
    ax.scatter(all_elos, all_rewards, alpha=0.6, s=40, c='steelblue', edgecolor='white')
    
    # Fit line
    z = np.polyfit(all_elos, all_rewards, 1)
    p = np.poly1d(z)
    x_line = np.linspace(all_elos.min(), all_elos.max(), 100)
    corr = np.corrcoef(all_elos, all_rewards)[0, 1]
    ax.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'r = {corr:.3f}')
    
    ax.set_xlabel('ELO Rating')
    ax.set_ylabel('Normalized Reward')
    ax.set_title('ELO vs Reward Correlation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2c: Reward Distribution by Quality Tier
    ax = axes2[2]
    tiers = ['High\n(>0.7)', 'Medium\n(0.3-0.7)', 'Low\n(<0.3)']
    counts = [dist_stats['high_count'], dist_stats['medium_count'], dist_stats['low_count']]
    colors = ['forestgreen', 'gold', 'tomato']
    
    bars = ax.bar(tiers, counts, color=colors, edgecolor='black')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{count}\n({100*count/len(all_rewards):.0f}%)', 
                ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('Count')
    ax.set_title('Reward Quality Tiers')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig2.savefig(f"{output_prefix}_distribution.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_distribution.png")
    
    # Figure 3: Reward by Model (Box Plot)
    fig3, ax = plt.subplots(figsize=(10, 5))
    
    model_reward_lists = []
    for item in data:
        sources = item.get('rollout_sources', [])
        rewards = item['rewards']
        for i, source in enumerate(sources):
            if source in sorted_models:
                while len(model_reward_lists) < sorted_models.index(source) + 1:
                    model_reward_lists.append([])
                model_reward_lists[sorted_models.index(source)].append(rewards[i])
    
    # Rebuild properly
    model_reward_dict = defaultdict(list)
    for item in data:
        sources = item.get('rollout_sources', [])
        rewards = item['rewards']
        for i, source in enumerate(sources):
            model_reward_dict[source].append(rewards[i])
    
    box_data = [model_reward_dict[m] for m in sorted_models]
    bp = ax.boxplot(box_data, labels=[m.replace('-', '\n') for m in sorted_models], 
                    patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Reward')
    ax.set_title('Reward Distribution by Generator Model')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig3.savefig(f"{output_prefix}_boxplot.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_boxplot.png")
    
    plt.close('all')


# =============================================================================
# REPORTING
# =============================================================================

def print_analysis_report(data: List[Dict], model_stats: Dict, correlations: Dict,
                          dist_stats: Dict, all_rollouts: List[Dict]):
    
    print("=" * 70)
    print("SHP ROLLOUT REWARDS - VALIDATION ANALYSIS")
    print("=" * 70)
    
    # 1. Overview
    print("\n1. OVERVIEW")
    print("-" * 50)
    print(f"  Prompts processed:    {len(data)}")
    print(f"  Total rollouts:       {dist_stats['total_rollouts']}")
    print(f"  Generator models:     {len(model_stats)}")
    print(f"  Mean reward:          {dist_stats['mean_reward']:.3f}")
    print(f"  Std reward:           {dist_stats['std_reward']:.3f}")
    
    # 2. Model Performance Ranking
    print("\n2. MODEL PERFORMANCE RANKING")
    print("-" * 50)
    print(f"  {'Model':<20s} {'Avg Reward':>10s} {'Win Rate':>10s} {'Top 3':>10s} {'Avg Rank':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    sorted_models = sorted(model_stats.keys(), key=lambda x: model_stats[x]['avg_reward'], reverse=True)
    for model in sorted_models:
        stats = model_stats[model]
        print(f"  {model:<20s} {stats['avg_reward']:>10.3f} {stats['win_rate']*100:>9.1f}% "
              f"{stats['top3_rate']*100:>9.1f}% {stats['avg_rank']:>10.1f}")
    
    # 3. Quality Signal Correlations
    print("\n3. QUALITY SIGNAL CORRELATIONS WITH REWARD")
    print("-" * 50)
    for signal, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        direction = "+" if corr > 0 else ""
        print(f"  {signal:<25s}: r = {direction}{corr:.3f}")
    
    # 4. Reward Distribution
    print("\n4. REWARD DISTRIBUTION")
    print("-" * 50)
    total = dist_stats['total_rollouts']
    print(f"  High (>0.7):     {dist_stats['high_count']:>3d} ({100*dist_stats['high_count']/total:>5.1f}%)")
    print(f"  Medium (0.3-0.7): {dist_stats['medium_count']:>3d} ({100*dist_stats['medium_count']/total:>5.1f}%)")
    print(f"  Low (<0.3):      {dist_stats['low_count']:>3d} ({100*dist_stats['low_count']/total:>5.1f}%)")
    print(f"\n  ELO-Reward Correlation: r = {dist_stats['elo_reward_corr']:.3f}")
    
    # 5. High vs Low Comparison
    print("\n5. HIGH vs LOW REWARD COMPARISON")
    print("-" * 50)
    
    sorted_rollouts = sorted(all_rollouts, key=lambda x: x['reward'], reverse=True)
    n = len(sorted_rollouts)
    top_25 = sorted_rollouts[:n//4]
    bottom_25 = sorted_rollouts[-n//4:]
    
    print(f"  {'Metric':<25s} {'Top 25%':>12s} {'Bottom 25%':>12s} {'Diff':>10s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    
    for signal in ['word_count', 'sentence_count', 'has_explanation', 'has_steps', 'has_list']:
        top_avg = np.mean([r[signal] for r in top_25])
        bot_avg = np.mean([r[signal] for r in bottom_25])
        diff = top_avg - bot_avg
        print(f"  {signal:<25s} {top_avg:>12.1f} {bot_avg:>12.1f} {diff:>+10.1f}")
    
    # 6. Model Distribution in Top/Bottom
    print("\n6. MODEL REPRESENTATION IN TOP vs BOTTOM 25%")
    print("-" * 50)
    
    top_models = defaultdict(int)
    bot_models = defaultdict(int)
    
    for r in top_25:
        top_models[r['source']] += 1
    for r in bottom_25:
        bot_models[r['source']] += 1
    
    all_models = set(top_models.keys()) | set(bot_models.keys())
    print(f"  {'Model':<20s} {'Top 25%':>10s} {'Bottom 25%':>12s}")
    print(f"  {'-'*20} {'-'*10} {'-'*12}")
    
    for model in sorted(all_models, key=lambda x: top_models.get(x, 0), reverse=True):
        print(f"  {model:<20s} {top_models.get(model, 0):>10d} {bot_models.get(model, 0):>12d}")
    
    # 7. Qualitative Examples
    print("\n7. QUALITATIVE EXAMPLES")
    print("-" * 50)
    
    print(f"\n>>> HIGHEST REWARD EXAMPLE (reward = {sorted_rollouts[0]['reward']:.3f})")
    print(f"    Model: {sorted_rollouts[0]['source']}")
    print(f"    Prompt: {sorted_rollouts[0].get('prompt', 'N/A')[:80]}...")
    print(f"    Response: {sorted_rollouts[0]['rollout'][:400]}...")
    
    print(f"\n>>> LOWEST REWARD EXAMPLE (reward = {sorted_rollouts[-1]['reward']:.3f})")
    print(f"    Model: {sorted_rollouts[-1]['source']}")
    print(f"    Prompt: {sorted_rollouts[-1].get('prompt', 'N/A')[:80]}...")
    print(f"    Response: {sorted_rollouts[-1]['rollout'][:400]}...")
    
    # 8. Key Insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    best_model = sorted_models[0]
    worst_model = sorted_models[-1]
    
    print(f"\n  • Best performing model: {best_model}")
    print(f"    - Avg reward: {model_stats[best_model]['avg_reward']:.3f}")
    print(f"    - Win rate: {model_stats[best_model]['win_rate']*100:.1f}%")
    
    print(f"\n  • Worst performing model: {worst_model}")
    print(f"    - Avg reward: {model_stats[worst_model]['avg_reward']:.3f}")
    print(f"    - Win rate: {model_stats[worst_model]['win_rate']*100:.1f}%")
    
    # Check if larger models perform better
    large_models = ['llama-3.3-70b', 'llama-3.1-70b', 'qwen-72b', 'gpt-4o-mini']
    small_models = ['qwen-7b', 'mistral-7b', 'mixtral-8x7b']
    
    large_avg = np.mean([model_stats[m]['avg_reward'] for m in large_models if m in model_stats])
    small_avg = np.mean([model_stats[m]['avg_reward'] for m in small_models if m in model_stats])
    
    print(f"\n  • Model size effect:")
    print(f"    - Large models avg reward: {large_avg:.3f}")
    print(f"    - Small models avg reward: {small_avg:.3f}")
    print(f"    - Difference: {large_avg - small_avg:+.3f}")
    
    # Top correlation
    top_corr_signal = max(correlations.keys(), key=lambda x: abs(correlations[x]))
    print(f"\n  • Strongest quality signal correlation: {top_corr_signal}")
    print(f"    - r = {correlations[top_corr_signal]:.3f}")
    
    print("\n" + "=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate and visualize SHP rollout rewards"
    )
    parser.add_argument(
        "--input", type=str, default="data/shp_rollout_rewards.json",
        help="Input rewards JSON file"
    )
    parser.add_argument(
        "--output-prefix", type=str, default="shp_reward_analysis",
        help="Prefix for output files"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating plots"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    data = load_rewards_data(args.input)
    print(f"  Loaded {len(data)} prompt results")
    
    # Run analyses
    model_stats = analyze_model_performance(data)
    correlations, all_rollouts = analyze_quality_correlations(data)
    dist_stats = get_reward_distribution_stats(data)
    
    # Add prompt info to rollouts for qualitative examples
    rollout_idx = 0
    for item in data:
        sources = item.get('rollout_sources', [])
        for i in range(len(item['rollouts'])):
            if rollout_idx < len(all_rollouts):
                all_rollouts[rollout_idx]['prompt'] = item['prompt'][:100] + '...'
                all_rollouts[rollout_idx]['rollout'] = item['rollouts'][i]
            rollout_idx += 1
    
    # Print report
    print_analysis_report(data, model_stats, correlations, dist_stats, all_rollouts)
    
    # Generate visualizations
    if not args.no_plots:
        print("\nGenerating visualizations...")
        create_visualizations(data, model_stats, dist_stats, args.output_prefix)
        print(f"\nAll visualizations saved with prefix: {args.output_prefix}_*.png")


if __name__ == "__main__":
    main()
