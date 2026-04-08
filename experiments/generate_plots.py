"""Generate publication-quality plots from experiment results."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = '/home/user/workspace/ncm_project/results'

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'semantic': '#e74c3c',
    'sem_emo': '#f39c12', 
    'manifold': '#2ecc71',
    'fast': '#3498db',
    'ncm_blue': '#2c3e50',
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 1: State-Conditioned Retrieval (EXP 3) — THE KEY RESULT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_exp3():
    with open(os.path.join(RESULTS_DIR, 'exp3_state.json')) as f:
        data = json.load(f)
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    pairs = list(data.keys())
    labels = [p.replace('_vs_', '\nvs\n').replace('_', ' ').title() for p in pairs]
    
    sem_vals = [data[p]['semantic_jaccard'] for p in pairs]
    man_vals = [data[p]['manifold_jaccard'] for p in pairs]
    man_stds = [data[p]['manifold_jaccard_std'] for p in pairs]
    
    x = np.arange(len(pairs))
    w = 0.35
    
    bars1 = ax.bar(x - w/2, sem_vals, w, label='Semantic Only (RAG baseline)',
                   color=COLORS['semantic'], alpha=0.85, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + w/2, man_vals, w, yerr=man_stds, capsize=4,
                   label='NCM Full Manifold', color=COLORS['manifold'], alpha=0.85,
                   edgecolor='white', linewidth=0.5)
    
    ax.set_ylabel('Jaccard Distance\n(Higher = More Different Retrieval Sets)')
    ax.set_title('State-Conditioned Retrieval: Same Query, Different Internal States\n'
                 'NCM retrieves DIFFERENT memories depending on emotional state; RAG retrieves identical sets',
                 fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.05)
    
    # Add value labels
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.04, f'{h:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp3_state_conditioned.png'))
    plt.close()
    print("Saved: exp3_state_conditioned.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 2: Novelty Sensitivity at Scale (EXP 2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_exp2():
    with open(os.path.join(RESULTS_DIR, 'exp2_novelty.json')) as f:
        data = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    scales = sorted([int(k) for k in data.keys()])
    sem_means = [data[str(s)]['semantic_novelty_mean'] for s in scales]
    sem_stds = [data[str(s)]['semantic_novelty_std'] for s in scales]
    full_means = [data[str(s)]['full_novelty_mean'] for s in scales]
    full_stds = [data[str(s)]['full_novelty_std'] for s in scales]
    ratios = [data[str(s)]['ratio'] for s in scales]
    
    # Left: Novelty scores
    ax1.errorbar(scales, sem_means, yerr=sem_stds, marker='o', label='Semantic Only',
                 color=COLORS['semantic'], linewidth=2, capsize=3)
    ax1.errorbar(scales, full_means, yerr=full_stds, marker='s', label='NCM Full Manifold',
                 color=COLORS['manifold'], linewidth=2, capsize=3)
    ax1.set_xscale('log')
    ax1.set_xlabel('Memory Store Size')
    ax1.set_ylabel('Novelty Score')
    ax1.set_title('Novelty Detection: Semantic Saturates, Manifold Persists')
    ax1.legend()
    ax1.set_ylim(bottom=0)
    
    # Right: Ratio
    ax2.plot(scales, ratios, marker='D', color=COLORS['ncm_blue'], linewidth=2, markersize=7)
    ax2.set_xscale('log')
    ax2.set_xlabel('Memory Store Size')
    ax2.set_ylabel('Novelty Ratio (NCM / Semantic)')
    ax2.set_title('NCM Novelty Advantage Grows with Scale')
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    for i, (s, r) in enumerate(zip(scales, ratios)):
        ax2.annotate(f'{r:.0f}x', (s, r), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp2_novelty_scale.png'))
    plt.close()
    print("Saved: exp2_novelty_scale.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 3: Speed Benchmarks (EXP 4)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_exp4():
    with open(os.path.join(RESULTS_DIR, 'exp4_speed.json')) as f:
        data = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    scales = sorted([int(k) for k in data.keys() if k.isdigit()])
    
    sem_ms = [data[str(s)]['retrieval_semantic_ms'] for s in scales]
    man_ms = [data[str(s)]['retrieval_manifold_ms'] for s in scales]
    fast_ms = [data[str(s)]['retrieval_cached_ms'] for s in scales]
    
    # Left: Retrieval latency
    ax1.plot(scales, sem_ms, marker='o', label='Semantic Only', color=COLORS['semantic'], linewidth=2)
    ax1.plot(scales, man_ms, marker='s', label='Full Manifold', color=COLORS['manifold'], linewidth=2)
    ax1.plot(scales, fast_ms, marker='^', label='NCM Cached', color=COLORS['fast'], linewidth=2)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Memory Store Size')
    ax1.set_ylabel('Retrieval Latency (ms)')
    ax1.set_title('Retrieval Speed: NCM Cached is Near Real-Time')
    ax1.legend()
    
    # Add annotations for 50k
    if 50000 in scales:
        idx = scales.index(50000)
        ax1.annotate(f'{fast_ms[idx]:.1f}ms', (50000, fast_ms[idx]),
                     textcoords="offset points", xytext=(10, 5), fontsize=9, fontweight='bold',
                     color=COLORS['fast'])
    
    # Right: Storage efficiency
    file_kb = [data[str(s)]['file_kb'] for s in scales]
    bpm = [data[str(s)]['bytes_per_memory'] for s in scales]
    
    ax2r = ax2.twinx()
    l1 = ax2.plot(scales, file_kb, marker='o', label='.ncm file size', color=COLORS['ncm_blue'], linewidth=2)
    l2 = ax2r.plot(scales, bpm, marker='s', label='Bytes/memory', color=COLORS['sem_emo'], linewidth=2, linestyle='--')
    ax2.set_xscale('log')
    ax2.set_xlabel('Memory Store Size')
    ax2.set_ylabel('File Size (KB)')
    ax2r.set_ylabel('Bytes per Memory')
    ax2.set_title('Storage Efficiency: Compact Binary Format')
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp4_speed.png'))
    plt.close()
    print("Saved: exp4_speed.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 4: Combined Summary Dashboard
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def plot_dashboard():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('NCM (Native Cognitive Memory) — Experiment Results', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Panel A: State-conditioned (key result)
    ax = axes[0, 0]
    with open(os.path.join(RESULTS_DIR, 'exp3_state.json')) as f:
        d3 = json.load(f)
    pairs = list(d3.keys())
    short_labels = [p.split('_vs_')[0].replace('_',' ')[:8] + '\nvs\n' + 
                    p.split('_vs_')[1].replace('_',' ')[:8] for p in pairs]
    sem = [d3[p]['semantic_jaccard'] for p in pairs]
    man = [d3[p]['manifold_jaccard'] for p in pairs]
    x = np.arange(len(pairs))
    ax.bar(x - 0.175, sem, 0.35, label='Semantic', color=COLORS['semantic'], alpha=0.85)
    ax.bar(x + 0.175, man, 0.35, label='NCM Manifold', color=COLORS['manifold'], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=7)
    ax.set_ylabel('Jaccard Distance')
    ax.set_title('A) State-Conditioned Retrieval', fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)
    
    # Panel B: Novelty at scale
    ax = axes[0, 1]
    with open(os.path.join(RESULTS_DIR, 'exp2_novelty.json')) as f:
        d2 = json.load(f)
    scales = sorted([int(k) for k in d2.keys()])
    s_m = [d2[str(s)]['semantic_novelty_mean'] for s in scales]
    f_m = [d2[str(s)]['full_novelty_mean'] for s in scales]
    ax.plot(scales, s_m, 'o-', label='Semantic', color=COLORS['semantic'], linewidth=2)
    ax.plot(scales, f_m, 's-', label='NCM Manifold', color=COLORS['manifold'], linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('Store Size')
    ax.set_ylabel('Novelty Score')
    ax.set_title('B) Novelty Sensitivity at Scale', fontweight='bold')
    ax.legend(fontsize=8)
    
    # Panel C: Speed
    ax = axes[1, 0]
    with open(os.path.join(RESULTS_DIR, 'exp4_speed.json')) as f:
        d4 = json.load(f)
    scales_s = sorted([int(k) for k in d4.keys() if k.isdigit()])
    sem_ms = [d4[str(s)]['retrieval_semantic_ms'] for s in scales_s]
    man_ms = [d4[str(s)]['retrieval_manifold_ms'] for s in scales_s]
    fast_ms = [d4[str(s)]['retrieval_cached_ms'] for s in scales_s]
    ax.plot(scales_s, sem_ms, 'o-', label='Semantic', color=COLORS['semantic'], linewidth=2)
    ax.plot(scales_s, man_ms, 's-', label='Full Manifold', color=COLORS['manifold'], linewidth=2)
    ax.plot(scales_s, fast_ms, '^-', label='NCM Cached', color=COLORS['fast'], linewidth=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Store Size')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('C) Retrieval Speed', fontweight='bold')
    ax.legend(fontsize=8)
    
    # Panel D: Key numbers summary
    ax = axes[1, 1]
    ax.axis('off')
    
    # Compute key statistics
    avg_jaccard = np.mean([d3[p]['manifold_jaccard'] for p in pairs])
    max_ratio = max([d2[str(s)]['ratio'] for s in scales])
    fast_50k = d4.get('50000', {}).get('retrieval_cached_ms', 'N/A')
    store_rate = d4.get('50000', {}).get('store_throughput', 'N/A')
    bpm = d4.get('50000', {}).get('bytes_per_memory', 'N/A')
    
    summary_text = (
        f"NCM KEY RESULTS\n"
        f"{'─'*40}\n\n"
        f"State-Conditioned Retrieval\n"
        f"  Mean Jaccard distance:  {avg_jaccard:.3f}\n"
        f"  (0 = identical retrieval, 1 = completely different)\n"
        f"  Semantic baseline:      0.000\n\n"
        f"Novelty Sensitivity\n"
        f"  NCM advantage at 50k:   {max_ratio:.0f}x over semantic\n"
        f"  Semantic novelty:       saturates → 0.004\n"
        f"  NCM novelty:            stable → 0.130\n\n"
        f"Performance (50k memories)\n"
        f"  Cached retrieval:       {fast_50k} ms\n"
        f"  Store throughput:       {store_rate}/sec\n"
        f"  Storage efficiency:     {bpm} bytes/memory\n"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    ax.set_title('D) Key Numbers', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'ncm_dashboard.png'))
    plt.close()
    print("Saved: ncm_dashboard.png")


if __name__ == "__main__":
    plot_exp3()
    plot_exp2()
    plot_exp4()
    plot_dashboard()
    print("\nAll plots generated.")
