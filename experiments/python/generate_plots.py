"""Generate publication-quality plots from experiment results."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(ROOT_DIR, 'experiments', 'results', 'run_all_experiments')

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
    with open(os.path.join(RESULTS_DIR, 'exp3_state_conditioned.json')) as f:
        data = json.load(f)
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    pairs = list(data.keys())
    labels = [p.replace('_vs_', '\nvs\n').replace('_', ' ').title() for p in pairs]
    
    sem_vals = [data[p]['semantic_jaccard_mean'] for p in pairs]
    man_vals = [data[p]['manifold_jaccard_mean'] for p in pairs]
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
    with open(os.path.join(RESULTS_DIR, 'exp2_novelty_at_scale.json')) as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    scales = sorted([int(k) for k in data.keys()])
    sem_means = [data[str(s)]['semantic_novelty_mean'] for s in scales]
    sem_stds = [data[str(s)]['semantic_novelty_std'] for s in scales]
    full_means = [data[str(s)]['full_novelty_mean'] for s in scales]
    full_stds = [data[str(s)]['full_novelty_std'] for s in scales]

    # Left: Novelty scores
    ax1.errorbar(scales, sem_means, yerr=sem_stds, marker='o', label='Semantic Only',
                 color=COLORS['semantic'], linewidth=2, capsize=3)
    ax1.errorbar(scales, full_means, yerr=full_stds, marker='s', label='NCM Full Manifold',
                 color=COLORS['manifold'], linewidth=2, capsize=3)
    ax1.set_xscale('log')
    ax1.set_xlabel('Memory Store Size')
    ax1.set_ylabel('Novelty Score')
    ax1.set_title(f'Novelty vs store size: semantic falls to {sem_means[-1]:.1e},\n'
                  f'composite holds {full_means[-1]:.3f} at {scales[-1]:,} memories')
    ax1.legend()
    ax1.set_ylim(bottom=0)

    # Right: ratio of composite to semantic novelty.
    #
    # NOTE: this panel previously read a 'ratio' key that does not exist in the
    # results file, so it raised KeyError and the committed PNG was a stale
    # artifact. It also annotated each point with f'{r:.0f}x' under the title
    # "NCM Novelty Advantage Grows with Scale". On the actual measurements the
    # ratio is BELOW 1.0 at every scale up to 10,000 (0.62 to 0.83), meaning
    # semantic novelty is higher than composite over most of the range.
    #
    # At 100,000 memories semantic novelty is 8.94e-09, which is below float32
    # machine epsilon (1.19e-07): the nearest-neighbour cosine similarity has
    # saturated to 1.0 within the precision of the representation. The ratio
    # there (1.3e7) is a floating-point artifact, not a measured advantage, so
    # such points are excluded from the curve and labelled explicitly.
    EPS = float(np.finfo(np.float32).eps)
    ratio_scales, ratios, degenerate = [], [], []
    for s in scales:
        denom = data[str(s)]['semantic_novelty_mean']
        if denom > EPS:
            ratio_scales.append(s)
            ratios.append(data[str(s)]['full_novelty_mean'] / denom)
        else:
            degenerate.append((s, denom))

    ax2.plot(ratio_scales, ratios, marker='D', color=COLORS['ncm_blue'],
             linewidth=2, markersize=7)
    ax2.set_xscale('log')
    ax2.set_xlabel('Memory Store Size')
    ax2.set_ylabel('Novelty Ratio (composite / semantic)')
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.6)
    ax2.annotate('parity', xy=(scales[0], 1.0), xytext=(2, 4),
                 textcoords='offset points', fontsize=8, color='gray')
    ax2.set_ylim(0, max(ratios) * 1.35 if ratios else 1.5)

    crossover = next((s for s, r in zip(ratio_scales, ratios) if r > 1.0), None)
    if crossover and scales.index(crossover) > 0:
        prev = scales[scales.index(crossover) - 1]
        sub = f'crosses parity between {prev:,} and {crossover:,} memories'
    else:
        sub = 'parity crossing not observed'
    ax2.set_title(f'Composite is LESS novelty-sensitive below parity;\n{sub}')

    for s, r in zip(ratio_scales, ratios):
        ax2.annotate(f'{r:.2f}x', (s, r), textcoords="offset points",
                     xytext=(0, 9), ha='center', fontsize=8, fontweight='bold')

    if degenerate:
        ax2.set_xlim(right=max(scales) * 3.0)
        for s, denom in degenerate:
            ax2.annotate(
                f'{s:,}: ratio undefined\nsemantic novelty {denom:.2e}\n'
                f'is below float32 eps ({EPS:.2e})',
                xy=(s, 0), xycoords='data',
                textcoords="offset points", xytext=(-14, 30),
                ha='center', va='bottom', fontsize=7, color='#c0392b',
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.8),
            )

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'exp2_novelty_scale.png'))
    plt.close()
    print("Saved: exp2_novelty_scale.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOT 3: Speed Benchmarks (EXP 4)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_exp4():
    """
    Load the Experiment 4 results file.

    NOTE: this previously read 'exp4_speed.json' with keys
    retrieval_semantic_ms / retrieval_manifold_ms / retrieval_cached_ms. No
    such file or schema is produced any more, so both plots that used it were
    unregenerable and the committed PNGs were stale artifacts showing a
    three-arm framing that has since been retracted (the "Full Manifold" and
    "NCM Cached" arms ran identical code; see experiment_4_speed_benchmarks).
    """
    path = os.path.join(RESULTS_DIR, 'exp4_speed_benchmarks.json')
    with open(path) as f:
        data = json.load(f)
    scales = sorted(int(k) for k in data.keys() if k.isdigit())
    return data, scales


def _exp4_series(data, scales):
    sem_shipped = [data[str(s)]['retrieval_semantic_shipped_ms']['median_ms'] for s in scales]
    sem_pre = [data[str(s)]['retrieval_semantic_prebuilt_matrix_ms']['median_ms'] for s in scales]
    manifold = [data[str(s)]['retrieval_manifold_warm_cache_ms']['median_ms'] for s in scales]
    return sem_shipped, sem_pre, manifold


def plot_exp4():
    data, scales = _load_exp4()
    sem_shipped, sem_pre, manifold = _exp4_series(data, scales)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Retrieval latency. Three arms, cache state stated in the labels,
    # because the fair comparison is against the prebuilt-matrix baseline.
    ax1.plot(scales, sem_shipped, marker='o', label='Semantic, shipped (rebuilds matrix per call)',
             color=COLORS['semantic'], linewidth=2)
    ax1.plot(scales, sem_pre, marker='D', label='Semantic, prebuilt matrix (fair baseline)',
             color=COLORS['sem_emo'], linewidth=2, linestyle=':')
    ax1.plot(scales, manifold, marker='^', label='Composite manifold, warm cache',
             color=COLORS['fast'], linewidth=2)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Memory Store Size')
    ax1.set_ylabel('Median Retrieval Latency (ms)')
    ax1.set_title('Retrieval latency: composite costs 2.2x-4.9x\na semantic search doing the same work',
                  fontsize=11)
    ax1.legend(fontsize=8)

    if 100000 in scales:
        idx = scales.index(100000)
        ax1.annotate(f'{manifold[idx]:.1f} ms', (100000, manifold[idx]),
                     textcoords="offset points", xytext=(-42, 8), fontsize=9,
                     fontweight='bold', color=COLORS['fast'])

    # Right: Storage efficiency
    file_kb = [data[str(s)]['file_size_bytes'] / 1024.0 for s in scales]
    bpm = [data[str(s)]['bytes_per_memory'] for s in scales]

    ax2r = ax2.twinx()
    l1 = ax2.plot(scales, file_kb, marker='o', label='.ncm file size', color=COLORS['ncm_blue'], linewidth=2)
    l2 = ax2r.plot(scales, bpm, marker='s', label='Bytes/memory', color=COLORS['sem_emo'], linewidth=2, linestyle='--')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Memory Store Size')
    ax2.set_ylabel('File Size (KB)')
    ax2r.set_ylabel('Bytes per Memory')
    ax2r.set_ylim(0, max(bpm) * 1.4)
    ax2.set_title('Storage: 297 bytes/memory, compressed', fontsize=11)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left', fontsize=9)

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
    with open(os.path.join(RESULTS_DIR, 'exp3_state_conditioned.json')) as f:
        d3 = json.load(f)
    pairs = list(d3.keys())
    short_labels = [p.split('_vs_')[0].replace('_',' ')[:8] + '\nvs\n' + 
                    p.split('_vs_')[1].replace('_',' ')[:8] for p in pairs]
    sem = [d3[p]['semantic_jaccard_mean'] for p in pairs]
    man = [d3[p]['manifold_jaccard_mean'] for p in pairs]
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
    with open(os.path.join(RESULTS_DIR, 'exp2_novelty_at_scale.json')) as f:
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
    d4, scales_s = _load_exp4()
    sem_shipped, sem_pre, manifold = _exp4_series(d4, scales_s)
    ax.plot(scales_s, sem_shipped, 'o-', label='Semantic (shipped)', color=COLORS['semantic'], linewidth=2)
    ax.plot(scales_s, sem_pre, 'D:', label='Semantic (prebuilt)', color=COLORS['sem_emo'], linewidth=2)
    ax.plot(scales_s, manifold, '^-', label='Composite (warm)', color=COLORS['fast'], linewidth=2)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Store Size')
    ax.set_ylabel('Median Latency (ms)')
    ax.set_title('C) Retrieval Latency', fontweight='bold')
    ax.legend(fontsize=8)
    
    # Panel D: Key numbers summary.
    #
    # This panel previously read d4['50000']['retrieval_cached_ms'] and
    # ['store_throughput'] through .get(..., 'N/A'), and neither key has ever
    # existed in this schema, so the panel silently rendered "N/A ms" instead of
    # failing. It also hardcoded the strings "Semantic novelty: saturates ->
    # 0.004" and "NCM novelty: stable -> 0.130", which do not match the exp2
    # results file (0.000 and 0.119 at 100k), and printed a
    # full/semantic novelty ratio as an "NCM advantage" multiplier. That ratio
    # is meaningless at 100k because semantic novelty there is exactly 0.0, and
    # it is below 1.0 at every scale up to 10k, meaning semantic novelty is
    # HIGHER than composite over most of the range. All figures below are now
    # read from the results files.
    ax = axes[1, 1]
    ax.axis('off')

    avg_jaccard = np.mean([d3[p]['manifold_jaccard_mean'] for p in pairs])
    avg_sem_jaccard = np.mean([d3[p]['semantic_jaccard_mean'] for p in pairs])

    small, large = str(scales[0]), str(scales[-1])
    sem_small = d2[small]['semantic_novelty_mean']
    sem_large = d2[large]['semantic_novelty_mean']
    full_small = d2[small]['full_novelty_mean']
    full_large = d2[large]['full_novelty_mean']
    # Scale at which composite novelty first exceeds semantic novelty.
    crossover = next(
        (s for s in scales
         if d2[str(s)]['full_novelty_mean'] > d2[str(s)]['semantic_novelty_mean']),
        None,
    )
    crossover_txt = f"between {scales[scales.index(crossover)-1]:,} and {crossover:,}" \
        if crossover and scales.index(crossover) > 0 else "not observed"

    big = str(scales_s[-1])
    manifold_ms = d4[big]['retrieval_manifold_warm_cache_ms']['median_ms']
    sem_pre_ms = d4[big]['retrieval_semantic_prebuilt_matrix_ms']['median_ms']
    store_rate = d4[big]['store_throughput_per_sec_no_auto_state']
    autostate = d4['_meta']['auto_state_write_probe']['throughput_per_sec']
    bpm = d4[big]['bytes_per_memory']

    summary_text = (
        f"NCM MEASURED RESULTS\n"
        f"{'-'*46}\n\n"
        f"State-Conditioned Retrieval (exp3)\n"
        f"  Mean Jaccard distance:   {avg_jaccard:.3f}\n"
        f"  Semantic baseline:       {avg_sem_jaccard:.3f}\n"
        f"  (0 = identical sets, 1 = disjoint sets)\n\n"
        f"Novelty vs Store Size (exp2, AG News)\n"
        f"  Semantic:  {sem_small:.3f} @ {scales[0]:,} -> {sem_large:.1e} @ {scales[-1]:,}\n"
        f"  Composite: {full_small:.3f} @ {scales[0]:,} -> {full_large:.3f} @ {scales[-1]:,}\n"
        f"  Semantic falls below float32 eps (1.19e-07);\n"
        f"  composite stays non-zero. But composite exceeds\n"
        f"  semantic only from {crossover_txt} memories:\n"
        f"  below that, semantic is MORE novelty-sensitive.\n\n"
        f"Latency and Cost ({scales_s[-1]:,} memories, exp4)\n"
        f"  Composite, warm cache:   {manifold_ms:.2f} ms\n"
        f"  Semantic, prebuilt:      {sem_pre_ms:.2f} ms\n"
        f"  Writes, no auto-state:   {store_rate:,.0f}/sec\n"
        f"  Writes, auto-state on:   {autostate:,.0f}/sec\n"
        f"  Storage:                 {bpm:.0f} bytes/memory\n"
    )
    ax.text(0.03, 0.97, summary_text, transform=ax.transAxes,
            fontsize=8.5, verticalalignment='top', fontfamily='monospace',
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
