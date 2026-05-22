"""
Generate all plots for the report after inference results are available.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "./results"
PLOTS_DIR = "./plots"
MODELS_DIR = "./models"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── 1. Retriever training curve ──────────────────────────────────────────────
def plot_curve(score_file, title, ylabel, fname, color="steelblue"):
    if not os.path.exists(score_file):
        print(f"[skip] {score_file}")
        return
    with open(score_file) as f:
        data = json.load(f)
    if not data:
        return
    steps = list(range(len(data)))
    scores = [d["score"] for d in data]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, scores, marker="o", color=color, lw=2, ms=6)
    for i, (s, v) in enumerate(zip(steps, scores)):
        ax.annotate(f"{v:.3f}", (s, v), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("Evaluation Checkpoint")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(steps)
    ax.set_xticklabels([f"E{d['epoch']:.1f}" for d in data], rotation=45, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close()
    print(f"Saved {fname}")

plot_curve(f"{MODELS_DIR}/retriever/training_scores.json",
           "Retriever Training: Dev NDCG@10", "NDCG@10",
           "retriever_training_curve.png", "steelblue")

plot_curve(f"{MODELS_DIR}/reranker/training_scores.json",
           "Reranker Training: Dev MRR@10", "MRR@10",
           "reranker_training_curve.png", "darkorange")

# ─── 2. Ablation comparison ───────────────────────────────────────────────────
ablation_files = {
    "Base Ret\n+ Base Rerank": "result_base_base.json",
    "FT Ret\n+ Base Rerank": "result_ft_ret_base_rerank.json",
    "FT Ret\n+ FT Rerank\n(Main)": "result_v2_instructed.json",
}
ablation_results = {}
for label, fname in ablation_files.items():
    path = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        ablation_results[label] = {
            "Recall@10": d.get("recall@10", 0),
            "MRR@10": d.get("mrr@10", 0),
            "Sentence Sim": d.get("Bi-Encoder_CosSim", 0),
        }

# ─── 3. Prompt comparison ─────────────────────────────────────────────────────
prompt_files = {
    "v1: Minimal": "result_v1_minimal.json",
    "v2: Instructed\n(Best)": "result_v2_instructed.json",
    "v3: Extractive": "result_v3_extractive.json",
}
prompt_results = {}
for label, fname in prompt_files.items():
    path = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        prompt_results[label] = {
            "Recall@10": d.get("recall@10", 0),
            "MRR@10": d.get("mrr@10", 0),
            "Sentence Sim": d.get("Bi-Encoder_CosSim", 0),
        }

def plot_comparison(results_dict, title, fname):
    if not results_dict:
        print(f"[skip] no data for {fname}")
        return
    labels = list(results_dict.keys())
    metrics = ["Recall@10", "MRR@10", "Sentence Sim"]
    x = np.arange(len(labels))
    width = 0.25
    colors = ["steelblue", "darkorange", "seagreen"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [results_dict[l].get(metric, 0) for l in labels]
        bars = ax.bar(x + i*width, vals, width, label=metric, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150)
    plt.close()
    print(f"Saved {fname}")

plot_comparison(ablation_results, "Ablation Study: Retriever & Reranker", "ablation_comparison.png")
plot_comparison(prompt_results, "Prompt Design Comparison", "prompt_comparison.png")

# ─── 4. Retrieval quality analysis ────────────────────────────────────────────
main_result = os.path.join(RESULTS_DIR, "result_v2_instructed.json")
if os.path.exists(main_result):
    with open(main_result) as f:
        data = json.load(f)
    records = data.get("records", [])

    gold_in_top3 = sum(1 for r in records
                       if any(p in set(r.get("gold_pids", [])) for p in [x["pid"] for x in r.get("retrieved", [])[:3]]))
    gold_in_top10 = sum(1 for r in records
                        if any(p in set(r.get("gold_pids", [])) for p in [x["pid"] for x in r.get("retrieved", [])[:10]]))
    total = len(records)

    categories = ["Gold in Top-3\n(→ LLM)", "Gold in Top-4~10\n(retrieved, not sent)", "Gold NOT\nretrieved"]
    values = [gold_in_top3, gold_in_top10 - gold_in_top3, total - gold_in_top10]
    colors_pie = ["seagreen", "gold", "tomato"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    wedges, texts, autotexts = ax1.pie(values, labels=categories, colors=colors_pie,
                                        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
    ax1.set_title("Retrieval Coverage Analysis\n(n={})".format(total))

    # Bar chart of retrieval ranks
    if records:
        rank_counts = [0] * 11  # rank 0 = not found, rank 1-10 = found at rank k
        for r in records:
            gold_pids = set(r.get("gold_pids", []))
            ret_pids = [x["pid"] for x in r.get("retrieved", [])]
            found = False
            for rank, pid in enumerate(ret_pids, 1):
                if pid in gold_pids:
                    rank_counts[rank] += 1
                    found = True
                    break
            if not found:
                rank_counts[0] += 1

        ranks = list(range(1, 11))
        ax2.bar(ranks, [rank_counts[r] for r in ranks], color="steelblue", alpha=0.8)
        ax2.set_xlabel("Rank at which gold passage was found")
        ax2.set_ylabel("Count")
        ax2.set_title("Gold Passage Rank Distribution (Top-10)")
        ax2.set_xticks(ranks)
        ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "retrieval_analysis.png"), dpi=150)
    plt.close()
    print("Saved retrieval_analysis.png")

# ─── 5. RL analysis ───────────────────────────────────────────────────────────
rl_analysis_path = "./models/rl_agent/rl_analysis.json"
if os.path.exists(rl_analysis_path):
    with open(rl_analysis_path) as f:
        rl = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # K choice distribution
    k_dist = rl.get("k_distribution", {})
    k_vals = [str(k) for k in sorted(k_dist.keys(), key=int)]
    k_counts = [k_dist[k] for k in k_vals]
    ax1.bar(k_vals, k_counts, color=["steelblue", "darkorange", "seagreen"], alpha=0.85)
    ax1.set_xlabel("Number of passages (k)")
    ax1.set_ylabel("Count")
    ax1.set_title("RL Agent: Passage Count Distribution")
    ax1.grid(axis="y", alpha=0.3)

    # Mean reward per k
    mean_rewards = rl.get("mean_reward_per_k", {})
    k_vals2 = [str(k) for k in sorted(mean_rewards.keys(), key=int)]
    rewards = [mean_rewards[k] for k in k_vals2]
    ax2.bar(k_vals2, rewards, color=["steelblue", "darkorange", "seagreen"], alpha=0.85)
    ax2.set_xlabel("Number of passages (k)")
    ax2.set_ylabel("Mean Sentence Similarity")
    ax2.set_title("Mean Reward by Passage Count")
    ax2.grid(axis="y", alpha=0.3)
    for bar, v in zip(ax2.patches, rewards):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    plt.suptitle(f"RL Agent Analysis (optimal rate: {rl.get('optimal_action_rate', 0):.1%})")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "rl_analysis.png"), dpi=150)
    plt.close()
    print("Saved rl_analysis.png")

# ─── 6. Print summary ─────────────────────────────────────────────────────────
print("\n=== RESULTS SUMMARY ===")
all_results = {**{f"ablation/{k}": v for k, v in ablation_results.items()},
               **{f"prompt/{k}": v for k, v in prompt_results.items()}}
for name, res in all_results.items():
    print(f"{name}: Recall@10={res['Recall@10']:.4f} MRR@10={res['MRR@10']:.4f} SentSim={res['Sentence Sim']:.4f}")
