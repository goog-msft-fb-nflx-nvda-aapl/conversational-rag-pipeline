"""
RL agent that decides how many reranked passages (k=1,2,3) to include in the LLM prompt.

State:  reranker score features for top-10 passages
        (max, mean, std, score_gap_1_2, score_gap_2_3, score_gap_top_rest, query_len_norm)
Action: k ∈ {1, 2, 3} — number of passages sent to LLM
Reward: sentence similarity of generated answer vs gold answer

We use an offline approach: run inference once for each k value on training data subset,
cache rewards, then train a lightweight MLP policy with PPO (stable-baselines3).
"""

import json, os, gc, argparse, numpy as np
import torch
from tqdm import tqdm

# ─── parse args ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", default="./data")
parser.add_argument("--index_folder", default="./vector_database")
parser.add_argument("--retriever_model", default="./models/retriever")
parser.add_argument("--reranker_model", default="./models/reranker")
parser.add_argument("--output_dir", default="./models/rl_agent")
parser.add_argument("--cache_file", default="./rl_cache.json")
parser.add_argument("--n_train", type=int, default=300,
                    help="Number of training queries to cache rewards for")
parser.add_argument("--rl_timesteps", type=int, default=50000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 10
K_CHOICES = [1, 2, 3]

# ─── cache rewards for each (query_idx, k) if not already done ────────────────
if not os.path.exists(args.cache_file):
    print("Building reward cache...")

    import faiss, sqlite3, random
    from sentence_transformers import SentenceTransformer, CrossEncoder, util
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from utils import get_inference_system_prompt, get_inference_user_prompt, parse_generated_answer

    random.seed(args.seed)

    # Load retriever + index + DB
    retriever = SentenceTransformer(args.retriever_model, device=DEVICE)
    index = faiss.read_index(os.path.join(args.index_folder, "passage_index.faiss"))
    conn = sqlite3.connect(os.path.join(args.index_folder, "passage_store.db"))
    cur = conn.cursor()

    # Load reranker
    reranker = CrossEncoder(args.reranker_model, device=DEVICE)

    # Load LLM
    model_id = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    llm = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

    # Load sentence scorer
    sent_scorer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)

    # Sample training queries
    with open(os.path.join(args.data_folder, "train.txt")) as f:
        all_train = [json.loads(l) for l in f if l.strip()]
    with open(os.path.join(args.data_folder, "qrels.txt")) as f:
        qrels = json.load(f)

    sampled = random.sample(all_train, min(args.n_train, len(all_train)))

    cache = []
    for item in tqdm(sampled, desc="Caching rewards"):
        qid = item["qid"]
        query = item["rewrite"]
        gold_answer = item["answer"]["text"]
        if gold_answer == "CANNOTANSWER":
            continue  # skip unanswerable for RL training

        # Retrieve
        q_emb = retriever.encode(["query: " + query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = index.search(q_emb, TOP_K)
        rowids = I[0].tolist()
        if not rowids:
            continue

        placeholders = ",".join(["?"] * len(rowids))
        rows = cur.execute(f"SELECT rowid, pid, text FROM passages WHERE rowid IN ({placeholders})", tuple(rowids)).fetchall()
        rowid2pt = {rid: (pid, text) for (rid, pid, text) in rows}

        cand_ids, cand_texts = [], []
        for rid in rowids:
            tup = rowid2pt.get(int(rid))
            if tup:
                cand_ids.append(tup[0])
                cand_texts.append(tup[1])

        if not cand_texts:
            continue

        # Rerank
        pairs = [(query, t) for t in cand_texts]
        scores = reranker.predict(pairs)
        reranked = sorted(zip(scores, cand_texts), key=lambda x: x[0], reverse=True)
        reranked_scores = [float(s) for s, _ in reranked]
        reranked_texts = [t for _, t in reranked]

        # State features
        arr = np.array(reranked_scores[:TOP_K])
        pad = np.zeros(TOP_K)
        pad[:len(arr)] = arr
        query_len = min(len(query.split()) / 50.0, 1.0)
        state_feats = list(pad) + [
            float(np.max(arr)),
            float(np.mean(arr)),
            float(np.std(arr) if len(arr) > 1 else 0),
            float(arr[0] - arr[1]) if len(arr) > 1 else 0,
            float(arr[1] - arr[2]) if len(arr) > 2 else 0,
            query_len
        ]

        # Get rewards for each k
        rewards = {}
        for k in K_CHOICES:
            ctx = reranked_texts[:k]
            messages = [
                {"role": "system", "content": get_inference_system_prompt()},
                {"role": "user", "content": get_inference_user_prompt(query, ctx)}
            ]
            rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                     tokenize=False, enable_thinking=False)
            inputs = tokenizer(rendered, return_tensors="pt").to(llm.device)
            with torch.no_grad():
                out = llm.generate(**inputs, max_new_tokens=256)
            decoded = tokenizer.decode(out[0], skip_special_tokens=True)
            pred = parse_generated_answer(decoded.strip())

            # Sentence sim reward
            e1 = sent_scorer.encode([pred], normalize_embeddings=True)
            e2 = sent_scorer.encode([gold_answer], normalize_embeddings=True)
            sim = float(util.cos_sim(torch.tensor(e1), torch.tensor(e2))[0][0])
            rewards[k] = sim

        cache.append({
            "qid": qid,
            "state_features": state_feats,
            "rewards": rewards,
        })

    with open(args.cache_file, "w") as f:
        json.dump(cache, f)
    print(f"Cached {len(cache)} entries to {args.cache_file}")

    del llm, retriever, reranker, sent_scorer
    gc.collect()
    torch.cuda.empty_cache()

else:
    print(f"Loading reward cache from {args.cache_file}")
    with open(args.cache_file) as f:
        cache = json.load(f)
    print(f"Loaded {len(cache)} cached entries")


# ─── gymnasium environment ─────────────────────────────────────────────────────
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

N_FEATURES = 10 + 6  # TOP_K scores + 6 derived features

class PassageSelectionEnv(gym.Env):
    """
    Episode: one query at a time (sampled from cache).
    State: reranker score features (16-dim)
    Action: k ∈ {0,1,2} → k+1 passages
    Reward: sentence similarity for chosen k
    """
    metadata = {"render_modes": []}

    def __init__(self, cache_data, seed=42):
        super().__init__()
        self.cache = cache_data
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(
            low=-20.0, high=20.0, shape=(N_FEATURES,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(K_CHOICES))
        self._current_entry = None

    def reset(self, seed=None, options=None):
        idx = self.rng.integers(0, len(self.cache))
        self._current_entry = self.cache[idx]
        obs = np.array(self._current_entry["state_features"][:N_FEATURES], dtype=np.float32)
        return obs, {}

    def step(self, action):
        k = K_CHOICES[int(action)]
        reward = self._current_entry["rewards"].get(k, 0.0)
        # Convert string keys (JSON) to int
        if str(k) in self._current_entry["rewards"]:
            reward = self._current_entry["rewards"][str(k)]
        obs, _ = self.reset()
        return obs, float(reward), False, True, {}  # truncated after 1 step


env = PassageSelectionEnv(cache)
check_env(env, warn=True)

print(f"Training PPO agent for {args.rl_timesteps} timesteps...")
model_ppo = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=args.seed,
    learning_rate=3e-4,
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    policy_kwargs={"net_arch": [64, 64]},
)
model_ppo.learn(total_timesteps=args.rl_timesteps)
model_ppo.save(os.path.join(args.output_dir, "ppo_passage_selector"))
print(f"Saved RL agent to {args.output_dir}")

# Evaluate on cache
correct = 0
total = 0
k_choices_count = {k: 0 for k in K_CHOICES}
for entry in cache:
    obs = np.array(entry["state_features"][:N_FEATURES], dtype=np.float32)
    action, _ = model_ppo.predict(obs, deterministic=True)
    k = K_CHOICES[int(action)]
    k_choices_count[k] += 1
    # Check if RL choice ≥ mean random choice
    rewards = {int(key): val for key, val in entry["rewards"].items()}
    rl_reward = rewards.get(k, 0.0)
    best_k = max(rewards, key=rewards.get)
    if k == best_k:
        correct += 1
    total += 1

print(f"RL optimal action rate: {correct}/{total} = {correct/max(total,1):.2%}")
print(f"K choices distribution: {k_choices_count}")

# Save analysis
analysis = {
    "optimal_action_rate": correct / max(total, 1),
    "k_distribution": k_choices_count,
    "mean_reward_per_k": {
        k: float(np.mean([
            entry["rewards"].get(str(k), entry["rewards"].get(k, 0.0))
            for entry in cache
        ]))
        for k in K_CHOICES
    }
}
with open(os.path.join(args.output_dir, "rl_analysis.json"), "w") as f:
    json.dump(analysis, f, indent=2)
print(f"Analysis: {analysis}")
