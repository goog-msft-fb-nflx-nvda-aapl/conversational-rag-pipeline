# Conversational RAG with Dense Retrieval and Passage Reranking

A complete **Retrieval-Augmented Generation (RAG)** pipeline for open-domain conversational question answering, featuring a fine-tuned bi-encoder retriever, cross-encoder reranker, a Qwen3 generator, and a lightweight RL agent that dynamically selects how many retrieved passages to include.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [Approaches](#approaches)
  - [Retriever Fine-tuning](#1-retriever-fine-tuning)
  - [Reranker Analysis](#2-reranker-analysis)
  - [Prompt Optimization](#3-prompt-optimization)
  - [RL for Passage Count Selection](#4-rl-for-passage-count-selection)
- [Results](#results)
- [Key Findings](#key-findings)
- [Reproduction Guide](#reproduction-guide)
- [Project Structure](#project-structure)

---

## Problem Statement

Given a conversational query (natural language question with prior dialogue context), the system must:

1. **Retrieve** relevant passages from a 100K-passage corpus using dense vector search (FAISS)
2. **Rerank** the top-10 retrieved passages using a cross-encoder
3. **Generate** a concise answer using an LLM conditioned on the top passages
4. Handle **unanswerable** questions by outputting `CANNOTANSWER`

**Evaluation metrics:**
- `Recall@10` — whether the gold passage appears in the top-10 retrieved results
- `MRR@10` — mean reciprocal rank of the gold passage after reranking
- `Sentence Similarity` — cosine similarity between generated answer and gold answer (via `all-MiniLM-L6-v2`)

**Public baselines to beat:** Recall@10 ≥ 0.780 · MRR@10 ≥ 0.695 · SentSim ≥ 0.340

---

## Dataset

The dataset is a conversational QA benchmark (CoQA-style with Wikipedia passages).

| File | Description |
|------|-------------|
| `corpus.txt` | 100,000 passages — each line: `{"text":..., "title":..., "id":"articleId@paraIdx"}` |
| `train.txt` | 31,526 training queries with BM25-retrieved evidence passages and retrieval labels |
| `test_open.txt` | 3,343 test queries (gold passage may not be in provided evidences) |
| `qrels.txt` | 40,527 query→positive-passage mappings |

Each training example has:
- `rewrite` — query text (conversation-aware rewrite)
- `evidences` — 5 BM25-retrieved passages (hard negatives)
- `retrieval_labels` — binary label per evidence (1=positive, 0=negative)
- `answer` — gold answer span or `"CANNOTANSWER"`

---

## System Architecture

```
                          Query
                            │
                  ┌─────────▼─────────┐
                  │  Bi-Encoder        │  intfloat/multilingual-e5-small
                  │  (Retriever)       │  fine-tuned with MNRL
                  └─────────┬─────────┘
                            │  query embedding
                  ┌─────────▼─────────┐
                  │  FAISS Index       │  IndexFlatIP, 100K passages
                  │  (Top-K=10)        │
                  └─────────┬─────────┘
                            │  candidate passages
                  ┌─────────▼─────────┐
                  │  Cross-Encoder     │  ms-marco-MiniLM-L-12-v2
                  │  (Reranker)        │  base model (see Findings)
                  └─────────┬─────────┘
                            │  reranked scores
          ┌─────────────────▼──────────────────┐
          │  RL Agent (PPO)                     │
          │  State: reranker score features     │
          │  Action: k ∈ {1, 2, 3} passages    │
          └─────────────────┬──────────────────┘
                            │  top-k passages
                  ┌─────────▼─────────┐
                  │  Qwen3-1.7B (bf16) │  generate answer
                  │  (Generator LLM)   │  enable_thinking=False
                  └─────────┬─────────┘
                            │
                        Answer
```

---

## Approaches

### 1. Retriever Fine-tuning

**Model:** `intfloat/multilingual-e5-small` (384-dim, ~120MB)

**Training data construction:**
- **Anchor:** `rewrite` field from `train.txt` (31,526 queries), prefixed with `"query: "`
- **Positive:** Gold passage from `qrels.txt`, prefixed with `"passage: "`
- **Negatives:** In-batch negatives (other positives in same batch) — with batch size 512, each anchor competes against 511 negatives per step

**Loss function — MultipleNegativesRankingLoss (MNRL):**

```
L = -log( exp(sim(q_i, p_i) / τ) / Σ_j exp(sim(q_i, p_j) / τ) )
```

where `τ = 20`, `sim` is cosine similarity. Larger batches = more negatives = stronger gradient signal.

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Base model | intfloat/multilingual-e5-small |
| Epochs | 3 |
| Batch size | 512 (single GPU — multi-GPU accidentally multiplied effective batch ×8) |
| Learning rate | 2e-5 |
| Max sequence length | 256 |
| Warmup ratio | 10% |
| Optimizer | AdamW |

**Dev set results (300-query sample from train):**

| Checkpoint | NDCG@10 |
|-----------|---------|
| Epoch 0.25 | 0.843 |
| Epoch 0.50 | 0.856 |
| Epoch 0.75 | 0.872 |
| Epoch 1.00 | 0.880 |
| Epoch 1.25 | 0.887 |
| Epoch 1.50 | 0.890 |
| Epoch 1.75 | 0.892 |
| Epoch 2.00 | 0.895 |
| Epoch 2.25 | 0.901 |
| Epoch 2.50 | 0.902 |
| Epoch 2.75 | 0.902 |
| Epoch 3.00 | **0.902** |

---

### 2. Reranker Analysis

**Model:** `cross-encoder/ms-marco-MiniLM-L-12-v2`

**Training data:**
- Positives: gold passage per query (1 per query, 31,526 total)
- Negatives: BM25 evidence passages with label=0 (up to 5 per query, ~125K total)

**Loss function — BCEWithLogitsLoss:**
```
L = -[y·log(σ(s)) + (1−y)·log(1−σ(s))]
```

**Key finding — Overfitting:** Despite achieving 99.7% dev MRR on training-distribution examples, the fine-tuned reranker *degraded* test performance:

| Config | MRR@10 (test) |
|--------|--------------|
| Base reranker (no fine-tuning) | **0.7003** |
| Fine-tuned reranker | 0.4372 |

**Root cause:** The dev evaluation used a sample from the *training distribution*. The reranker memorized the specific BM25 negatives in `train.txt` but failed to generalize to harder test negatives. **Decision: submit base reranker.**

---

### 3. Prompt Optimization

The generator is `Qwen/Qwen3-1.7B` (bf16, non-thinking mode). Three prompt strategies were tested:

**Critical discovery:** Qwen3 generates `<think>...</think>` tokens even with `enable_thinking=False` when no system prompt is provided. These tokens must be stripped:

```python
pred_ans = re.sub(r"<think>.*?</think>", "", pred_ans, flags=re.DOTALL).strip()
```

Without this fix: SentSim ≈ 0.099 (think-block parsed as the answer).  
With fix: SentSim ≈ 0.358.

**Prompt variants tested:**

<details>
<summary><b>v1: Minimal (no system prompt)</b></summary>

```
[1] {passage_1}
[2] {passage_2}
[3] {passage_3}

Question: {query}
Short answer (1 sentence max, or CANNOTANSWER):
```
</details>

<details>
<summary><b>v2: Instructed (best)</b></summary>

System: *"You are a precise question-answering assistant. Answer the question using only the provided context passages. If the answer is not in the context, respond with 'CANNOTANSWER'. Give a concise, direct answer."*

```
Context:
[Passage 1]
{passage_1}
[Passage 2]
{passage_2}
[Passage 3]
{passage_3}

Question: {query}
Answer (if not in context, say CANNOTANSWER):
```
</details>

<details>
<summary><b>v3: Extractive</b></summary>

System: *"You are a reading comprehension assistant. Extract the shortest exact answer span from the context. Reply with ONLY the answer. If the context does not contain the answer, say 'CANNOTANSWER'."*

```
Passages:
(1) {passage_1}
(2) {passage_2}
(3) {passage_3}

Q: {query}
A (extracted span or CANNOTANSWER):
```
</details>

**Results (FT retriever + base reranker):**

| Prompt | SentSim |
|--------|---------|
| v1: Minimal | 0.3585 |
| **v2: Instructed** | **0.3592** |
| v3: Extractive | 0.3560 |

All three variants beat the 0.340 baseline after the think-token fix.

---

### 4. RL for Passage Count Selection

**Motivation:** The standard pipeline sends a fixed `TOP_M=3` passages to the LLM. An RL agent can learn to select `k ∈ {1, 2, 3}` based on retrieval confidence.

**Reference:** [FLARE — Active Retrieval Augmented Generation](https://arxiv.org/abs/2406.06475)

**Environment (Gymnasium):**

| Component | Details |
|-----------|---------|
| State | 16-dim vector: top-10 reranker scores + max/mean/std/gap₁₂/gap₂₃/query_len |
| Action | Discrete(3): k ∈ {1, 2, 3} passages |
| Reward | Sentence similarity of generated answer vs. gold |

**Offline RL approach:**  
Run LLM inference for k=1,2,3 on 400 training queries once, cache rewards, then train PPO entirely on the cache — no online LLM calls during policy training.

**PPO (stable-baselines3):**

| Parameter | Value |
|-----------|-------|
| Policy | MlpPolicy [64, 64] |
| Learning rate | 3e-4 |
| Timesteps | 50,000 |
| n_steps | 256, batch_size 64 |

**Results:**

| k | Mean Reward |
|---|------------|
| 1 | 0.155 |
| 2 | 0.200 |
| **3** | **0.212** |
| RL policy | adaptive |

Agent distribution: k=1 (6.4%), k=2 (25.7%), k=3 (67.9%). The agent learns that k=3 dominates in most cases, but selects k=1 when reranker confidence is very high (clear top passage).

---

## Results

### Ablation Study

| Configuration | Recall@10 | MRR@10 | SentSim |
|---------------|-----------|--------|---------|
| **Public baseline** | 0.780 | 0.695 | 0.340 |
| Base retriever + Base reranker | 0.7151 | 0.6335 | 0.3518 |
| **FT retriever** + Base reranker | **0.8181** | **0.7003** | **0.3592** |
| FT retriever + FT reranker | 0.8181 | 0.4372 | 0.2872 |

### Prompt Comparison (FT retriever + Base reranker)

| Prompt | Recall@10 | MRR@10 | SentSim |
|--------|-----------|--------|---------|
| v1: Minimal | 0.8181 | 0.7003 | 0.3585 |
| **v2: Instructed** ← submitted | 0.8181 | 0.7003 | **0.3592** |
| v3: Extractive | 0.8181 | 0.7003 | 0.3560 |

### vs. Public Baselines

| Metric | Baseline | Ours | Δ |
|--------|----------|------|---|
| Recall@10 | 0.780 | **0.818** | +3.8pp ✓ |
| MRR@10 | 0.695 | **0.700** | +0.5pp ✓ |
| SentSim | 0.340 | **0.359** | +1.9pp ✓ |

---

## Key Findings

1. **Retriever fine-tuning dominates** — The biggest gains come from fine-tuning the bi-encoder: +6.3pp Recall@10 and +0.7pp MRR vs. the base model.

2. **Reranker fine-tuning can hurt** — Evaluating the reranker on training-distribution data (99.7% dev MRR) masked severe overfitting; test MRR dropped from 0.70 → 0.44. Always use a held-out evaluation set with diverse hard negatives.

3. **Think-token stripping is essential for Qwen3** — Qwen3-1.7B generates `<think>` blocks even with `enable_thinking=False` when given a minimal/empty system prompt. The parser must strip these: without stripping, SentSim ≈ 0.10.

4. **CANNOTANSWER guidance is critical** — Explicit `CANNOTANSWER` instruction prevents hallucination on unanswerable questions. All three prompt variants with this guidance achieved SentSim > 0.340.

5. **RL agent confirms k=3 is near-optimal** — The learned policy mostly chooses k=3 (67.9%) but can reduce to k=1 for high-confidence retrievals, confirming that more context generally helps this dataset.

---

## Reproduction Guide

### 1. Environment Setup

```bash
conda create -n rag_env python=3.12 -y
conda activate rag_env

pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.56.1 datasets==4.0.0 tqdm==4.67.1 \
    faiss-gpu-cu12==1.12.0 sentence-transformers==5.1.0 \
    python-dotenv==1.1.1 accelerate==1.10.1 gdown \
    stable-baselines3 gymnasium matplotlib seaborn
```

Create `.env` in the project root:
```
hf_token=<your_huggingface_read_token>
```

### 2. Download Data

```bash
python -m pip install gdown
gdown --folder https://drive.google.com/drive/folders/1v5hSQYPyQuUnzaE1Lp3F1vejNazW48TH -O ./
# Places data/ and provided scripts in current directory
```

### 3. Download Pre-trained Models

```bash
bash download.sh
# Downloads models/retriever/ (449MB) and models/reranker/ (128MB)
```

### 4. Fine-tune Retriever (~1 min on H200, ~15 min on RTX 3070)

```bash
python code/train_retriever.py \
    --data_folder ./data \
    --output_dir ./models/retriever \
    --base_model intfloat/multilingual-e5-small \
    --epochs 3 \
    --batch_size 512 \
    --lr 2e-5
```

### 5. Fine-tune Reranker (~45 min on H200)

```bash
python code/train_reranker.py \
    --data_folder ./data \
    --output_dir ./models/reranker \
    --base_model cross-encoder/ms-marco-MiniLM-L-12-v2 \
    --epochs 3 \
    --batch_size 64 \
    --lr 2e-5
```

> **Note:** The fine-tuned reranker overfits on this dataset. The submitted checkpoint uses the **base** `cross-encoder/ms-marco-MiniLM-L-12-v2` (downloaded via `download.sh`).

### 6. Build FAISS Index (~1 min)

```bash
python save_embeddings.py \
    --data_folder ./data \
    --retriever_model_path ./models/retriever \
    --output_folder ./vector_database \
    --batch_size 512 \
    --build_db
```

### 7. Run Inference

```bash
cp code/utils_v2_instructed.py utils.py   # select best prompt

python inference_batch.py \
    --retriever_model_path ./models/retriever \
    --reranker_model_path ./models/reranker \
    --test_data_path ./data/test_open.txt \
    --result_file_name result_v2_instructed.json
```

### 8. Train RL Agent (optional, ~30 min for cache + 5 min PPO)

```bash
python code/train_rl.py \
    --retriever_model ./models/retriever \
    --reranker_model ./models/reranker \
    --n_train 400 \
    --rl_timesteps 50000
```

### 9. Reproduce All Prompt Experiments

```bash
for variant in v1_minimal v2_instructed v3_extractive; do
    cp code/utils_${variant}.py utils.py
    python inference_batch.py \
        --retriever_model_path ./models/retriever \
        --reranker_model_path ./models/reranker \
        --test_data_path ./data/test_open.txt \
        --result_file_name result_${variant}.json
done
```

---

## Project Structure

```
.
├── download.sh                  # Download pre-trained model checkpoints
├── utils.py                     # Active prompt config (copy from code/utils_v2_instructed.py)
├── save_embeddings.py           # Build FAISS index + SQLite passage DB
├── inference_batch.py           # Full RAG pipeline evaluation
├── .env                         # HuggingFace token (hf_token=...)
│
├── code/
│   ├── train_retriever.py       # Fine-tune bi-encoder retriever
│   ├── train_reranker.py        # Fine-tune cross-encoder reranker
│   ├── train_rl.py              # RL agent for passage count selection
│   ├── plot_results.py          # Generate all report figures
│   ├── utils_v1_minimal.py      # Prompt variant 1: minimal
│   ├── utils_v2_instructed.py   # Prompt variant 2: instructed (best)
│   └── utils_v3_extractive.py   # Prompt variant 3: extractive
│
├── models/                      # Created by download.sh
│   ├── retriever/               # Fine-tuned multilingual-e5-small
│   ├── reranker/                # Base ms-marco-MiniLM-L-12-v2
│   └── rl_agent/                # PPO passage-count selector
│
├── data/                        # Dataset (download separately)
│   ├── corpus.txt
│   ├── train.txt
│   ├── test_open.txt
│   └── qrels.txt
│
├── vector_database/             # FAISS index + SQLite DB (built by save_embeddings.py)
├── results/                     # Inference output JSONs
└── plots/                       # Training curves and analysis figures
```

---

## Model Weights

Pre-trained checkpoints hosted on HuggingFace Hub:

| Component | Repository | Size |
|-----------|-----------|------|
| Fine-tuned Retriever | [jameswatanabegoogle2024/adl-hw3-retriever](https://huggingface.co/jameswatanabegoogle2024/adl-hw3-retriever) | 449 MB |
| Base Reranker | [jameswatanabegoogle2024/adl-hw3-reranker](https://huggingface.co/jameswatanabegoogle2024/adl-hw3-reranker) | 128 MB |

---

## References

- [FLARE: Active Retrieval Augmented Generation](https://arxiv.org/abs/2406.06475)
- [Faiss: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://sbert.net)
- [Mean Reciprocal Rank](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io)
- [Gymnasium](https://gymnasium.farama.org)
