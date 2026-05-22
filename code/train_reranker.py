"""
Fine-tune cross-encoder/ms-marco-MiniLM-L-12-v2 as reranker.
Uses sentence-transformers 5.x CrossEncoder.fit() API.
"""

import json, os, random, logging, argparse
import torch
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_folder", default="./data")
    p.add_argument("--output_dir", default="./models/reranker")
    p.add_argument("--base_model", default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_corpus(data_folder):
    pid2text = {}
    with open(os.path.join(data_folder, "corpus.txt")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid2text[obj["id"]] = obj["text"]
    logger.info(f"Loaded {len(pid2text)} corpus passages")
    return pid2text


def load_qrels(data_folder):
    with open(os.path.join(data_folder, "qrels.txt")) as f:
        return json.load(f)


def build_training_examples(data_folder, pid2text, qrels, seed=42):
    """(query, passage, label) pairs; label=1.0 positive, 0.0 negative."""
    random.seed(seed)
    examples = []
    with open(os.path.join(data_folder, "train.txt")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj["qid"]
            query = obj["rewrite"]
            evidences = obj["evidences"]
            labels = obj.get("retrieval_labels", [])
            gold_pids = {pid for pid, lbl in qrels.get(qid, {}).items() if lbl != 0}

            # One positive from corpus
            for pid in gold_pids:
                if pid in pid2text:
                    examples.append(InputExample(texts=[query, pid2text[pid]], label=1.0))
                    break

            # Negatives from BM25 evidences
            for ev_text, lbl in zip(evidences, labels):
                if lbl == 0:
                    examples.append(InputExample(texts=[query, ev_text], label=0.0))

    logger.info(f"Built {len(examples)} reranker training examples")
    return examples


def build_dev_evaluator(data_folder, pid2text, qrels, max_queries=200):
    """CERerankingEvaluator expects list of dicts with positive/negative as lists."""
    random.seed(0)
    with open(os.path.join(data_folder, "train.txt")) as f:
        lines = [l.strip() for l in f if l.strip()]

    samples = []
    sampled = random.sample(lines, min(max_queries * 5, len(lines)))
    for line in sampled:
        if len(samples) >= max_queries:
            break
        obj = json.loads(line)
        qid = obj["qid"]
        query = obj["rewrite"]
        evidences = obj["evidences"]
        labels = obj.get("retrieval_labels", [])
        gold_pids = {pid for pid, lbl in qrels.get(qid, {}).items() if lbl != 0}

        pos_texts = [pid2text[pid] for pid in gold_pids if pid in pid2text][:1]
        neg_texts = [ev for ev, lbl in zip(evidences, labels) if lbl == 0]

        if pos_texts and neg_texts:
            samples.append({
                "query": query,
                "positive": pos_texts,
                "negative": neg_texts,
            })

    logger.info(f"Dev evaluator: {len(samples)} queries")
    return CERerankingEvaluator(samples=samples, name="dev")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    pid2text = load_corpus(args.data_folder)
    qrels = load_qrels(args.data_folder)
    train_examples = build_training_examples(args.data_folder, pid2text, qrels, args.seed)

    logger.info(f"Loading base model: {args.base_model}")
    model = CrossEncoder(args.base_model, num_labels=1, max_length=args.max_length)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size, drop_last=False)
    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    logger.info(f"Steps/epoch: {len(train_dataloader)}, total: {total_steps}, warmup: {warmup_steps}")

    dev_evaluator = build_dev_evaluator(args.data_folder, pid2text, qrels)

    score_log = []
    def callback(score, epoch, steps):
        score_log.append({"epoch": epoch, "steps": steps, "score": score})
        logger.info(f"[Epoch {epoch} Step {steps}] MRR: {score:.4f}")

    model.fit(
        train_dataloader=train_dataloader,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=args.output_dir,
        save_best_model=True,
        evaluator=dev_evaluator,
        evaluation_steps=len(train_dataloader) // 4,
        callback=callback,
        show_progress_bar=True,
    )

    logger.info(f"Saved model to {args.output_dir}")
    with open(os.path.join(args.output_dir, "training_scores.json"), "w") as f:
        json.dump(score_log, f, indent=2)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
