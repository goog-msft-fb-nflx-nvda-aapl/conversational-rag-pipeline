"""
Fine-tune intfloat/multilingual-e5-small as bi-encoder retriever.
Uses MultipleNegativesRankingLoss with large batches (= more in-batch negatives).
"""

import json, os, random, logging, argparse
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_folder", default="./data")
    p.add_argument("--output_dir", default="./models/retriever")
    p.add_argument("--base_model", default="intfloat/multilingual-e5-small")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_seq_length", type=int, default=256)
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

            gold_pids = {pid for pid, lbl in qrels.get(qid, {}).items() if lbl != 0}
            if not gold_pids:
                continue

            for pid in gold_pids:
                if pid in pid2text:
                    examples.append(InputExample(texts=[
                        "query: " + query,
                        "passage: " + pid2text[pid]
                    ]))
                    break

    logger.info(f"Built {len(examples)} training examples")
    return examples


def build_dev_evaluator(data_folder, pid2text, qrels, max_queries=300):
    random.seed(0)
    with open(os.path.join(data_folder, "train.txt")) as f:
        lines = [l.strip() for l in f if l.strip()]

    queries, corpus_sub, relevant = {}, {}, {}
    sampled = random.sample(lines, min(max_queries * 5, len(lines)))
    count = 0
    for line in sampled:
        if count >= max_queries:
            break
        obj = json.loads(line)
        qid = obj["qid"]
        gold_pids = {pid for pid, lbl in qrels.get(qid, {}).items() if lbl != 0}
        has_text = [pid for pid in gold_pids if pid in pid2text]
        if not has_text:
            continue
        queries[qid] = "query: " + obj["rewrite"]
        relevant[qid] = set(has_text)
        for pid in has_text:
            corpus_sub[pid] = "passage: " + pid2text[pid]
        count += 1

    neg_pids = random.sample(list(pid2text.keys()), min(5000, len(pid2text)))
    for pid in neg_pids:
        corpus_sub[pid] = "passage: " + pid2text[pid]

    logger.info(f"Dev evaluator: {len(queries)} queries, {len(corpus_sub)} passages")
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus_sub,
        relevant_docs=relevant,
        name="dev",
        mrr_at_k=[10],
        ndcg_at_k=[10],
        precision_recall_at_k=[10],
        show_progress_bar=False,
    )


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    pid2text = load_corpus(args.data_folder)
    qrels = load_qrels(args.data_folder)
    train_examples = build_training_examples(args.data_folder, pid2text, qrels, args.seed)

    logger.info(f"Loading base model: {args.base_model}")
    model = SentenceTransformer(args.base_model)
    model.max_seq_length = args.max_seq_length

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model=model)

    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    logger.info(f"Steps/epoch: {len(train_dataloader)}, total: {total_steps}, warmup: {warmup_steps}")

    dev_evaluator = build_dev_evaluator(args.data_folder, pid2text, qrels)

    score_log = []
    def callback(score, epoch, steps):
        score_log.append({"epoch": epoch, "steps": steps, "score": score})
        logger.info(f"[Epoch {epoch} Step {steps}] Score: {score:.4f}")

    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=args.output_dir,
        save_best_model=True,
        evaluator=dev_evaluator,
        evaluation_steps=len(train_dataloader) // 4,
        callback=callback,
        show_progress_bar=True,
        checkpoint_path=os.path.join(args.output_dir, "checkpoints"),
        checkpoint_save_steps=len(train_dataloader),
        checkpoint_save_total_limit=2,
    )

    logger.info(f"Saved model to {args.output_dir}")
    with open(os.path.join(args.output_dir, "training_scores.json"), "w") as f:
        json.dump(score_log, f, indent=2)
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
