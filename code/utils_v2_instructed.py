from typing import List
import re


def get_inference_system_prompt() -> str:
    return (
        "You are a precise question-answering assistant. "
        "Answer the question using only the provided context passages. "
        "If the answer is not in the context, respond with 'CANNOTANSWER'. "
        "Give a concise, direct answer — avoid extra explanation."
    )


def get_inference_user_prompt(query: str, context_list: List[str]) -> str:
    ctx_block = "\n\n".join(
        f"[Passage {i+1}]\n{text}" for i, text in enumerate(context_list)
    )
    return (
        f"Context:\n{ctx_block}\n\n"
        f"Question: {query}\n\n"
        f"Answer (if not in context, say CANNOTANSWER):"
    )


def parse_generated_answer(pred_ans: str) -> str:
    if "assistant" in pred_ans.lower():
        parts = re.split(r"(?i)assistant\s*\n?", pred_ans)
        pred_ans = parts[-1].strip()
    # Strip Qwen3 thinking blocks
    pred_ans = re.sub(r"<think>.*?</think>", "", pred_ans, flags=re.DOTALL).strip()
    pred_ans = re.sub(
        r"\n(Note|Explanation|Source|Reference):.*", "", pred_ans,
        flags=re.DOTALL | re.IGNORECASE
    )
    return pred_ans.strip()
