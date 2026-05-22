from typing import List
import re


def get_inference_system_prompt() -> str:
    return (
        "You are a reading comprehension assistant. "
        "Extract the shortest exact answer span from the context that answers the question. "
        "Reply with ONLY the answer (no explanation). If the context does not contain the answer, say 'CANNOTANSWER'."
    )


def get_inference_user_prompt(query: str, context_list: List[str]) -> str:
    passages = "\n".join(f"({i+1}) {t}" for i, t in enumerate(context_list))
    return f"Passages:\n{passages}\n\nQ: {query}\nA (extracted span or CANNOTANSWER):"


def parse_generated_answer(pred_ans: str) -> str:
    if "assistant" in pred_ans.lower():
        parts = re.split(r"(?i)assistant\s*\n?", pred_ans)
        pred_ans = parts[-1].strip()
    # Strip Qwen3 thinking blocks
    pred_ans = re.sub(r"<think>.*?</think>", "", pred_ans, flags=re.DOTALL).strip()
    return pred_ans.strip()
