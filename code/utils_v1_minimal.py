from typing import List
import re


def get_inference_system_prompt() -> str:
    return ""


def get_inference_user_prompt(query: str, context_list: List[str]) -> str:
    ctx = "\n\n".join(f"[{i+1}] {t}" for i, t in enumerate(context_list))
    return (
        f"{ctx}\n\n"
        f"Question: {query}\n"
        f"Short answer (1 sentence max, or CANNOTANSWER):"
    )


def parse_generated_answer(pred_ans: str) -> str:
    if "assistant" in pred_ans.lower():
        parts = re.split(r"(?i)assistant\s*\n?", pred_ans)
        pred_ans = parts[-1].strip()
    # Strip Qwen3 thinking blocks
    pred_ans = re.sub(r"<think>.*?</think>", "", pred_ans, flags=re.DOTALL).strip()
    return pred_ans.strip()
