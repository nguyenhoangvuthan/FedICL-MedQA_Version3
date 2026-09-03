from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from fedicl_mqa.evaluation.retrieval import RetrievedExample
from fedicl_mqa.core.schema import LABELS, MCQExample

SYSTEM_PROMPT = (
    "You are answering a medical multiple-choice question. "
    "Reason briefly, state the answer text, and end with exactly `Final answer: X`, "
    "where X is A, B, C, or D."
)


def render_question(example: MCQExample) -> str:
    choices = "\n".join(
        f"{label}. {answer}" for label, answer in zip(LABELS, example.options, strict=True)
    )
    return f"Question: {example.question}\n{choices}"


def render_demonstration(example: MCQExample) -> str:
    return (
        f"{render_question(example)}\n"
        f"Answer: {example.correct_answer}\n"
        f"Final answer: {example.answer_label}"
    )


@dataclass(frozen=True, slots=True)
class Prompt:
    system: str
    user: str


def build_prompt(
    query: MCQExample,
    exemplars: Sequence[RetrievedExample | MCQExample] = (),
) -> Prompt:
    blocks: list[str] = []
    for position, value in enumerate(exemplars, start=1):
        example = value.example if isinstance(value, RetrievedExample) else value
        blocks.append(f"Example {position}:\n{render_demonstration(example)}")
    blocks.append(f"Now answer this item:\n{render_question(query)}\nAnswer:")
    return Prompt(system=SYSTEM_PROMPT, user="\n\n".join(blocks))


def training_completion(example: MCQExample) -> str:
    return f" {example.correct_answer}\nFinal answer: {example.answer_label}"
