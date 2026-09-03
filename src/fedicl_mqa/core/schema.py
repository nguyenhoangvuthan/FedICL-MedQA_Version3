from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

LABELS = ("A", "B", "C", "D")


def normalize_text(value: str) -> str:
    """Canonical text used only for leakage and exact-match checks."""
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


@dataclass(frozen=True, slots=True)
class MCQExample:
    example_id: str
    question: str
    options: tuple[str, str, str, str]
    label: int
    split: str
    subject: str = "unknown"
    topic: str = "unknown"
    provenance_group: str | None = None
    explanation: str | None = None

    def __post_init__(self) -> None:
        if not self.example_id.strip():
            raise ValueError("example_id cannot be empty")
        if not self.question.strip():
            raise ValueError(f"question cannot be empty: {self.example_id}")
        if len(self.options) != 4 or any(not str(option).strip() for option in self.options):
            raise ValueError(f"exactly four non-empty options are required: {self.example_id}")
        if self.label not in range(4):
            raise ValueError(f"label must be in [0, 3]: {self.example_id}")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported split {self.split!r}: {self.example_id}")

    @property
    def answer_label(self) -> str:
        return LABELS[self.label]

    @property
    def correct_answer(self) -> str:
        return self.options[self.label]

    @property
    def normalized_question(self) -> str:
        return normalize_text(self.question)

    @property
    def question_options_hash(self) -> str:
        payload = {
            "question": self.normalized_question,
            "options": [normalize_text(option) for option in self.options],
        }
        packed = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return sha256(packed.encode("utf-8")).hexdigest()

    @property
    def retrieval_text(self) -> str:
        options = "\n".join(
            f"{label}. {value}" for label, value in zip(LABELS, self.options, strict=True)
        )
        return f"Question: {self.question}\n{options}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["options"] = list(self.options)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MCQExample:
        data = dict(payload)
        data["options"] = tuple(str(value) for value in data["options"])
        return cls(**data)


def label_to_index(value: Any, *, integer_base: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid MCQ label")
    if isinstance(value, int):
        index = value - integer_base
    else:
        text = str(value).strip().upper()
        if text in LABELS:
            return LABELS.index(text)
        if re.fullmatch(r"[0-4]", text):
            index = int(text) - integer_base
        else:
            raise ValueError(f"cannot parse MCQ label: {value!r}")
    if index not in range(4):
        raise ValueError(f"MCQ label outside [0, 3]: {value!r}")
    return index


def options_tuple(values: Mapping[str, Any] | Sequence[Any]) -> tuple[str, str, str, str]:
    if isinstance(values, Mapping):
        normalized = {str(key).strip().upper(): str(value) for key, value in values.items()}
        result = tuple(normalized[label] for label in LABELS)
    else:
        result = tuple(str(value) for value in values)
    if len(result) != 4:
        raise ValueError(f"expected four answer options, got {len(result)}")
    return result  # type: ignore[return-value]
