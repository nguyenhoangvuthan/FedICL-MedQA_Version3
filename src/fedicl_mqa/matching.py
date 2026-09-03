from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .schema import LABELS, MCQExample, normalize_text

_TERMINAL_LABEL = re.compile(r"Final\s+answer\s*:\s*([A-D])\s*[.)]?\s*$", re.IGNORECASE)


class MatchEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> Any: ...


@dataclass(frozen=True, slots=True)
class MatchResult:
    label: int | None
    stage: str
    confidence: float | None = None
    margin: float | None = None

    @property
    def resolved(self) -> bool:
        return self.label is not None


class AnswerMatcher:
    def __init__(
        self,
        encoder: MatchEncoder | None = None,
        *,
        semantic_threshold: float = 0.82,
        semantic_margin: float = 0.05,
    ) -> None:
        self.encoder = encoder
        self.semantic_threshold = semantic_threshold
        self.semantic_margin = semantic_margin

    def match(self, generated: str, item: MCQExample) -> MatchResult:
        terminal = _TERMINAL_LABEL.search(generated)
        if terminal:
            return MatchResult(LABELS.index(terminal.group(1).upper()), "terminal_label", 1.0, 1.0)

        normalized = normalize_text(generated)
        exact = [
            index
            for index, option in enumerate(item.options)
            if normalized == normalize_text(option)
        ]
        if len(exact) == 1:
            return MatchResult(exact[0], "exact_option", 1.0, 1.0)
        if self.encoder is None or not normalized:
            return MatchResult(None, "unresolved")

        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - full runtime only
            raise RuntimeError("numpy is required for semantic answer matching") from exc
        vectors = np.asarray(self.encoder.encode([generated, *item.options]), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        scores = vectors[1:] @ vectors[0]
        order = np.argsort(-scores)
        best, runner_up = int(order[0]), int(order[1])
        confidence = float(scores[best])
        margin = confidence - float(scores[runner_up])
        if confidence < self.semantic_threshold or margin < self.semantic_margin:
            return MatchResult(None, "unresolved", confidence, margin)
        return MatchResult(best, "semantic_option", confidence, margin)
