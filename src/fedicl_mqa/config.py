from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .io import object_hash


@dataclass(slots=True)
class ExperimentSettings:
    name: str = "fedicl_mqa"
    output_dir: str = "outputs/default"
    data_seed: int = 20260901
    training_seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    deterministic: bool = True


@dataclass(slots=True)
class DataSettings:
    dataset: str = "medqa"
    medqa_id: str = "openlifescienceai/medqa"
    medmcqa_id: str = "openlifescienceai/medmcqa"
    revision: str = "main"
    num_clients: int = 5
    dirichlet_alpha: float = 0.5
    medqa_fit_ratio: float = 0.70
    medmcqa_fit_ratio: float = 0.80
    min_support_per_client: int = 5
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_test_samples: int | None = None


@dataclass(slots=True)
class ModelSettings:
    id: str = "Qwen/Qwen3-0.6B"
    revision: str = "main"
    dtype: str = "bfloat16"
    attention: str = "sdpa"
    trust_remote_code: bool = False
    gradient_checkpointing: bool = False
    use_cache: bool = False
    max_seq_length: int = 2048
    max_new_tokens: int = 64


@dataclass(slots=True)
class LoraSettings:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass(slots=True)
class TrainingSettings:
    local_epochs: int = 1
    fl_round_candidates: list[int] = field(default_factory=lambda: [4, 6, 8])
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    train_micro_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    optimizer: str = "adamw_torch_fused"
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    save_every_steps: int = 250
    checkpoint_keep: int = 3


@dataclass(slots=True)
class RetrievalSettings:
    encoder_id: str = "BAAI/bge-small-en-v1.5"
    encoder_revision: str = "main"
    top_k: int = 5
    initial_pool: int = 50
    expanded_pool: int = 100
    duplicate_similarity_threshold: float = 0.92
    lexical_jaccard_threshold: float = 0.85
    alpha: float = 1.0
    beta: float = 0.15
    gamma: float = 0.20


@dataclass(slots=True)
class EvaluationSettings:
    semantic_match_threshold: float = 0.82
    semantic_margin: float = 0.05
    bootstrap_samples: int = 10_000
    confidence_level: float = 0.95
    generation_batch_size: int = 16
    likelihood_candidate_batch_size: int = 8


@dataclass(slots=True)
class HardwareSettings:
    device: str = "cuda"
    tf32: bool = True
    bf16: bool = True
    compile: bool = False
    empty_cache_between_clients: bool = True


@dataclass(slots=True)
class Config:
    experiment: ExperimentSettings = field(default_factory=ExperimentSettings)
    data: DataSettings = field(default_factory=DataSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    lora: LoraSettings = field(default_factory=LoraSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    hardware: HardwareSettings = field(default_factory=HardwareSettings)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - full runtime only
            raise RuntimeError("PyYAML is required to read configuration files") from exc
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls.from_mapping(payload)

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        path = Path(path)
        if path.suffix.casefold() == ".json":
            import json

            with path.open("r", encoding="utf-8") as handle:
                return cls.from_mapping(json.load(handle))
        return cls.from_yaml(path)

    @classmethod
    def from_mapping(cls, payload: Any) -> Config:
        if not isinstance(payload, Mapping):
            raise ValueError("configuration root must be a mapping")
        config = cls(
            experiment=_construct(ExperimentSettings, payload.get("experiment")),
            data=_construct(DataSettings, payload.get("data")),
            model=_construct(ModelSettings, payload.get("model")),
            lora=_construct(LoraSettings, payload.get("lora")),
            training=_construct(TrainingSettings, payload.get("training")),
            retrieval=_construct(RetrievalSettings, payload.get("retrieval")),
            evaluation=_construct(EvaluationSettings, payload.get("evaluation")),
            hardware=_construct(HardwareSettings, payload.get("hardware")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.data.dataset not in {"medqa", "medmcqa"}:
            raise ValueError("data.dataset must be medqa or medmcqa")
        if self.data.num_clients != 5:
            raise ValueError("the sealed protocol requires exactly five clients")
        if self.data.dirichlet_alpha <= 0:
            raise ValueError("data.dirichlet_alpha must be positive")
        if not 0 < self.data.medqa_fit_ratio < 1 or not 0 < self.data.medmcqa_fit_ratio < 1:
            raise ValueError("fit ratios must be between zero and one")
        if self.data.min_support_per_client < self.retrieval.top_k:
            raise ValueError("each client needs at least top_k support examples")
        if self.retrieval.top_k != 5:
            raise ValueError("the sealed protocol requires exactly five exemplars")
        if self.lora.rank != 16 or self.lora.alpha != 32 or self.lora.dropout != 0.05:
            raise ValueError("LoRA rank/alpha/dropout must remain 16/32/0.05")
        if self.training.local_epochs != 1:
            raise ValueError("the sealed protocol requires one local epoch")
        if sorted(self.training.fl_round_candidates) != [4, 6, 8]:
            raise ValueError("FL round candidates must be [4, 6, 8]")
        if len(set(self.experiment.training_seeds)) != len(self.experiment.training_seeds):
            raise ValueError("training seeds must be unique")
        if not self.experiment.training_seeds:
            raise ValueError("at least one training seed is required")
        if self.model.dtype != "bfloat16" or not self.hardware.bf16:
            raise ValueError("the A5000 profile requires BF16")
        if self.hardware.compile:
            raise ValueError(
                "hardware.compile is disabled to preserve exact PEFT checkpoint serialization"
            )
        if self.model.max_seq_length <= self.model.max_new_tokens:
            raise ValueError("model.max_seq_length must exceed model.max_new_tokens")
        positive_batches = {
            "train_micro_batch_size": self.training.train_micro_batch_size,
            "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
            "generation_batch_size": self.evaluation.generation_batch_size,
            "likelihood_candidate_batch_size": self.evaluation.likelihood_candidate_batch_size,
        }
        if any(value <= 0 for value in positive_batches.values()):
            raise ValueError(f"batch sizes must be positive: {positive_batches}")
        if self.evaluation.bootstrap_samples <= 0:
            raise ValueError("evaluation.bootstrap_samples must be positive")
        if not 0 < self.evaluation.confidence_level < 1:
            raise ValueError("evaluation.confidence_level must be between zero and one")

    @property
    def hash(self) -> str:
        return object_hash(asdict(self))

    @property
    def dataset_id(self) -> str:
        return self.data.medqa_id if self.data.dataset == "medqa" else self.data.medmcqa_id

    @property
    def fit_ratio(self) -> float:
        return (
            self.data.medqa_fit_ratio
            if self.data.dataset == "medqa"
            else self.data.medmcqa_fit_ratio
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct(kind: type[Any], values: Any) -> Any:
    if values is None:
        return kind()
    if not isinstance(values, Mapping):
        raise ValueError(f"{kind.__name__} configuration must be a mapping")
    known = set(kind.__dataclass_fields__)
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {kind.__name__} field(s): {sorted(unknown)}")
    return kind(**dict(values))
