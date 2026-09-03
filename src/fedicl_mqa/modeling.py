from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .config import Config
from .prompting import Prompt
from .schema import LABELS, MCQExample


def configure_runtime(config: Config, seed: int) -> None:
    import os
    import random

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - full runtime only
        raise RuntimeError("torch is required for model execution") from exc
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = config.hardware.tf32
        torch.backends.cudnn.allow_tf32 = config.hardware.tf32
    torch.set_float32_matmul_precision("high")
    if config.experiment.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False


def resolve_model_revision(model_id: str, revision: str | None) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - full runtime only
        raise RuntimeError("huggingface-hub is required to resolve model revisions") from exc
    info = HfApi().model_info(repo_id=model_id, revision=revision or "main")
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a commit SHA for {model_id}")
    return info.sha


@dataclass(slots=True)
class ModelBundle:
    model: Any
    tokenizer: Any
    device: Any


def load_lora_bundle(config: Config, *, seed: int) -> ModelBundle:
    configure_runtime(config, seed)
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - full runtime only
        raise RuntimeError("install torch, transformers and peft to load the model") from exc

    if config.hardware.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    device = torch.device(config.hardware.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[config.model.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.id,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.id,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        torch_dtype=dtype,
        attn_implementation=config.model.attention,
        low_cpu_mem_usage=True,
    )
    base_model.config.use_cache = config.model.use_cache
    if config.model.gradient_checkpointing:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base_model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config).to(device)
    if config.hardware.compile:
        model = torch.compile(model, dynamic=False)
    return ModelBundle(model=model, tokenizer=tokenizer, device=device)


def chat_prefix(tokenizer: Any, prompt: Prompt, *, tokenize: bool) -> Any:
    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]
    kwargs = {"tokenize": tokenize, "add_generation_prompt": True}
    # Qwen3 defaults to a potentially long thinking trace; the experiment needs a
    # deterministic, short answer-selection completion.
    if "qwen3" in str(getattr(tokenizer, "name_or_path", "")).casefold():
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


class GenerationEngine:
    def __init__(self, bundle: ModelBundle, config: Config) -> None:
        self.model = bundle.model
        self.tokenizer = bundle.tokenizer
        self.device = bundle.device
        self.config = config

    def generate(self, prompts: Sequence[Prompt]) -> list[str]:
        if not prompts:
            return []
        import torch

        self.model.eval()
        self.tokenizer.padding_side = "left"
        rendered = [chat_prefix(self.tokenizer, prompt, tokenize=False) for prompt in prompts]
        encoded = self.tokenizer(
            rendered,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=False,
            pad_to_multiple_of=8,
        ).to(self.device)
        if encoded["input_ids"].shape[1] + self.config.model.max_new_tokens > (
            self.config.model.max_seq_length
        ):
            raise ValueError(
                "generation prompt exceeds the reserved context budget; do not silently "
                "truncate exemplars or the query"
            )
        input_width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.config.model.max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[:, input_width:]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def score_options(self, prompt: Prompt, item: MCQExample) -> tuple[int, float, list[float]]:
        """Mean conditional log-likelihood of answer-text + terminal-label completions."""
        return self.score_options_batch([prompt], [item])[0]

    def score_options_batch(
        self, prompts: Sequence[Prompt], items: Sequence[MCQExample]
    ) -> list[tuple[int, float, list[float]]]:
        """Vectorized four-option likelihood scoring with bounded candidate batches."""
        import torch

        if len(prompts) != len(items):
            raise ValueError("prompts and items must have the same length")
        if not prompts:
            return []
        self.model.eval()
        sequences: list[list[int]] = []
        starts: list[int] = []
        for prompt, item in zip(prompts, items, strict=True):
            prefix = chat_prefix(self.tokenizer, prompt, tokenize=False)
            prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
            if not prefix_ids:
                raise ValueError("chat template produced an empty prompt")
            for label, answer in zip(LABELS, item.options, strict=True):
                completion = f" {answer}\nFinal answer: {label}"
                completion_ids = self.tokenizer(completion, add_special_tokens=False)["input_ids"]
                if len(completion_ids) >= self.config.model.max_seq_length:
                    raise ValueError(
                        "an answer completion is too long for model.max_seq_length: "
                        f"item={item.example_id} completion_tokens={len(completion_ids)}"
                    )
                available = self.config.model.max_seq_length - len(completion_ids)
                if len(prefix_ids) > available:
                    raise ValueError(
                        "likelihood prompt exceeds model.max_seq_length; do not silently "
                        f"truncate exemplars or the query: item={item.example_id}"
                    )
                starts.append(len(prefix_ids))
                sequences.append(prefix_ids + completion_ids)

        candidate_scores: list[float] = []
        candidate_batch = self.config.evaluation.likelihood_candidate_batch_size
        self.tokenizer.padding_side = "right"
        for offset in range(0, len(sequences), candidate_batch):
            chunk_sequences = sequences[offset : offset + candidate_batch]
            chunk_starts = starts[offset : offset + candidate_batch]
            batch = self.tokenizer.pad(
                {"input_ids": chunk_sequences},
                padding=True,
                pad_to_multiple_of=8,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                # Keep the full-vocabulary logits in BF16. Upcasting the complete
                # [batch, sequence, vocabulary] tensor can exceed 24 GB even for a
                # 0.6B model. Only the answer-completion rows need FP32 log-softmax.
                logits = self.model(**batch).logits
                for row, (sequence, start) in enumerate(
                    zip(chunk_sequences, chunk_starts, strict=True)
                ):
                    targets = torch.tensor(sequence[start:], device=self.device)
                    positions = torch.arange(start - 1, len(sequence) - 1, device=self.device)
                    selected_logits = logits[row, positions].float()
                    token_scores = torch.log_softmax(selected_logits, dim=-1).gather(
                        1, targets.unsqueeze(1)
                    )
                    candidate_scores.append(float(token_scores.mean().item()))
            del logits, batch

        results: list[tuple[int, float, list[float]]] = []
        for offset in range(0, len(candidate_scores), 4):
            score_tensor = torch.tensor(candidate_scores[offset : offset + 4])
            probabilities = torch.softmax(score_tensor, dim=0)
            predicted = int(torch.argmax(score_tensor).item())
            values = [float(value) for value in probabilities.tolist()]
            results.append((predicted, values[predicted], values))
        return results


def gpu_telemetry() -> dict[str, float | str]:
    try:
        import torch
    except ImportError:
        return {"device": "unavailable"}
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    properties = torch.cuda.get_device_properties(0)
    return {
        "device": properties.name,
        "memory_total_bytes": float(properties.total_memory),
        "memory_allocated_bytes": float(torch.cuda.memory_allocated()),
        "memory_reserved_bytes": float(torch.cuda.memory_reserved()),
        "peak_memory_allocated_bytes": float(torch.cuda.max_memory_allocated()),
    }
