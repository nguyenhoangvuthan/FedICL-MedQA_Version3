from __future__ import annotations

import logging
import math
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from fedicl_mqa.core.config import Config
from fedicl_mqa.core.io import object_hash
from fedicl_mqa.core.schema import MCQExample
from fedicl_mqa.modeling.loader import ModelBundle, chat_prefix, gpu_telemetry
from fedicl_mqa.modeling.prompting import build_prompt, training_completion
from fedicl_mqa.training.checkpointing import CheckpointManager, TrainerState


class AnswerOnlyDataset:
    def __init__(
        self,
        examples: Sequence[MCQExample],
        tokenizer: Any,
        max_length: int,
        exemplars: Mapping[str, Sequence[MCQExample]] | None = None,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.exemplars = exemplars
        if exemplars is not None:
            if set(exemplars) != {q.example_id for q in examples}:
                raise ValueError("training exemplar plan must cover exactly the fit cohort")
            for q in examples:
                demos = exemplars[q.example_id]
                if len(demos) != 5 or len({d.example_id for d in demos}) != 5:
                    raise ValueError("training requires five distinct exemplars per target")
                if any(d.split != "train" or d.example_id == q.example_id for d in demos):
                    raise ValueError("training exemplars must be train-only and exclude the target")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        demos = () if self.exemplars is None else self.exemplars[example.example_id]
        prompt_ids = chat_prefix(self.tokenizer, build_prompt(example, demos), tokenize=True)
        completion_ids = self.tokenizer(training_completion(example), add_special_tokens=False)[
            "input_ids"
        ]
        if self.tokenizer.eos_token_id is not None:
            completion_ids = completion_ids + [self.tokenizer.eos_token_id]
        if len(completion_ids) >= self.max_length:
            raise ValueError(
                "an answer completion is too long for max_length: "
                f"item={example.example_id} completion_tokens={len(completion_ids)}"
            )
        if len(prompt_ids) + len(completion_ids) > self.max_length:
            raise ValueError(
                "a training item exceeds model.max_seq_length; exclude it during cohort "
                f"preparation instead of silently truncating it: item={example.example_id} "
                f"tokens={len(prompt_ids) + len(completion_ids)}"
            )
        prompt_ids = list(prompt_ids)
        input_ids = prompt_ids + completion_ids
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + completion_ids,
        }


class AnswerCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: Sequence[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = math.ceil(max(len(item["input_ids"]) for item in features) / 8) * 8
        pad_id = self.tokenizer.pad_token_id
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            padding = width - len(item["input_ids"])
            result["input_ids"].append(item["input_ids"] + [pad_id] * padding)
            result["attention_mask"].append(item["attention_mask"] + [0] * padding)
            result["labels"].append(item["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


logger = logging.getLogger(__name__)

# Progress cadence, in optimizer steps. Deliberately a module constant rather than a
# TrainingSettings field: Config.hash covers every config field, so adding one here
# would change the sealed hash and invalidate prepared data and existing checkpoints.
LOG_EVERY_STEPS = 10


def _forward_batch_size(device: Any, configured: int) -> int:
    # Bound activation/full-vocabulary loss memory before backward can OOM. Keep
    # DataLoader batches intact: checkpoint batch offsets and optimizer steps refer
    # to those logical batches, including in historical sealed runs.
    return 1 if device.type == "cuda" else configured


def _training_chunks(
    batch: dict[str, Any], limit: int
) -> Iterator[tuple[dict[str, Any], float]]:
    """Split a host batch, preserving the causal LM's mean over supervised tokens."""
    rows = batch["input_ids"].shape[0]
    if rows <= limit:
        yield batch, 1.0
        return
    # Causal LM loss shifts labels by one; prompt/padding labels are ignored.
    counts = (batch["labels"][:, 1:] != -100).sum(dim=1)
    total = int(counts.sum().item())
    if total == 0:
        raise ValueError("training batch has no supervised next-token targets")
    for start in range(0, rows, limit):
        stop = start + limit
        count = int(counts[start:stop].sum().item())
        if count == 0:
            continue
        chunk = {key: value[start:stop] for key, value in batch.items()}
        # AnswerCollator right-pads. Remove padding introduced by other chunks;
        # preserve every prompt and completion token and the multiple-of-eight pad.
        length = int(chunk["attention_mask"].sum(dim=1).max().item())
        width = math.ceil(length / 8) * 8
        yield {key: value[:, :width] for key, value in chunk.items()}, count / total


def _synchronize_before_release(device: Any) -> None:
    """Surface queued CUDA failures before tensor destructors run on Windows."""
    if sys.platform == "win32" and device.type == "cuda":
        import torch

        torch.cuda.synchronize(device)


def create_optimizer(model: Any, config: Config) -> Any:
    import torch

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    kwargs = {
        "lr": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
    }
    if config.training.optimizer == "adamw_torch_fused" and torch.cuda.is_available():
        try:
            return torch.optim.AdamW(parameters, fused=True, **kwargs)
        except TypeError:
            pass
    return torch.optim.AdamW(parameters, **kwargs)


def train(
    bundle: ModelBundle,
    examples: Sequence[MCQExample],
    config: Config,
    *,
    seed: int,
    epochs: int,
    kind: str,
    checkpoint_manager: CheckpointManager | None = None,
    resume: str | None = None,
    initial_state: TrainerState | None = None,
    exemplars: Mapping[str, Sequence[MCQExample]] | None = None,
) -> tuple[TrainerState, dict[str, float | str]]:
    import torch
    from torch.utils.data import DataLoader

    if not examples:
        raise ValueError("training data cannot be empty")
    model, tokenizer = bundle.model, bundle.tokenizer
    forward_batch_size = _forward_batch_size(
        bundle.device, config.training.train_micro_batch_size
    )
    tokenizer.padding_side = "right"
    dataset = AnswerOnlyDataset(examples, tokenizer, config.model.max_seq_length, exemplars)
    context_hash = (
        object_hash(
            {q.example_id: [e.example_id for e in exemplars[q.example_id]] for q in examples}
        )
        if exemplars is not None
        else None
    )
    context_extra = {"training_context_sha256": context_hash} if context_hash else {}
    context_extra["forward_micro_batch_size"] = forward_batch_size
    optimizer = create_optimizer(model, config)
    state = initial_state or TrainerState(kind=kind, seed=seed)
    if checkpoint_manager and resume:
        latest = (
            checkpoint_manager.latest() if resume == "auto" else checkpoint_manager.resolve(resume)
        )
        if latest is not None:
            loaded = checkpoint_manager.load(
                latest, model=model, optimizer=optimizer, restore_rng=True
            )
            state = loaded.trainer_state
            if loaded.extra.get("training_context_sha256") != context_hash:
                raise ValueError("resume checkpoint has a different training exemplar context")
            logger.info(
                "%s: resumed %s (step %d, epoch %d, batch %d)",
                kind,
                latest,
                state.global_step,
                state.epoch + 1,
                state.batch_in_epoch,
            )

    model.train()
    model.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)
    initial_target_exposures = state.target_exposures
    initial_optimizer_updates = state.optimizer_updates
    initial_elapsed_seconds = state.elapsed_seconds
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    logger.info(
        "%s: %d examples, starting at epoch %d/%d, batch %d",
        kind,
        len(examples),
        min(state.epoch + 1, epochs),
        epochs,
        state.batch_in_epoch,
    )
    if forward_batch_size < config.training.train_micro_batch_size:
        logger.info(
            "%s: CUDA memory guard: forward/backward batch %d, logical batch %d, "
            "gradient accumulation %d; optimizer/checkpoint boundaries unchanged",
            kind,
            forward_batch_size,
            config.training.train_micro_batch_size,
            config.training.gradient_accumulation_steps,
        )
    for epoch in range(state.epoch, epochs):
        epoch_started = time.perf_counter()
        generator = torch.Generator()
        generator.manual_seed(seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=config.training.train_micro_batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=AnswerCollator(tokenizer),
            num_workers=config.training.dataloader_num_workers,
            pin_memory=config.training.pin_memory and bundle.device.type == "cuda",
            persistent_workers=config.training.dataloader_num_workers > 0,
        )
        total_batches = len(loader)
        skip_batches = state.batch_in_epoch if epoch == state.epoch else 0
        for batch_index, batch in enumerate(loader):
            if batch_index < skip_batches:
                continue
            group_start = (
                batch_index // config.training.gradient_accumulation_steps
            ) * config.training.gradient_accumulation_steps
            group_size = min(
                config.training.gradient_accumulation_steps, total_batches - group_start
            )
            should_step = (
                batch_index + 1
            ) % config.training.gradient_accumulation_steps == 0 or batch_index + 1 == total_batches
            # Keep diagnostic context on the host: querying CUDA after a device failure
            # can mask the original exception with another allocator/driver error.
            batch_shape = tuple(batch["input_ids"].shape)
            phase = "batch transfer"
            try:
                loss_value = 0.0
                for host_chunk, weight in _training_chunks(batch, forward_batch_size):
                    phase = "batch transfer"
                    chunk = {
                        key: value.to(bundle.device, non_blocking=True)
                        for key, value in host_chunk.items()
                    }
                    phase = "forward"
                    loss = model(**chunk).loss * (weight / group_size)
                    chunk_loss = float(loss.detach().item() * group_size)
                    if not math.isfinite(chunk_loss):
                        raise RuntimeError(f"non-finite training loss: {chunk_loss}")
                    loss_value += chunk_loss
                    phase = "backward"
                    loss.backward()
                    _synchronize_before_release(bundle.device)
                    # No graph or device inputs survive into the next forward.
                    del loss, chunk
                if should_step:
                    phase = "optimizer step"
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.training.max_grad_norm, error_if_nonfinite=True
                    )
                    optimizer.step()
                    _synchronize_before_release(bundle.device)
                    optimizer.zero_grad(set_to_none=True)
            except RuntimeError:
                logger.exception(
                    "%s: training failed during %s at epoch %d/%d batch %d/%d "
                    "(last completed step %d, input shape %s). "
                    "Restart with --resume auto --cuda-debug to diagnose from the last "
                    "committed checkpoint; no checkpoint is saved from this failed batch.",
                    kind,
                    phase,
                    epoch + 1,
                    epochs,
                    batch_index + 1,
                    total_batches,
                    state.global_step,
                    batch_shape,
                )
                raise
            state.target_exposures += int(batch_shape[0])
            del batch
            if should_step:
                state.global_step += 1
                state.optimizer_updates += 1
                state.batch_in_epoch = batch_index + 1
                if state.global_step % LOG_EVERY_STEPS == 0:
                    logger.info(
                        "%s: epoch %d/%d batch %d/%d step %d loss %.4f",
                        kind,
                        epoch + 1,
                        epochs,
                        batch_index + 1,
                        total_batches,
                        state.global_step,
                        loss_value,
                    )
                if (
                    checkpoint_manager
                    and config.training.save_every_steps > 0
                    and state.global_step % config.training.save_every_steps == 0
                ):
                    state.elapsed_seconds = initial_elapsed_seconds + (
                        time.perf_counter() - started
                    )
                    checkpoint_manager.save(
                        f"checkpoint-step-{state.global_step:08d}",
                        model=model,
                        optimizer=optimizer,
                        trainer_state=state,
                        extra={"last_loss": loss_value, **context_extra},
                    )
        state.epoch = epoch + 1
        state.batch_in_epoch = 0
        logger.info(
            "%s: epoch %d/%d complete in %.1fs, step %d",
            kind,
            state.epoch,
            epochs,
            time.perf_counter() - epoch_started,
            state.global_step,
        )
        if checkpoint_manager:
            state.elapsed_seconds = initial_elapsed_seconds + (time.perf_counter() - started)
            name = f"checkpoint-epoch-{state.epoch:04d}-step-{state.global_step:08d}"
            checkpoint_manager.save(
                name,
                model=model,
                optimizer=optimizer,
                trainer_state=state,
                extra=context_extra,
            )
            logger.info("%s: saved %s", kind, name)

    elapsed = time.perf_counter() - started
    state.elapsed_seconds = initial_elapsed_seconds + elapsed
    run_target_exposures = state.target_exposures - initial_target_exposures
    run_optimizer_updates = state.optimizer_updates - initial_optimizer_updates
    telemetry: dict[str, float | str] = {
        "wall_clock_seconds": state.elapsed_seconds,
        "examples_per_second": (
            state.target_exposures / state.elapsed_seconds if state.elapsed_seconds > 0 else 0.0
        ),
        "run_wall_clock_seconds": elapsed,
        "total_wall_clock_seconds": state.elapsed_seconds,
        "run_examples_per_second": run_target_exposures / elapsed if elapsed > 0 else 0.0,
        "total_examples_per_second": (
            state.target_exposures / state.elapsed_seconds if state.elapsed_seconds > 0 else 0.0
        ),
        "run_optimizer_updates": float(run_optimizer_updates),
        "run_target_exposures": float(run_target_exposures),
        "total_optimizer_updates": float(state.optimizer_updates),
        "total_target_exposures": float(state.target_exposures),
        "effective_batch_size": float(
            config.training.train_micro_batch_size * config.training.gradient_accumulation_steps
        ),
        "forward_micro_batch_size": float(forward_batch_size),
        "training_exemplars_per_target": 5 if exemplars is not None else 0,
        **gpu_telemetry(),
    }
    return state, telemetry
