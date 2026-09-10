from __future__ import annotations

import argparse
import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fedicl_mqa.cli import paths, pipeline
from fedicl_mqa.cli.commands import evaluation, training
from fedicl_mqa.core.config import Config, ControlSettings, ICLTrainingSettings
from fedicl_mqa.core.io import read_json, write_json
from fedicl_mqa.core.schema import MCQExample
from fedicl_mqa.evaluation.arms import ARMS, active_arms
from fedicl_mqa.training import context, loop
from fedicl_mqa.training.checkpointing import CheckpointManager, TrainerState
from tests.evaluation.test_retrieval import _HashEncoder, _LengthTokenizer
from tests.training.test_loop import _config, _examples


class Tokenizer(_LengthTokenizer):
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [3] * len(text)}


def item(name, text):
    return MCQExample(name, text, ("a", "b", "c", "d"), 0, "train", subject="Anatomy")


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = Config()
        self.config.experiment.output_dir = self.tmp.name
        self.config.data.dataset = "medmcqa"
        self.config.controls = ControlSettings()
        self.config.icl_training = ICLTrainingSettings()
        self.config.model.max_seq_length = 1024
        self.config.retrieval.duplicate_similarity_threshold = 0.99999
        self.parts = {}
        for c in range(5):
            q = item(f"q{c}", f"uniquequery{c}")
            support = [item(f"s{c}-{i}", f"medical{i} topic{c} fact{i}") for i in range(10)]
            # Same text with a different ID must be excluded by the retriever.
            support.append(replace(q, example_id=f"duplicate{c}"))
            self.parts[c] = {
                "fit": [q, item(f"long{c}", "length " * 1000)],
                "support": support,
                "validation": [],
                "test": [],
            }
        write_json(context.plan_path(self.config).with_name("file_hashes.json"), {"fixture": "v1"})
        write_json(
            paths.checkpoint_root(self.config, "federated") / "selected_round.json", {"round": 6}
        )

    def prepare(self):
        return context.audit_training_context(
            self.config, self.parts, encoder=_HashEncoder(), tokenizer=Tokenizer()
        )

    def test_plan_uses_only_local_support_and_common_fit_cohort(self):
        plan = self.prepare()
        fit, demos, loaded = context.training_inputs(self.config, self.parts)
        self.assertEqual(plan, loaded)
        for c in range(5):
            self.assertEqual([q.example_id for q in fit[c]], [f"q{c}"])
            ids = {q.example_id for q in demos[c][f"q{c}"]}
            self.assertEqual(len(ids), 5)
            self.assertTrue(all(i.startswith(f"s{c}-") for i in ids))
            self.assertEqual(
                plan["clients"][str(c)]["excluded"],
                [{"query_id": f"long{c}", "reason": "context_budget"}],
            )
        with patch.object(context, "SentenceTransformerEncoder") as encoder:
            self.assertEqual(context.audit_training_context(self.config, self.parts), plan)
            encoder.assert_not_called()

    def test_modified_plan_or_partition_is_rejected(self):
        self.prepare()
        path = context.plan_path(self.config)
        original = path.read_bytes()
        plan = read_json(path)
        plan["clients"]["0"]["queries"][0]["exemplar_ids"].reverse()
        write_json(path, plan)
        with self.assertRaisesRegex(ValueError, "stale or modified"):
            context.load_plan(self.config)
        path.write_bytes(original)
        write_json(path.with_name("file_hashes.json"), {"fixture": "changed"})
        with self.assertRaisesRegex(ValueError, "stale or modified"):
            context.load_plan(self.config)

    def test_training_masks_every_exemplar_and_supervises_only_target_completion(self):
        self.prepare()
        fit, demos, _ = context.training_inputs(self.config, self.parts)
        tok = Tokenizer()
        base = loop.AnswerOnlyDataset(fit[0], tok, 1024)[0]
        conditioned = loop.AnswerOnlyDataset(fit[0], tok, 1024, demos[0])[0]
        self.assertGreater(len(conditioned["input_ids"]), len(base["input_ids"]))
        self.assertEqual(
            [x for x in conditioned["labels"] if x != -100],
            [x for x in base["labels"] if x != -100],
        )
        self.assertTrue(
            all(
                x == -100
                for x in conditioned["labels"][: -len([x for x in base["labels"] if x != -100])]
            )
        )
        with self.assertRaisesRegex(ValueError, "five distinct"):
            loop.AnswerOnlyDataset(fit[0], tok, 1024, {"q0": demos[0]["q0"][:4]})

    def test_protocol_prevents_reusing_wrong_context_checkpoints(self):
        plan = self.prepare()
        root = Path(self.tmp.name) / "training" / "local-icl" / "seed-42"
        context.bind_protocol(self.config, root, plan, train_icl=True, create=True)
        context.bind_protocol(self.config, root, plan, train_icl=True)
        with self.assertRaisesRegex(ValueError, "context differs"):
            context.bind_protocol(self.config, root, plan, train_icl=False)
        with self.assertRaisesRegex(ValueError, "context differs"):
            context.bind_protocol(self.config, root, {"sha256": "changed"}, train_icl=True)

    def test_cli_trains_baselines_and_icl_on_same_targets_and_separate_roots(self):
        self.prepare()
        with (
            patch.object(training, "seal_config", return_value=self.config),
            patch.object(training, "load_partition", return_value=self.parts),
            patch.object(training, "train_local_clients", return_value={}) as local,
        ):
            for mode in ("local", "local-icl"):
                training.command_train(
                    argparse.Namespace(
                        config="unused",
                        mode=mode,
                        seed=42,
                        all_seeds=False,
                        fl_round=None,
                        resume="auto",
                    )
                )
        normal, icl = local.call_args_list
        self.assertEqual(normal.args[1], icl.args[1])
        self.assertNotIn("client_exemplars", normal.kwargs)
        self.assertEqual(set(icl.kwargs["client_exemplars"][0]), {"q0"})
        self.assertNotEqual(normal.kwargs["output_root"], icl.kwargs["output_root"])

    def test_federated_icl_uses_selected_baseline_round_and_all_client_plans(self):
        self.prepare()
        trainer = SimpleNamespace(
            final_state=TrainerState(kind="federated-icl", seed=42),
            checkpoints=SimpleNamespace(root="new-family"),
            history=[],
        )
        with (
            patch.object(training, "seal_config", return_value=self.config),
            patch.object(training, "load_partition", return_value=self.parts),
            patch.object(training, "train_federated", return_value=trainer) as federated,
        ):
            training.command_train(
                argparse.Namespace(
                    config="unused",
                    mode="federated-icl",
                    seed=42,
                    all_seeds=False,
                    fl_round=None,
                    resume="auto",
                )
            )
        self.assertEqual(federated.call_args.kwargs["rounds"], 6)
        self.assertEqual(set(federated.call_args.kwargs["client_exemplars"]), set(range(5)))

    def test_eval_pairs_load_same_new_checkpoint_and_check_common_fit_budget(self):
        self.prepare()
        for family in ("local-icl", "federated-icl"):
            context.bind_protocol(
                self.config,
                paths.checkpoint_root(self.config, family) / "seed-42",
                context.load_plan(self.config),
                train_icl=True,
                create=True,
            )
        for arms in (("LT0", "LT1"), ("FT0", "FT1")):
            roots = []

            def manager(root, roots=roots, arms=arms, **kwargs):
                roots.append(Path(root))
                state = TrainerState(
                    kind="test",
                    seed=42,
                    epoch=1 if arms[0] == "LT0" else 6,
                    target_exposures=1 if arms[0] == "LT0" else 30,
                    round_index=6,
                )
                return SimpleNamespace(
                    load=lambda *args, **kw: SimpleNamespace(trainer_state=state)
                )

            with (
                patch.object(evaluation, "load_partition", return_value=self.parts),
                patch.object(
                    evaluation, "load_lora_bundle", return_value=SimpleNamespace(model=object())
                ),
                patch.object(evaluation, "adapter_state", return_value={}),
                patch.object(evaluation, "CheckpointManager", side_effect=manager),
            ):
                for arm in arms:
                    bundle, callback = evaluation._load_arm_checkpoint(self.config, arm, 42, None)
                    if callback:
                        callback(0, bundle.model)
            self.assertEqual(roots[0], roots[1])
            self.assertIn(ARMS[arms[0]].checkpoint_family, roots[0].parts)

    def test_opt_in_arms_and_pipeline_dependencies(self):
        self.assertEqual(len(active_arms(self.config)), 17)
        names = [s.name for s in pipeline.build_steps(self.config, split="test", force=False)]
        self.assertLess(names.index("audit-training-icl"), names.index("train-local"))
        self.assertLess(names.index("select-round"), names.index("train-federated-icl"))
        self.assertLess(names.index("train-federated-icl"), names.index("evaluate-all"))
        config = Config.from_file("configs/a5000-medmcqa-train-icl.yaml")
        self.assertIsNotNone(config.icl_training)
        self.assertNotIn("icl_training", Config().to_dict())
        self.config.icl_training = None
        self.assertEqual(len(active_arms(self.config)), 13)


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("torch", "peft", "transformers")),
    "requires PyTorch, PEFT and Transformers",
)
class TrainICLIntegrationTests(unittest.TestCase):
    def setUp(self):
        import torch

        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)

    def bundle(self):
        from tests.training.test_loop import ResumeIntegrationTests

        bundle = ResumeIntegrationTests()._bundle()
        bundle.tokenizer = Tokenizer()
        return bundle

    def test_real_lora_train_with_context_resumes_exactly_and_rejects_changed_demos(self):
        import torch
        from peft import get_peft_model_state_dict

        config = _config()
        demos = {
            q.example_id: [item(f"demo{i}", f"exemplar {i}") for i in range(5)] for q in _examples()
        }
        with tempfile.TemporaryDirectory() as temporary:

            def manager(name):
                return CheckpointManager(
                    Path(temporary) / name,
                    config_hash=config.hash,
                    model_id=config.model.id,
                    model_revision=config.model.revision,
                )

            full_manager, resumed_manager = manager("full"), manager("resume")
            full, interrupted = self.bundle(), self.bundle()
            # Recreate the reference seed immediately before each run (LoRA dropout).
            torch.manual_seed(123)
            state, telemetry = loop.train(
                full,
                _examples(),
                config,
                seed=42,
                epochs=2,
                kind="local-icl",
                checkpoint_manager=full_manager,
                exemplars=demos,
            )
            save = resumed_manager.save

            def stop(*args, **kwargs):
                save(*args, **kwargs)
                raise RuntimeError("interrupted")

            torch.manual_seed(123)
            with (
                patch.object(resumed_manager, "save", side_effect=stop),
                self.assertRaisesRegex(RuntimeError, "interrupted"),
            ):
                loop.train(
                    interrupted,
                    _examples(),
                    config,
                    seed=42,
                    epochs=2,
                    kind="local-icl",
                    checkpoint_manager=resumed_manager,
                    exemplars=demos,
                )
            restored = self.bundle()
            resumed, _ = loop.train(
                restored,
                _examples(),
                config,
                seed=42,
                epochs=2,
                kind="local-icl",
                checkpoint_manager=resumed_manager,
                exemplars=demos,
                resume="auto",
            )
            torch.testing.assert_close(
                get_peft_model_state_dict(full.model),
                get_peft_model_state_dict(restored.model),
                rtol=0,
                atol=0,
            )
            self.assertEqual(state.target_exposures, resumed.target_exposures)
            self.assertEqual(telemetry["training_exemplars_per_target"], 5)
            changed = {qid: list(reversed(values)) for qid, values in demos.items()}
            with self.assertRaisesRegex(ValueError, "different training exemplar context"):
                loop.train(
                    self.bundle(),
                    _examples(),
                    config,
                    seed=42,
                    epochs=2,
                    kind="local-icl",
                    checkpoint_manager=resumed_manager,
                    exemplars=changed,
                    resume="auto",
                )

    def test_real_fedavg_with_context_trains_each_client_and_resumes(self):
        from fedicl_mqa.training.federated import FederatedTrainer

        config = _config()
        config.data.num_clients = 2
        config.hardware.empty_cache_between_clients = False
        fit = {
            c: [replace(q, example_id=f"{c}-{q.example_id}") for q in _examples()[:2]]
            for c in range(2)
        }
        demos = {
            c: {
                q.example_id: [item(f"{c}-d{i}", f"client {c} demo {i}") for i in range(5)]
                for q in qs
            }
            for c, qs in fit.items()
        }
        with tempfile.TemporaryDirectory() as temporary:
            trainer = FederatedTrainer(self.bundle(), config, seed=42, run_root=temporary)
            state, history = trainer.run(fit, rounds=2, client_exemplars=demos)
            self.assertEqual((state.round_index, state.target_exposures), (2, 8))
            self.assertTrue(
                all(
                    metrics["training_exemplars_per_target"] == 5
                    for row in history
                    for metrics in row["clients"].values()
                )
            )
            restored = FederatedTrainer(self.bundle(), config, seed=42, run_root=temporary)
            with patch("fedicl_mqa.training.federated.train") as train_call:
                final, _ = restored.run(fit, rounds=2, client_exemplars=demos, resume="auto")
                train_call.assert_not_called()
            self.assertEqual(final.target_exposures, 8)
            with self.assertRaisesRegex(ValueError, "different training exemplar context"):
                restored.run(fit, rounds=2, client_exemplars=None, resume="auto")
