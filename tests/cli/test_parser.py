from __future__ import annotations

import unittest

from fedicl_mqa.cli import parser
from fedicl_mqa.cli.commands import evaluation

class ParserTests(unittest.TestCase):
    def test_evaluate_arm_defaults_to_the_test_split(self) -> None:
        args = parser.build_parser().parse_args(
            ["evaluate-arm", "--config", "c.yaml", "--arm", "B1"]
        )
        self.assertEqual(args.split, "test")
        self.assertEqual(args.arm, "B1")
        self.assertIsNone(args.gpu)
        self.assertFalse(args.force)

    def test_evaluate_all_needs_only_a_config(self) -> None:
        args = parser.build_parser().parse_args(["evaluate-all", "--config", "c.yaml"])
        self.assertEqual(args.split, "test")
        self.assertEqual(args.func, parser.command_evaluate_all)

    def test_gpu_flag_is_accepted_after_every_subcommand(self) -> None:
        for argv in (
            ["train", "--config", "c.yaml", "--mode", "local", "--seed", "42", "--gpu", "1"],
            ["evaluate-all", "--config", "c.yaml", "--gpu", "0"],
            ["doctor", "--config", "c.yaml", "--gpu", "1"],
        ):
            self.assertIn(parser.build_parser().parse_args(argv).gpu, (0, 1))

    def test_gpu_flag_rejects_devices_outside_the_pair(self) -> None:
        with self.assertRaises(SystemExit):
            parser.build_parser().parse_args(["evaluate-all", "--config", "c.yaml", "--gpu", "2"])


class SingleEpochBudgetTests(unittest.TestCase):
    """--fl-round 1 is a centralized-only diagnostic budget, not a protocol change."""

    def test_train_accepts_one_epoch(self) -> None:
        args = parser.build_parser().parse_args(
            ["train", "--config", "c.json", "--mode", "centralized", "--seed", "42",
             "--fl-round", "1"]
        )
        self.assertEqual(args.fl_round, 1)

    def test_validate_checkpoints_still_rejects_one(self) -> None:
        with self.assertRaises(SystemExit):
            parser.build_parser().parse_args(
                ["validate-checkpoints", "--config", "c.json", "--mode", "local-matched",
                 "--seed", "42", "--fl-round", "1"]
            )

    def test_matched_local_still_rejects_an_off_protocol_budget(self) -> None:
        from fedicl_mqa.training.workflows import train_local_clients

        with self.assertRaisesRegex(ValueError, "valid selected FL round"):
            train_local_clients(
                _config_with_candidates(), {}, seed=42, output_root=".", fl_rounds=1
            )


def _config_with_candidates():
    class _Training:
        fl_round_candidates = [4, 6, 8]

    class _Config:
        training = _Training()

    return _Config()


if __name__ == "__main__":
    unittest.main()
