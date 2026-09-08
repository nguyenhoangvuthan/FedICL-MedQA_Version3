"""Report how many sealed cohort items build a prompt too long to evaluate.

Diagnostic only: reads the sealed configuration, the partition and the retrieval audit,
and rebuilds each query's ICL prompt to measure it. It loads the tokenizer but never the
model, so it needs no GPU and finishes in about a minute.

    python scripts/check_prompt_budget.py --config outputs/a5000/sealed_config.json

evaluate_arm rejects an item when token_count + max_new_tokens > max_seq_length, which is
what stops the evaluate-all step. That check runs per item during evaluation, so a single
oversized item can halt a sweep hours in; this script finds every such item up front.

The count is measured against the audit's frozen Top-5. F2 reranks within the same
candidate pool, so it may select a different five and its own worst case can be larger;
--pool-worst-case measures the largest five in each pool instead, which bounds every arm.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from fedicl_mqa.core.config import Config
from fedicl_mqa.core.io import read_json
from fedicl_mqa.data.preparation import load_partition
from fedicl_mqa.modeling.loader import chat_prefix
from fedicl_mqa.modeling.prompting import build_prompt, training_completion
from fedicl_mqa.core.schema import MCQExample


def _tokenizer(config: Config):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        config.model.id,
        revision=config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
        use_fast=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="the sealed_config.json of the run")
    parser.add_argument(
        "--training-lengths",
        action="store_true",
        help="measure fit-split training sequences, which carry no exemplars",
    )
    parser.add_argument(
        "--pool-worst-case",
        action="store_true",
        help="measure the longest five in each candidate pool, bounding F2 as well",
    )
    args = parser.parse_args()

    config = Config.from_file(args.config)
    output_root = Path(config.experiment.output_dir)
    data_root = output_root / "data" / config.data.dataset
    audit = read_json(data_root / "retrieval_audit.json")

    if audit.get("config_hash") != config.hash:
        raise SystemExit("retrieval audit was sealed against a different configuration")

    by_id: dict[str, MCQExample] = {}
    for client in load_partition(data_root, expected_config_hash=config.hash).values():
        for examples in client.values():
            for example in examples:
                by_id[example.example_id] = example

    tokenizer = _tokenizer(config)
    budget = config.model.max_seq_length - config.model.max_new_tokens

    if args.training_lengths:
        # Training builds a prompt with no exemplars, so its sequences are far shorter
        # than an ICL prompt. If none reach max_seq_length then no training sequence was
        # ever truncated, and raising the limit would not change a single training input
        # even though it does change the config hash.
        lengths = []
        for client in load_partition(data_root, expected_config_hash=config.hash).values():
            for example in client["fit"]:
                rendered = build_prompt(example)
                lengths.append(
                    len(chat_prefix(tokenizer, rendered, tokenize=True))
                    + len(tokenizer(training_completion(example))["input_ids"])
                )
        lengths.sort()
        print(f"training sequences : {len(lengths)}")
        print(f"truncation limit   : {config.model.max_seq_length}")
        print(f"median             : {lengths[len(lengths) // 2]}")
        print(f"p95                : {lengths[int(len(lengths) * 0.95)]}")
        print(f"max                : {lengths[-1]}")
        truncated = sum(1 for value in lengths if value > config.model.max_seq_length)
        print(f"truncated at 2048  : {truncated}")
        return
    counts: list[int] = []
    offenders: list[tuple[str, str, int]] = []
    per_split: Counter[str] = Counter()

    for row in audit["queries"]:
        query = by_id[row["query_id"]]
        if args.pool_worst_case:
            pool = [by_id[value] for value in row["candidate_pool_ids"]]
            chosen = sorted(pool, key=lambda item: len(str(item.question)), reverse=True)[
                : audit["top_k"]
            ]
        else:
            chosen = [by_id[value] for value in row["top5_exemplar_ids"]]
        prompt = build_prompt(query, chosen)
        tokens = len(chat_prefix(tokenizer, prompt, tokenize=True))
        counts.append(tokens)
        if tokens > budget:
            offenders.append((row["query_id"], row["split"], tokens))
            per_split[row["split"]] += 1

    counts.sort()
    total = len(counts)
    print(f"queries measured : {total}")
    print(f"budget           : {budget} tokens "
          f"({config.model.max_seq_length} context - {config.model.max_new_tokens} generation)")
    print(f"median           : {counts[total // 2]}")
    print(f"p95              : {counts[int(total * 0.95)]}")
    print(f"max              : {counts[-1]}")
    print(f"over budget      : {len(offenders)} ({len(offenders) / total:.3%})")
    for split, count in sorted(per_split.items()):
        print(f"  {split}: {count}")
    if offenders:
        print("\nlongest offenders:")
        for query_id, split, tokens in sorted(offenders, key=lambda row: -row[2])[:20]:
            print(f"  {query_id}  {split}  {tokens}")


if __name__ == "__main__":
    main()
