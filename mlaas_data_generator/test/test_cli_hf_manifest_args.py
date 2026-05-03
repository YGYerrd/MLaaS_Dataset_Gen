from __future__ import annotations

from mlaas_data_generator.cli.main import build_parser
from mlaas_data_generator.cli.manifest.cmd_hf_manifest import _parse_csv_arg


def test_parse_csv_arg_accepts_mixed_comma_and_space_values() -> None:
    assert _parse_csv_arg(["text_classification,token_classification", "sentence_similarity"]) == [
        "text_classification",
        "token_classification",
        "sentence_similarity",
    ]


def test_hf_manifest_parser_accepts_split_task_keys_and_training_regimes() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "hf-manifest",
            "--task-keys",
            "text_classification,token_classification,",
            "sentence_similarity,fill_mask",
            "--training-regimes",
            "finetune_transfer,",
            "inference_only",
        ]
    )

    assert _parse_csv_arg(args.task_keys) == [
        "text_classification",
        "token_classification",
        "sentence_similarity",
        "fill_mask",
    ]
    assert _parse_csv_arg(args.training_regimes) == [
        "finetune_transfer",
        "inference_only",
    ]


def test_hf_manifest_parser_accepts_smoketest_resource_tier() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "hf-manifest",
            "--resource-tier",
            "smoketest",
        ]
    )

    assert args.resource_tier == "smoketest"
