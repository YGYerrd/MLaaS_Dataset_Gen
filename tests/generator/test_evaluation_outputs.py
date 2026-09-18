from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mlaas_data_generator.services.evaluation_outputs import (
    TaskOutput,
    UnsupportedTaskOutput,
    candidate_units,
    compare_task_outputs,
    derive_input_calibration,
    output_from_raw,
    perturb_input,
    prepare_task_input,
    resolve_image_calibration,
    predict_task_output,
    predict_task_outputs,
    task_output_quality,
)
from mlaas_data_generator.services.perturbation import _meaningful_units, _tensor_prediction


def test_classification_measures_probability_movement_even_when_class_is_constant():
    clean = output_from_raw([[0.9, 0.1]], task_family="classification")
    candidate = output_from_raw([[0.6, 0.4]], task_family="classification")
    result = compare_task_outputs(clean, candidate)
    assert result["components"]["class_agreement"] == 1
    assert result["components"]["probability_total_variation"] == pytest.approx(0.3)
    assert result["score"] == pytest.approx(0.85)
    assert task_output_quality(clean, 1)["score"] == 0
    assert compare_task_outputs(clean, clean)["score"] == 1


def test_classification_rejects_hard_label_as_probability_evidence():
    with pytest.raises(UnsupportedTaskOutput, match="probability_distribution"):
        output_from_raw(np.array([1]), task_family="classification")
    with pytest.raises(ValueError, match="invalid_classification_probability_distribution"):
        output_from_raw([[0.9, 0.9]], task_family="classification", config={"evaluation_scores_are_probabilities": True})


def test_generic_classification_batches_predict_proba_and_preserves_class_ids():
    class Classifier:
        classes_ = np.array([4, 9])
        calls = 0

        def predict_proba(self, batch):
            self.calls += 1
            return np.tile([0.2, 0.8], (len(batch), 1))

        def predict(self, batch):
            raise AssertionError("probabilities must be requested")

    model = Classifier()
    results = predict_task_outputs(model, [np.array([1, 2]), np.array([3, 4])], task_family="classification")
    assert model.calls == 1
    assert [result.prediction for result in results] == [9, 9]
    assert results[0].metadata["inference_backend"] == "generic_batched"
    assert task_output_quality(results[0], 9)["score"] == 1


def test_regression_uses_fixed_scale_near_zero_and_preserves_vector():
    clean = output_from_raw([[0.0, 0.0]], task_family="regression", config={"regression_target_scale": 10})
    candidate = output_from_raw([[1.0, 3.0]], task_family="regression")
    result = compare_task_outputs(clean, candidate)
    assert result["score"] == pytest.approx(0.8)
    assert result["components"]["prediction_absolute_delta"] == 2
    assert task_output_quality(candidate, [0.0, 0.0], target_scale=10)["mae"] == 2
    with pytest.raises(UnsupportedTaskOutput, match="fixed_positive_target_scale"):
        compare_task_outputs(candidate, candidate)


def test_token_output_uses_complete_valid_sequence_and_excludes_padding():
    x = {"input_ids": np.arange(260), "attention_mask": np.r_[np.ones(257), np.zeros(3)],
         "special_tokens_mask": np.r_[1, np.zeros(259)]}
    clean_ids = np.zeros(260, dtype=int)
    changed_ids = clean_ids.copy()
    changed_ids[200] = 1
    changed_ids[-1] = 1
    clean = output_from_raw(clean_ids, task_family="token_classification", x_sample=x, batched=False)
    candidate = output_from_raw(changed_ids, task_family="token_classification", x_sample=x, batched=False)
    result = compare_task_outputs(clean, candidate)
    assert len(candidate.prediction) == 260
    assert result["components"]["valid_position_count"] == 256
    assert result["score"] == pytest.approx(255 / 256)
    assert len(_tensor_prediction(torch.arange(260).unsqueeze(0))) == 260


def test_token_alignment_change_is_unsupported_instead_of_prefix_comparison():
    clean = TaskOutput("token_classification", np.array([0, 1]), valid_mask=np.array([True, True]))
    candidate = TaskOutput("token_classification", np.array([0, 1]), valid_mask=np.array([True, False]))
    with pytest.raises(UnsupportedTaskOutput, match="alignment_changed"):
        compare_task_outputs(clean, candidate)


def test_segmentation_preserves_complete_mask_and_per_class_overlap():
    left = np.zeros((20, 20), dtype=int)
    right = left.copy()
    left[:4, :4] = 1
    right[-4:, -4:] = 1
    clean = output_from_raw(left, task_family="segmentation", batched=False)
    candidate = output_from_raw(right, task_family="segmentation", batched=False)
    result = compare_task_outputs(clean, candidate)
    assert clean.prediction.shape == (20, 20)
    assert result["score"] < 0.5
    assert result["components"]["per_class_overlap"]["1"]["iou"] == 0
    assert result["components"]["pixel_agreement"] == pytest.approx(368 / 400)


def test_segmentation_reference_quality_upsamples_every_prediction_pixel():
    output = output_from_raw(np.array([[0, 1], [0, 1]]), task_family="segmentation", batched=False)
    reference = np.repeat(np.repeat(output.prediction, 4, axis=0), 4, axis=1)
    reference[0, 0] = 1
    result = task_output_quality(output, reference)
    assert result["score"] < 1
    assert result["pixel_accuracy"] == pytest.approx(63 / 64)


def test_segmentation_reference_quality_excludes_dataset_specific_void_id():
    output = output_from_raw(np.zeros((3, 3), dtype=int), task_family="segmentation", batched=False, meta={"ignore_index": 255})
    reference = np.zeros((3, 3), dtype=int)
    reference[0, 0] = 255
    quality = task_output_quality(output, reference)
    assert quality["score"] == 1
    assert "255" not in quality["per_class_overlap"]


def detection(boxes, labels):
    return output_from_raw({"boxes": boxes, "labels": labels, "scores": np.ones(len(labels)), "normalized": True}, task_family="detection")


def test_detection_matching_handles_reordering_class_changes_and_movement():
    clean = detection([[0, 0, 0.3, 0.3], [0.6, 0.6, 1, 1]], [1, 2])
    reorder = detection([[0.6, 0.6, 1, 1], [0, 0, 0.3, 0.3]], [2, 1])
    assert compare_task_outputs(clean, reorder)["score"] == 1
    changed = detection([[0.6, 0.6, 1, 1], [0.01, 0, 0.31, 0.3]], [7, 1])
    result = compare_task_outputs(clean, changed)
    assert result["components"]["matched_object_count"] == 2
    assert result["components"]["class_agreement"] == 0.5
    assert result["components"]["matched_box_movement"] > 0
    assert result["score"] < 0.5
    empty = detection([], [])
    assert compare_task_outputs(empty, empty)["components"]["empty_output_pair"]
    assert task_output_quality(empty, {"boxes": [[0, 0, 1, 1]], "labels": [1], "normalized": True})["score"] == 0


def test_generation_compares_actual_entire_generated_sequence():
    clean = output_from_raw(np.arange(300), task_family="generation", batched=False)
    ids = np.arange(300)
    ids[280] = 999
    candidate = output_from_raw(ids, task_family="generation", batched=False)
    assert len(clean.prediction) == 300
    assert compare_task_outputs(clean, candidate)["score"] == pytest.approx(299 / 300)
    assert task_output_quality(clean, "reference words")["score"] is None


def test_retrieval_requires_fixed_nondegenerate_candidate_pool():
    with pytest.raises(UnsupportedTaskOutput, match="frozen_candidate_pool"):
        output_from_raw([0.9], task_family="retrieval", batched=False)
    config = {"evaluation_retrieval_candidate_ids": ["a", "b", "c"]}
    clean = output_from_raw([0.9, 0.8, 0.1], task_family="retrieval", config=config, batched=False)
    changed = output_from_raw([0.1, 0.8, 0.9], task_family="retrieval", config=config, batched=False)
    result = compare_task_outputs(clean, changed)
    assert result["score"] == 0
    assert result["components"]["top1_agreement"] == 0


class FakeSpec:
    supports_generation = False

    def encode_batch(self, tokenizer, xb, yb, max_length, torch, device, **kwargs):
        assert yb is None, "prediction must not consume labels"
        assert kwargs["inference_only"]
        return {key: torch.as_tensor(value) for key, value in xb.items()}, None, {}

    def build_forward_inputs(self, enc, **kwargs):
        return enc

    def extract_logits(self, outputs):
        return outputs.logits


def hf_model(forward, spec=None, tokenizer=None):
    class Model(torch.nn.Module):
        def forward(self, **kwargs):
            return forward(kwargs)

    model = Model()
    return SimpleNamespace(core=SimpleNamespace(model=model, task_spec=spec or FakeSpec(), torch=torch,
                                               tokenizer=tokenizer, max_length=512, device="cpu", label_pad_value=-100,
                                               generation_config={}, _run_with_precision_context=lambda fn: fn()))


def test_hf_regression_does_not_softmax_scalar_logits_and_restores_training_mode():
    model = hf_model(lambda enc: SimpleNamespace(logits=torch.tensor([[0.01]]).repeat(len(enc["input_ids"]), 1)))
    results = predict_task_outputs(model, [{"input_ids": np.array([1, 2])}] * 2, task_family="regression")
    assert [result.prediction.item() for result in results] == pytest.approx([0.01, 0.01])
    assert model.core.model.training
    assert results[0].metadata["inference_backend"] == "hf_batched"


def test_hf_segmentation_logits_keep_spatial_masks():
    logits = torch.zeros((1, 2, 16, 16))
    logits[:, 1, 2:4, 2:4] = 8
    model = hf_model(lambda enc: SimpleNamespace(logits=logits))
    output = predict_task_output(model, {"pixel_values": np.zeros((3, 64, 64))}, task_family="segmentation")
    assert output.prediction.shape == (16, 16)
    assert output.prediction.sum() == 4
    assert output.scores.shape == (16, 16, 2)


def test_hf_detection_preserves_box_outputs_and_excludes_no_object_probability():
    model = hf_model(lambda enc: SimpleNamespace(logits=torch.tensor([[[8.0, 0.0, -8.0], [-8.0, -8.0, 8.0]]]),
                                                  pred_boxes=torch.tensor([[[0.5, 0.5, 0.2, 0.4], [0.2, 0.2, 0.1, 0.1]]])))
    model.core.model.config = SimpleNamespace(model_type="detr")
    output = predict_task_output(model, {"pixel_values": np.zeros((3, 24, 24))}, task_family="detection")
    assert len(output.prediction["boxes"]) == 1
    np.testing.assert_allclose(output.prediction["boxes"][0], [0.4, 0.3, 0.6, 0.7], atol=1e-6)


def test_hf_detection_uses_model_postprocessor_for_non_detr_score_contract():
    class Processor:
        def post_process_object_detection(self, outputs, threshold, target_sizes):
            assert tuple(target_sizes.shape) == (1, 2)
            assert torch.all(target_sizes == 1)
            return [{"boxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]]), "labels": torch.tensor([5]), "scores": torch.tensor([0.9])}]

    model = hf_model(lambda enc: SimpleNamespace(logits=torch.zeros((1, 2, 6)), pred_boxes=torch.zeros((1, 2, 4))))
    model.core.task_spec._image_processor = Processor()
    output = predict_task_output(model, {"pixel_values": np.zeros((3, 24, 24))}, task_family="detection")
    assert output.prediction["labels"].tolist() == [5]
    assert output.metadata["detection_postprocessor"].endswith("Processor")


def test_hf_retrieval_cannot_relabel_current_batch_as_frozen_candidate_pool():
    model = hf_model(lambda enc: SimpleNamespace(logits=torch.ones((2, 2))))
    with pytest.raises(UnsupportedTaskOutput, match="frozen_external_candidate_pool"):
        predict_task_outputs(model, [{"input_ids": np.array([4])}] * 2, task_family="retrieval",
                             config={"evaluation_retrieval_candidate_ids": ["a", "b"]})


def test_hf_generation_is_deterministic_and_preserves_all_generated_tokens():
    class GenerationSpec(FakeSpec):
        supports_generation = True

        def generate_predictions(self, model, enc, tokenizer, torch, config):
            assert config["do_sample"] is False
            return torch.arange(300).unsqueeze(0)

    tokenizer = SimpleNamespace(all_special_ids=[0], decode=lambda ids, **kwargs: "hello world")
    model = hf_model(lambda enc: (_ for _ in ()).throw(AssertionError("generation must not teacher-force")), GenerationSpec(), tokenizer)
    output = predict_task_output(model, {"input_ids": np.array([4, 5])}, task_family="generation")
    assert len(output.prediction) == 299
    assert task_output_quality(output, "hello world")["score"] == 1


def test_fill_mask_compares_original_queries_when_explanation_adds_masks():
    cfg = {"evaluation_mask_token_id": 99}
    x = {"input_ids": np.array([1, 99, 4]), "attention_mask": np.ones(3)}
    extra = {"input_ids": np.array([99, 99, 4]), "attention_mask": np.ones(3)}
    logits = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
    clean = output_from_raw(logits, task_family="fill_mask", x_sample=x, config=cfg, batched=False)
    changed = output_from_raw(logits, task_family="fill_mask", x_sample=extra, config=cfg, batched=False)
    assert clean.prediction.shape == (3,)
    assert clean.scores.shape == (1, 2)
    assert changed.scores.shape == (2, 2)
    assert compare_task_outputs(clean, changed)["score"] == 1


def test_candidate_units_are_independent_of_entity_annotations_and_image_grid_scales():
    x = {"input_ids": np.arange(12), "attention_mask": np.ones(12)}
    annotations = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    candidates = _meaningful_units(x, annotations, task_family="token_classification", limit=12)
    assert candidates == _meaningful_units(x, np.ones(12), task_family="token_classification", limit=12)
    assert ("token", 1) in candidates
    assert len(candidate_units(np.zeros((32, 32, 3)), task_family="classification", limit=64)) == 64


def test_tabular_candidates_group_onehot_features_and_exclude_immutable_groups():
    metadata = {"one_hot_feature_groups": [[0, 1, 2]], "feature_groups": {"coordinates": [3, 4]},
                "immutable_feature_indices": [6]}
    units = candidate_units(np.arange(8), task_family="classification", meta=metadata)
    assert units == [("feature_group", (3, 4)), ("feature_group", (0, 1, 2)), ("feature", 5), ("feature", 7)]
    metadata["immutable_feature_indices"].append(1)
    assert ("feature_group", (0, 1, 2)) not in candidate_units(np.arange(8), meta=metadata)


def test_token_candidates_include_real_endpoints_and_only_exclude_declared_boundaries():
    sample = {"input_ids": np.array([4, 5, 6, 7]), "attention_mask": np.ones(4)}
    assert candidate_units(sample, task_family="generation") == [("token", i) for i in range(4)]
    assert candidate_units(sample, task_family="token_classification", meta={"evaluation_boundary_token_ids": [4, 7]}) == [("token", 1), ("token", 2)]
    assert candidate_units(sample, task_family="fill_mask", meta={"evaluation_mask_token_id": 5}) == [("token", 0), ("token", 2), ("token", 3)]


def test_causal_preparation_removes_reference_tokens_using_only_supervision_mask():
    model = hf_model(lambda enc: None, tokenizer=SimpleNamespace(pad_token_id=0))
    model.core.task_spec.name = "causal_lm_generation"
    source = {"input_ids": np.array([11, 12, 51, 52, 0]), "attention_mask": np.array([1, 1, 1, 1, 0])}
    prepared = prepare_task_input(source, np.array([-100, -100, 51, 52, -100]), model=model)
    assert prepared["input_ids"].tolist() == [11, 12, 0, 0, 0]
    assert prepared["attention_mask"].tolist() == [1, 1, 0, 0, 0]
    assert source["input_ids"].tolist() == [11, 12, 51, 52, 0]
    different_labels = prepare_task_input(source, np.array([-100, -100, 81, 92, -100]), model=model)
    np.testing.assert_array_equal(prepared["input_ids"], different_labels["input_ids"])
    with pytest.raises(UnsupportedTaskOutput, match="prompt_boundary_missing"):
        prepare_task_input(source, np.array([11, 12, 51, 52, -100]), model=model)


def test_hf_valid_tokens_exclude_known_boundary_tokens_but_keep_unknown_words():
    tokenizer = SimpleNamespace(cls_token_id=1, sep_token_id=2, pad_token_id=0, unk_token_id=3)
    model = hf_model(lambda enc: SimpleNamespace(logits=torch.zeros((1, 5, 2))), tokenizer=tokenizer)
    output = predict_task_output(model, {"input_ids": np.array([1, 3, 17, 2, 0]), "attention_mask": np.array([1, 1, 1, 1, 0])}, task_family="token_classification")
    assert output.valid_mask.tolist() == [False, True, True, False, False]


def test_hf_token_adapter_uses_actual_spec_and_keeps_masks_out_of_model_forward():
    from mlaas_data_generator.models.adapters.hf_task import TokenClassificationSpec

    def forward(enc):
        assert "special_tokens_mask" not in enc
        return SimpleNamespace(logits=torch.zeros((1, 260, 3)))

    model = hf_model(forward, TokenClassificationSpec())
    sample = {"input_ids": np.arange(260), "attention_mask": np.ones(260),
              "special_tokens_mask": np.r_[1, np.zeros(258), 1]}
    result = predict_task_output(model, sample, task_family="token_classification")
    assert result.prediction.shape == (260,)
    assert result.valid_mask.sum() == 258


def test_tabular_calibration_uses_per_feature_scale_and_freezes_discrete_columns():
    values = np.column_stack([np.linspace(0, 1, 100), np.linspace(100, 10000, 100), np.tile([0, 1], 50)])
    config = derive_input_calibration(values, task_family="regression")
    x = values[50].copy()
    trial = perturb_input(x, np.random.default_rng(42), task_family="regression", config=config)
    assert trial.changed
    assert trial.sample[2] == x[2]
    assert abs(trial.sample[1] - x[1]) > 100 * abs(trial.sample[0] - x[0])
    assert np.all(trial.sample >= config["feature_min"])
    assert np.all(trial.sample <= config["feature_max"])
    np.testing.assert_array_equal(x, values[50])


def test_tabular_missing_calibration_and_relationship_violation_are_explicit():
    x = np.array([0.2, 0.8])
    assert perturb_input(x, np.random.default_rng(1), task_family="regression").status == "unsupported"
    config = {"feature_scales": [1, 1], "feature_min": [0, 0], "feature_max": [1, 1],
              "feature_constraints": [{"coefficients": [1, 1], "min": 1, "max": 1}]}
    trial = perturb_input(x, np.random.default_rng(1), task_family="regression", config=config)
    assert trial.status == "excluded"
    assert trial.reason == "feature_relationship_constraint"


def test_out_of_calibration_inputs_are_excluded_without_clipping_into_range():
    config = {"feature_scales": [1, 1], "feature_min": [0, 0], "feature_max": [1, 1]}
    x = np.array([9.0, 0.8])
    trial = perturb_input(x, np.random.default_rng(1), task_family="regression", config=config)
    assert trial.status == "excluded"
    assert trial.reason == "input_outside_calibrated_feature_range"
    np.testing.assert_array_equal(trial.sample, x)


@pytest.mark.parametrize("trial_index,operator", [(0, "gaussian_noise"), (1, "brightness"), (2, "gaussian_blur"), (3, "jpeg_compression")])
def test_image_operators_record_actual_changes_and_calibration(trial_index, operator):
    x = np.random.default_rng(12).uniform(0, 1, size=(3, 24, 24)).astype(np.float32)
    original = x.copy()
    trial = perturb_input(x, np.random.default_rng(1), task_family="classification", trial_index=trial_index)
    assert trial.operator == operator
    assert trial.changed and trial.status == "complete"
    assert trial.sample.shape == x.shape
    assert trial.sample.dtype == x.dtype
    assert 0 <= trial.sample.min() <= trial.sample.max() <= 1
    assert trial.metadata["value_range"] == [0, 1]
    np.testing.assert_array_equal(x, original)


def test_normalized_images_are_corrupted_in_physical_pixel_space():
    physical = np.full((3, 12, 12), 0.3, dtype=np.float32)
    mean = np.array([0.5, 0.4, 0.3])[:, None, None]
    std = np.array([0.2, 0.3, 0.4])[:, None, None]
    x = ((physical - mean) / std).astype(np.float32)
    config = {"image_mean": mean.reshape(-1).tolist(), "image_std": std.reshape(-1).tolist()}
    trial = perturb_input(x, np.random.default_rng(1), task_family="segmentation", config=config, trial_index=1)
    recovered = trial.sample * std + mean
    assert trial.status == "complete"
    np.testing.assert_allclose(recovered, physical + trial.metadata["brightness_offset"], atol=1e-7)
    assert trial.metadata["corruption_space"] == "physical_pixels"
    assert perturb_input(x, np.random.default_rng(1), task_family="segmentation").status == "unsupported"


def test_image_calibration_prefers_frozen_and_explicit_parameters_without_cache_reads(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", lambda *args, **kwargs: pytest.fail("explicit parameters must avoid loading assets"))
    meta = {"processor_config": {"image_mean": [0.4, 0.5, 0.6], "image_std": [0.2, 0.3, 0.4], "do_normalize": True}}
    config = {"image_mean": [0.1, 0.2, 0.3]}
    result = resolve_image_calibration(SimpleNamespace(), config=config, meta=meta)
    assert result["image_mean"] == config["image_mean"]
    assert result["image_std"] == meta["processor_config"]["image_std"]
    assert result["image_value_range"] == [0, 1]
    assert len(result["image_calibration_contract"]["sha256"]) == 64
    assert "image_mean" not in meta


def test_image_calibration_reads_only_cached_processor_and_records_effective_hash(monkeypatch):
    import transformers

    calls = []

    def load(source, **kwargs):
        calls.append((source, kwargs))
        return SimpleNamespace(to_dict=lambda: {"image_mean": [0.5] * 3, "image_std": [0.5] * 3,
                                               "do_normalize": False, "do_rescale": True, "rescale_factor": 1 / 255})

    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", load)
    model = SimpleNamespace(core=SimpleNamespace(model_source="cached/model", tokenizer=None))
    meta = {"hf_processor": "cached/model"}
    result = resolve_image_calibration(model, meta=meta)
    # The image loader explicitly requests do_normalize=True even if an
    # upstream processor's default says False.
    assert result["image_mean"] == [0.5] * 3
    assert calls == [("cached/model", {"local_files_only": True, "use_fast": False})]
    assert result["image_calibration_contract"]["processor_source"] == "cached/model"
    revised = resolve_image_calibration(model, config={"image_std": [0.2] * 3}, meta=meta)
    assert revised["image_std"] == [0.2] * 3
    assert revised["image_calibration_contract"]["sha256"] != result["image_calibration_contract"]["sha256"]


def test_image_calibration_missing_cached_assets_is_explicit(monkeypatch):
    import transformers

    def missing(*args, **kwargs):
        assert kwargs["local_files_only"]
        raise OSError("no cached processor")

    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", missing)
    model = SimpleNamespace(core=SimpleNamespace(model_source="missing/model", tokenizer=None))
    with pytest.raises(UnsupportedTaskOutput, match="unavailable_locally"):
        resolve_image_calibration(model)


def test_image_calibration_honors_disabled_normalization_with_processor_defaults():
    meta = {"processor_config": {"image_mean": [0.5] * 3, "image_std": [0.5] * 3,
                                 "do_normalize": False, "do_rescale": True, "rescale_factor": 1 / 255}}
    result = resolve_image_calibration(SimpleNamespace(), meta=meta)
    assert result["image_mean"] == [0.0]
    assert result["image_std"] == [1.0]
    assert result["image_value_range"] == [0, 1]
    assert result["image_calibration_contract"]["do_normalize"] is False


def test_unchanged_corruption_is_excluded():
    result = perturb_input(np.full((12, 12, 3), 0.5), np.random.default_rng(1), task_family="classification", trial_index=2)
    assert result.status == "excluded"
    assert result.reason == "unchanged_input"


def test_text_never_treats_random_mask_or_unvalidated_replacement_as_benign():
    x = {"input_ids": np.arange(10), "attention_mask": np.ones(10)}
    trial = perturb_input(x, np.random.default_rng(1), task_family="token_classification")
    assert trial.status == "unsupported" and not trial.changed
    np.testing.assert_array_equal(trial.sample["input_ids"], x["input_ids"])
    assert perturb_input("same  text", np.random.default_rng(1), task_family="classification").status == "unsupported"
    cfg = {"evaluation_text_transformations": [{"source": "cat", "replacement": "feline", "validation_id": "review-5", "meaning_preserving": True}]}
    changed = perturb_input("a cat sat", np.random.default_rng(1), task_family="classification", config=cfg)
    assert changed.sample == "a feline sat" and changed.changed
    assert changed.metadata["validation_id"] == "review-5"
    unchanged = perturb_input("a cathedral", np.random.default_rng(1), task_family="classification", config=cfg)
    assert unchanged.status == "excluded"
