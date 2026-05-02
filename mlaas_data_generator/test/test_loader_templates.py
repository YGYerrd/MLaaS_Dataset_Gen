from mlaas_data_generator.data.preprocessors import hf as hf_preprocessors
from mlaas_data_generator.models.adapters.hf_adapter import build_task_spec, resolve_hf_task


def test_dataset_loader_templates_dispatch_to_expected_preprocessors(monkeypatch):
    calls = []

    def _stub(name):
        def _inner(train, test, meta, **kwargs):
            calls.append(name)
            meta = dict(meta)
            return ({"input_ids": [[1]], "attention_mask": [[1]]}, [0]), ({"input_ids": [[1]], "attention_mask": [[1]]}, [0]), meta
        return _inner

    monkeypatch.setattr(hf_preprocessors, "preprocess_hf_text_sequence", _stub("sequence"))
    monkeypatch.setattr(hf_preprocessors, "preprocess_hf_text_token", _stub("token"))
    monkeypatch.setattr(hf_preprocessors, "preprocess_hf_text_similarity", _stub("similarity"))
    monkeypatch.setattr(hf_preprocessors, "preprocess_hf_text_fill_mask", _stub("fill_mask"))
    monkeypatch.setattr(hf_preprocessors, "preprocess_hf_text_causal_lm_generation", _stub("causal_lm"))
    monkeypatch.setattr(hf_preprocessors, "preprocess_hf_text_seq2seq_generation", _stub("seq2seq"))

    base_meta = {"modality": "text", "hf_id": "dummy"}
    base_train = (["x"], None)
    base_test = (["y"], None)

    template_to_expected = {
        "hf_text_classification": ("sequence", "sequence_classification"),
        "hf_token_classification": ("token", "token_classification"),
        "hf_sentence_pair_classification": ("similarity", "sentence_similarity"),
        "hf_masked_lm": ("fill_mask", "fill_mask"),
        "hf_causal_lm": ("causal_lm", "causal_lm_generation"),
        "hf_seq2seq": ("seq2seq", "seq2seq_generation"),
    }

    for template, (expected_call, expected_task) in template_to_expected.items():
        calls.clear()
        _, _, meta = hf_preprocessors.preprocess_hf(
            base_train,
            base_test,
            dict(base_meta, loader_template=template),
            hf_model_id="dummy/model",
            loader_template=template,
            text_column="text",
            label_column="label",
        )
        assert calls == [expected_call]
        assert meta["hf_task"] == expected_task


def test_model_loader_templates_override_legacy_hf_task():
    assert resolve_hf_task(loader_template="hf_fill_mask", hf_task="sequence_classification") == "fill_mask"
    assert resolve_hf_task(loader_template="hf_seq2seq", hf_task="fill_mask") == "seq2seq_generation"

    task, spec = build_task_spec("sequence_classification", loader_template="hf_token_classification", num_labels=3)
    assert task == "token_classification"
    assert spec.name == "token_classification"

    task, spec = build_task_spec("fill_mask", loader_template="hf_sentence_similarity", num_labels=1, label_format="continuous")
    assert task == "sentence_similarity"
    assert spec.name == "sentence_similarity"



def test_aliases_normalize_identically_across_preprocess_and_model_paths():
    aliases = {
        "object_detection": "image_detection",
        "detection": "image_detection",
        "semantic_segmentation": "image_segmentation",
        "segmentation": "image_segmentation",
        "image_to_text": "image_captioning",
        "retrieval": "text_image_retrieval",
        "vqa": "visual_question_answering",
    }

    for alias, canonical in aliases.items():
        assert hf_preprocessors.normalize_hf_task(alias) == canonical
        assert resolve_hf_task(hf_task=alias) == canonical


