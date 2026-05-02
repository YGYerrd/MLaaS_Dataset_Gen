import sys
import types

import numpy as np

from mlaas_data_generator.data.sources.huggingface import load_huggingface_source
from mlaas_data_generator.data.preprocessors.hf_multimodal import _encode_vqa_token_labels, preprocess_hf_multimodal


class DummyDS:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r.get(key) for r in self.rows]
        return self.rows[key]

    def select(self, idxs):
        return DummyDS([self.rows[i] for i in idxs])

    def train_test_split(self, test_size=0.2, seed=42, shuffle=True):
        n_test = max(1, int(len(self.rows) * test_size))
        return {"train": DummyDS(self.rows[:-n_test]), "test": DummyDS(self.rows[-n_test:])}


class BrokenRowDS(DummyDS):
    def __init__(self, rows, broken_indices):
        super().__init__(rows)
        self._broken_indices = set(broken_indices)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r.get(key) for r in self.rows]
        if key in self._broken_indices:
            raise FileNotFoundError(f"missing-row-{key}.jpg")
        return self.rows[key]


def test_vqa_token_label_encoding_masks_special_tokens():
    class DummyTokenizer:
        all_special_ids = [101, 102, 0]

        def __call__(self, text, **kwargs):
            return {
                "input_ids": [101, 200, 201, 102, 0],
                "attention_mask": [1, 1, 1, 1, 0],
                "special_tokens_mask": [1, 0, 0, 1, 1],
            }

    labels = _encode_vqa_token_labels(DummyTokenizer(), ["blue car"], max_length=5, ignore_index=-100)

    assert labels.tolist() == [[-100, 200, 201, -100, -100]]


class DecodingImageDS(DummyDS):
    def __init__(self, rows, *, decode=True):
        super().__init__(rows)
        self.features = {"image": types.SimpleNamespace(decode=decode)}
        self._decode = decode

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "image" and self._decode:
                raise FileNotFoundError("stale decoded image path")
            return [r.get(key) for r in self.rows]
        return self.rows[key]

    def cast_column(self, column, feature):
        if column != "image":
            return self
        return DecodingImageDS(self.rows, decode=getattr(feature, "decode", True))

    def select(self, idxs):
        return DecodingImageDS([self.rows[i] for i in idxs], decode=self._decode)


def test_hf_source_multimodal_pair_drop(monkeypatch):
    train_ds = DummyDS([
        {"image": np.zeros((8, 8, 3), dtype=np.uint8), "text": "a", "label": 0},
        {"image": None, "text": "b", "label": 1},
    ])
    test_ds = DummyDS([
        {"image": np.zeros((8, 8, 3), dtype=np.uint8), "text": "c", "label": 1},
    ])
    fake_mod = types.SimpleNamespace(
        load_dataset=lambda *args, **kwargs: train_ds if kwargs.get("split") == "train" else test_ds
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)

    (train, _), (test, _), meta = load_huggingface_source(
        dataset_name="dummy",
        modality="multimodal",
        image_column="image",
        text_column="text",
        label_column="label",
        missing_pair_handling="drop",
    )

    assert len(train) + len(test) == 2
    assert meta["schema"]["text_column"] == "text"
    assert meta["schema"]["pair_validation"]["missing_pair_handling"] == "drop"
    assert meta["accounting"]["raw_record_count"] == 2
    assert meta["accounting"]["post_filter_record_count"] == 1


def test_hf_source_multimodal_pair_check_uses_raw_image_paths(monkeypatch):
    train_ds = DecodingImageDS([
        {"image": {"path": "stale.jpg", "bytes": None}, "question": "q", "answers": "a"},
    ])
    test_ds = DecodingImageDS([
        {"image": {"path": "stale-test.jpg", "bytes": None}, "question": "q2", "answers": "a2"},
    ])
    fake_mod = types.SimpleNamespace(
        Image=lambda decode=True: types.SimpleNamespace(decode=decode),
        load_dataset=lambda *args, **kwargs: train_ds if kwargs.get("split") == "train" else test_ds,
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)

    (train, _), (test, _), meta = load_huggingface_source(
        dataset_name="dummy",
        modality="multimodal",
        hf_task="visual_question_answering",
        image_column="image",
        text_column="question",
        label_column="answers",
        missing_pair_handling="drop",
    )

    assert len(train) == 1
    assert len(test) == 1
    assert train["image"][0]["path"] == "stale.jpg"
    assert meta["schema"]["pair_validation"]["train"]["aligned_pairs"] == 1


def test_hf_source_multimodal_resolves_numbered_caption_column(monkeypatch):
    train_ds = DummyDS([
        {"image": np.zeros((8, 8, 3), dtype=np.uint8), "caption_0": "a cat"},
        {"image": np.zeros((8, 8, 3), dtype=np.uint8), "caption_0": "a dog"},
    ])
    test_ds = DummyDS([
        {"image": np.zeros((8, 8, 3), dtype=np.uint8), "caption_0": "a bird"},
    ])
    fake_mod = types.SimpleNamespace(
        load_dataset=lambda *args, **kwargs: train_ds if kwargs.get("split") == "train" else test_ds
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)

    (train, _), (test, _), meta = load_huggingface_source(
        dataset_name="jxie/flickr8k",
        modality="multimodal",
        hf_task="image_captioning",
        image_column="image",
        text_column="caption",
        label_column="caption",
        missing_pair_handling="drop",
    )

    assert len(train) == 2
    assert len(test) == 1
    assert meta["schema"]["text_column"] == "caption_0"
    assert meta["schema"]["label_column"] == "caption_0"


def test_hf_multimodal_preprocessor_contract(monkeypatch):
    train = DummyDS([
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "text": "hello", "label": 1},
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "text": "world", "label": 0},
    ])
    test = DummyDS([
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "text": "test", "label": 1},
    ])

    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3, 0], "attention_mask": [1, 1, 1, 0]}

    class DummyImageProcessor:
        def __call__(self, image, **kwargs):
            chw = np.transpose(np.asarray(image, dtype=np.float32), (2, 0, 1))
            return {"pixel_values": chw}

    fake_tr = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTokenizer()),
        AutoImageProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyImageProcessor()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tr)

    (x_train, y_train), (_, _), meta = preprocess_hf_multimodal(
        (train, None),
        (test, None),
        {"task_type": "classification"},
        hf_model_id="dummy/model",
        image_column="image",
        text_column="text",
        label_column="label",
        max_length=4,
    )

    assert set(x_train.keys()) == {"input_ids", "attention_mask", "pixel_values"}
    assert x_train["input_ids"].shape[0] == x_train["pixel_values"].shape[0] == len(y_train)
    assert meta["schema"]["batch_contract"]["combined_keys"] == ["input_ids", "attention_mask", "pixel_values"]
    assert meta["accounting"]["sequence_count"] == 2


def test_hf_multimodal_preprocessor_skips_decode_errors(monkeypatch):
    train = DummyDS([
        {"image": "missing-file.jpg", "text": "bad", "label": 0},
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "text": "good", "label": 1},
    ])
    test = DummyDS([
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "text": "test", "label": 1},
    ])

    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 0], "attention_mask": [1, 1, 0]}

    class DummyImageProcessor:
        def __call__(self, image, **kwargs):
            if isinstance(image, str):
                raise FileNotFoundError(image)
            chw = np.transpose(np.asarray(image, dtype=np.float32), (2, 0, 1))
            return {"pixel_values": chw}

    fake_tr = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTokenizer()),
        AutoImageProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyImageProcessor()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tr)

    (x_train, y_train), (_, _), meta = preprocess_hf_multimodal(
        (train, None),
        (test, None),
        {"task_type": "classification"},
        hf_model_id="dummy/model",
        image_column="image",
        text_column="text",
        label_column="label",
        max_length=4,
        on_decode_error="skip",
        report_decode_errors=True,
    )

    assert x_train["pixel_values"].shape[0] == 1
    assert y_train.tolist() == [1]
    assert meta["schema"]["decode_report"]["train"]["failed"] == 1
    assert meta["schema"]["decode_report"]["train"]["survived"] == 1


def test_hf_multimodal_preprocessor_relocates_moved_hf_cache_paths(monkeypatch, tmp_path):
    from PIL import Image

    cache_root = tmp_path / "hf-cache"
    relative_image = (
        "downloads",
        "extracted",
        "abc123",
        "train2014",
        "COCO_train2014_000000192867.jpg",
    )
    relocated = cache_root.joinpath(*relative_image)
    relocated.parent.mkdir(parents=True)
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(relocated)
    stale = tmp_path.joinpath("old-home", ".cache", "huggingface", "datasets", *relative_image)

    train = DummyDS([
        {"image": {"path": str(stale), "bytes": None}, "question": "what?", "answers": "blue"},
    ])
    test = DummyDS([
        {"image": {"path": str(stale), "bytes": None}, "question": "where?", "answers": "home"},
    ])

    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 0], "attention_mask": [1, 1, 0]}

    class DummyImageProcessor:
        def __call__(self, image, **kwargs):
            chw = np.transpose(np.asarray(image, dtype=np.float32), (2, 0, 1))
            return {"pixel_values": chw}

    fake_tr = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTokenizer()),
        AutoImageProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyImageProcessor()),
    )
    monkeypatch.setenv("HF_DATASETS_CACHE", str(cache_root))
    monkeypatch.setitem(sys.modules, "transformers", fake_tr)

    (x_train, y_train), (_, _), meta = preprocess_hf_multimodal(
        (train, None),
        (test, None),
        {},
        hf_model_id="dummy/model",
        hf_task="visual_question_answering",
        label_column="answers",
    )

    assert x_train["pixel_values"].shape == (1, 3, 4, 4)
    assert y_train.tolist() == ["blue"]
    assert meta["schema"]["decode_report"]["train"]["failed"] == 0


def test_hf_multimodal_preprocessor_skips_row_fetch_errors(monkeypatch):
    train = BrokenRowDS(
        [
            {"image": np.ones((8, 8, 3), dtype=np.uint8), "text": "bad", "label": 0},
            {"image": np.ones((8, 8, 3), dtype=np.uint8), "text": "good", "label": 1},
        ],
        broken_indices={0},
    )
    test = DummyDS([
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "text": "test", "label": 1},
    ])

    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 0], "attention_mask": [1, 1, 0]}

    class DummyImageProcessor:
        def __call__(self, image, **kwargs):
            chw = np.transpose(np.asarray(image, dtype=np.float32), (2, 0, 1))
            return {"pixel_values": chw}

    fake_tr = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTokenizer()),
        AutoImageProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyImageProcessor()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tr)

    (x_train, y_train), (_, _), meta = preprocess_hf_multimodal(
        (train, None),
        (test, None),
        {"task_type": "classification"},
        hf_model_id="dummy/model",
        image_column="image",
        text_column="text",
        label_column="label",
        max_length=4,
        on_decode_error="skip",
        report_decode_errors=True,
    )

    assert x_train["pixel_values"].shape[0] == 1
    assert y_train.tolist() == [1]
    assert meta["schema"]["decode_report"]["train"]["failed"] == 1
    assert meta["schema"]["pair_validation"]["train"]["decode_error_rows"] == 1


def test_hf_multimodal_preprocessor_squeezes_singleton_image_batches(monkeypatch):
    train = DummyDS([
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "text": "hello", "label": 1},
    ])
    test = DummyDS([
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "text": "test", "label": 1},
    ])

    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 0], "attention_mask": [1, 1, 0]}

    class DummyImageProcessor:
        def __call__(self, image, **kwargs):
            chw = np.transpose(np.asarray(image, dtype=np.float32), (2, 0, 1))
            return {"pixel_values": chw.reshape(1, 1, *chw.shape)}

    fake_tr = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTokenizer()),
        AutoImageProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyImageProcessor()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tr)

    (x_train, _), (_, _), meta = preprocess_hf_multimodal(
        (train, None),
        (test, None),
        {"task_type": "classification"},
        hf_model_id="dummy/model",
        image_column="image",
        text_column="text",
        label_column="label",
        max_length=4,
    )

    assert x_train["pixel_values"].shape == (1, 3, 8, 8)
    assert meta["input_shape"] == (3, 8, 8)



def test_preprocess_hf_dispatches_vqa_without_multimodal_metadata(monkeypatch):
    from mlaas_data_generator.data.preprocessors import hf as hf_preprocessors

    captured = {}

    def _stub(train, test, meta, **kwargs):
        captured["meta"] = dict(meta)
        captured["kwargs"] = dict(kwargs)
        return (
            {
                "input_ids": np.array([[1, 2]]),
                "attention_mask": np.array([[1, 1]]),
                "pixel_values": np.ones((1, 3, 2, 2), dtype=np.float32),
            },
            np.array([1]),
        ), (
            {
                "input_ids": np.array([[3, 4]]),
                "attention_mask": np.array([[1, 1]]),
                "pixel_values": np.ones((1, 3, 2, 2), dtype=np.float32),
            },
            np.array([0]),
        ), meta

    monkeypatch.setattr(hf_preprocessors, "preprocess_hf_multimodal", _stub)

    (_, _), (_, _), meta = hf_preprocessors.preprocess_hf(
        (DummyDS([{"image": np.ones((2, 2, 3), dtype=np.uint8), "question": "q", "answer": "a"}]), None),
        (DummyDS([{"image": np.ones((2, 2, 3), dtype=np.uint8), "question": "q2", "answer": "a2"}]), None),
        {"hf_task": "visual_question_answering"},
        hf_model_id="dummy/model",
    )

    assert meta["modality"] == "multimodal"
    assert meta["hf_task"] == "visual_question_answering"
    assert captured["kwargs"]["hf_task"] == "visual_question_answering"



def test_preprocess_hf_dispatches_retrieval_without_multimodal_metadata(monkeypatch):
    from mlaas_data_generator.data.preprocessors import hf as hf_preprocessors

    captured = {}

    def _stub(train, test, meta, **kwargs):
        captured["meta"] = dict(meta)
        captured["kwargs"] = dict(kwargs)
        return (
            {
                "input_ids": np.array([[1, 2]]),
                "attention_mask": np.array([[1, 1]]),
                "pixel_values": np.ones((1, 3, 2, 2), dtype=np.float32),
            },
            np.array([0]),
        ), (
            {
                "input_ids": np.array([[3, 4]]),
                "attention_mask": np.array([[1, 1]]),
                "pixel_values": np.ones((1, 3, 2, 2), dtype=np.float32),
            },
            np.array([0]),
        ), meta

    monkeypatch.setattr(hf_preprocessors, "preprocess_hf_multimodal", _stub)

    (_, _), (_, _), meta = hf_preprocessors.preprocess_hf(
        (DummyDS([{"image": np.ones((2, 2, 3), dtype=np.uint8), "text": "caption"}]), None),
        (DummyDS([{"image": np.ones((2, 2, 3), dtype=np.uint8), "text": "caption 2"}]), None),
        {"hf_task": "text_image_retrieval"},
        hf_model_id="dummy/model",
    )

    assert meta["modality"] == "multimodal"
    assert meta["hf_task"] == "text_image_retrieval"
    assert captured["kwargs"]["hf_task"] == "text_image_retrieval"



def test_hf_multimodal_vqa_defaults(monkeypatch):
    train = DummyDS([
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "question": "what?", "answer": "cat"},
    ])
    test = DummyDS([
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "question": "where?", "answer": "home"},
    ])

    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3, 0], "attention_mask": [1, 1, 1, 0]}

    class DummyImageProcessor:
        def __call__(self, image, **kwargs):
            chw = np.transpose(np.asarray(image, dtype=np.float32), (2, 0, 1))
            return {"pixel_values": chw}

    fake_tr = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTokenizer()),
        AutoImageProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyImageProcessor()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tr)

    (_, y_train), (_, y_test), meta = preprocess_hf_multimodal(
        (train, None),
        (test, None),
        {},
        hf_model_id="dummy/model",
        hf_task="visual_question_answering",
    )

    assert y_train.tolist() == ["cat"]
    assert y_test.tolist() == ["home"]
    assert meta["text_column"] == "question"
    assert meta["label_column"] == "answer"


def test_hf_multimodal_vqa_answers_and_variable_image_shapes(monkeypatch):
    train = DummyDS([
        {
            "image": np.ones((8, 6, 3), dtype=np.uint8),
            "question": "what?",
            "answers": {"answer": ["cat", "dog", "cat"]},
        },
        {
            "image": np.ones((5, 9, 3), dtype=np.uint8),
            "question": "where?",
            "answers": {"answer": ["home"]},
        },
    ])
    test = DummyDS([
        {
            "image": np.ones((7, 4, 3), dtype=np.uint8),
            "question": "what color?",
            "answers": {"answer": ["blue", "blue", "red"]},
        },
    ])

    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3, 0], "attention_mask": [1, 1, 1, 0]}

    class DummyImageProcessor:
        size = {"height": 4, "width": 4}

        def __call__(self, image, **kwargs):
            chw = np.transpose(np.asarray(image, dtype=np.float32), (2, 0, 1))
            return {"pixel_values": chw}

    fake_tr = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTokenizer()),
        AutoImageProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyImageProcessor()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tr)

    (x_train, y_train), (x_test, y_test), meta = preprocess_hf_multimodal(
        (train, None),
        (test, None),
        {},
        hf_model_id="dummy/model",
        hf_task="visual_question_answering",
        label_column="answers",
    )

    assert x_train["pixel_values"].shape == (2, 3, 4, 4)
    assert x_test["pixel_values"].shape == (1, 3, 4, 4)
    assert y_train.tolist() == ["cat", "home"]
    assert y_test.tolist() == ["blue"]
    assert meta["label_column"] == "answers"


def test_hf_multimodal_vqa_classification_labels_from_model_vocab(monkeypatch):
    train = DummyDS([
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "question": "what?", "answer": "cat"},
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "question": "where?", "answer": "home"},
    ])
    test = DummyDS([
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "question": "what?", "answer": "bird"},
    ])

    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 0], "attention_mask": [1, 1, 0]}

    class DummyImageProcessor:
        def __call__(self, image, **kwargs):
            chw = np.transpose(np.asarray(image, dtype=np.float32), (2, 0, 1))
            return {"pixel_values": chw}

    class DummyConfig:
        model_type = "vilt"
        label2id = {"cat": 0, "home": 1}
        id2label = {0: "cat", 1: "home"}

    fake_tr = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTokenizer()),
        AutoImageProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyImageProcessor()),
        AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyConfig()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tr)

    (_, y_train), (_, y_test), meta = preprocess_hf_multimodal(
        (train, None),
        (test, None),
        {},
        hf_model_id="dummy/vilt",
        hf_task="visual_question_answering",
        vqa_label_mode="classification",
    )

    assert y_train.tolist() == [0, 1]
    assert y_test.tolist() == [-100]
    assert meta["label_format"] == "vqa_class_index"
    assert meta["num_labels"] == 2
    assert meta["vqa_answer_vocab_source"] == "model_config"
    assert meta["vqa_test_unseen_answer_count"] == 1


def test_hf_multimodal_vqa_generation_token_labels(monkeypatch):
    train = DummyDS([
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "question": "what?", "answer": "cat"},
    ])
    test = DummyDS([
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "question": "where?", "answer": "home"},
    ])

    class DummyTokenizer:
        vocab_size = 10

        def __call__(self, text, **kwargs):
            if str(text) == "home":
                return {"input_ids": [3, 4, 0], "attention_mask": [1, 1, 0]}
            return {"input_ids": [1, 2, 0], "attention_mask": [1, 1, 0]}

    class DummyImageProcessor:
        def __call__(self, image, **kwargs):
            chw = np.transpose(np.asarray(image, dtype=np.float32), (2, 0, 1))
            return {"pixel_values": chw}

    class DummyConfig:
        model_type = "blip"

    fake_tr = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTokenizer()),
        AutoImageProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyImageProcessor()),
        AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyConfig()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tr)

    (_, y_train), (_, y_test), meta = preprocess_hf_multimodal(
        (train, None),
        (test, None),
        {},
        hf_model_id="dummy/blip",
        hf_task="visual_question_answering",
        vqa_label_mode="generation",
    )

    assert y_train.tolist() == [[1, 2, -100]]
    assert y_test.tolist() == [[3, 4, -100]]
    assert meta["label_format"] == "vqa_token_index"
    assert meta["vqa_label_mode"] == "generation"
    assert meta["num_labels"] == 10


def test_hf_multimodal_clamps_text_length_to_model_limit(monkeypatch):
    train = DummyDS([
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "question": "what?", "answer": "cat"},
    ])
    test = DummyDS([
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "question": "where?", "answer": "home"},
    ])

    class DummyTokenizer:
        model_max_length = 40

        def __call__(self, text, **kwargs):
            max_length = int(kwargs["max_length"])
            return {
                "input_ids": list(range(max_length)),
                "attention_mask": [1] * max_length,
            }

    class DummyImageProcessor:
        def __call__(self, image, **kwargs):
            chw = np.transpose(np.asarray(image, dtype=np.float32), (2, 0, 1))
            return {"pixel_values": chw}

    class DummyConfig:
        max_position_embeddings = 40

    fake_tr = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTokenizer()),
        AutoImageProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyImageProcessor()),
        AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyConfig()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tr)

    (x_train, _), (_, _), meta = preprocess_hf_multimodal(
        (train, None),
        (test, None),
        {},
        hf_model_id="dummy/vilt",
        hf_task="visual_question_answering",
        max_length=48,
    )

    assert x_train["input_ids"].shape == (1, 40)
    assert x_train["attention_mask"].shape == (1, 40)
    assert meta["max_length"] == 40
    assert meta["requested_max_length"] == 48
    assert meta["model_text_max_length"] == 40
    assert meta["max_length_adjusted"] is True


def test_hf_multimodal_caption_fallback_and_image_dict_payload(monkeypatch):
    train = DummyDS([
        {"image": {"array": np.ones((8, 8, 3), dtype=np.uint8)}, "caption": "a cat"},
    ])
    test = DummyDS([
        {"image": {"array": np.ones((8, 8, 3), dtype=np.uint8)}, "caption": "a dog"},
    ])

    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 0], "attention_mask": [1, 1, 0]}

    class DummyImageProcessor:
        def __call__(self, image, **kwargs):
            chw = np.transpose(np.asarray(image, dtype=np.float32), (2, 0, 1))
            return {"pixel_values": chw}

    fake_tr = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTokenizer()),
        AutoImageProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyImageProcessor()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tr)

    (x_train, _), (x_test, _), meta = preprocess_hf_multimodal(
        (train, None),
        (test, None),
        {},
        hf_model_id="dummy/model",
        hf_task="image_captioning",
        text_column="text",
    )

    assert x_train["pixel_values"].shape[0] == 1
    assert x_test["pixel_values"].shape[0] == 1
    assert meta["text_column"] == "caption"


def test_hf_multimodal_numbered_caption_fallback_and_token_labels(monkeypatch):
    train = DummyDS([
        {"image": {"array": np.ones((8, 8, 3), dtype=np.uint8)}, "caption_0": "a cat"},
    ])
    test = DummyDS([
        {"image": {"array": np.ones((8, 8, 3), dtype=np.uint8)}, "caption_0": "a dog"},
    ])

    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 0], "attention_mask": [1, 1, 0]}

    class DummyImageProcessor:
        def __call__(self, image, **kwargs):
            chw = np.transpose(np.asarray(image, dtype=np.float32), (2, 0, 1))
            return {"pixel_values": chw}

    fake_tr = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTokenizer()),
        AutoImageProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyImageProcessor()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tr)

    (_, y_train), (_, y_test), meta = preprocess_hf_multimodal(
        (train, None),
        (test, None),
        {},
        hf_model_id="dummy/model",
        hf_task="image_captioning",
        text_column="caption",
        label_column="caption",
    )

    assert meta["text_column"] == "caption_0"
    assert meta["label_column"] == "caption_0"
    assert y_train.tolist() == [[1, 2, -100]]
    assert y_test.tolist() == [[1, 2, -100]]


def test_hf_multimodal_retrieval_ignores_caption_label_column(monkeypatch):
    train = DummyDS([
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "caption_0": "a cat"},
    ])
    test = DummyDS([
        {"image": np.ones((8, 8, 3), dtype=np.uint8), "caption_0": "a dog"},
    ])

    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 0], "attention_mask": [1, 1, 0]}

    class DummyImageProcessor:
        def __call__(self, image, **kwargs):
            chw = np.transpose(np.asarray(image, dtype=np.float32), (2, 0, 1))
            return {"pixel_values": chw}

    fake_tr = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyTokenizer()),
        AutoImageProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: DummyImageProcessor()),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tr)

    (x_train, y_train), (x_test, y_test), meta = preprocess_hf_multimodal(
        (train, None),
        (test, None),
        {},
        hf_model_id="dummy/model",
        hf_task="text_image_retrieval",
        text_column="caption",
        label_column="caption",
    )

    assert meta["text_column"] == "caption_0"
    assert meta["label_column"] is None
    assert y_train.tolist() == [0]
    assert y_test.tolist() == [0]
    assert x_train["caption_lengths"].tolist() == [2]
    assert x_test["image_sizes"].tolist() == [[8, 8]]
