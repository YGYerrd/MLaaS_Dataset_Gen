import numpy as np

from mlaas_data_generator.data.preprocessors import hf_text_token


class FakeClassLabel:
    def __init__(self, names):
        self.names = names


class FakeSequence:
    def __init__(self, feature):
        self.feature = feature


class FakeDataset:
    column_names = ["tokens", "ner_tags"]

    def __init__(self, rows, label_feature):
        self._rows = rows
        self.features = {"ner_tags": label_feature}

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        return [row[key] for row in self._rows]


class FakeBatchEncoding(dict):
    def __init__(self, input_ids, attention_mask, word_id_rows):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)
        self._word_id_rows = word_id_rows

    def word_ids(self, batch_index=0):
        return self._word_id_rows[batch_index]


class FakeTokenizer:
    def __call__(
        self,
        tokens_list,
        *,
        is_split_into_words,
        truncation,
        padding,
        max_length,
        return_tensors,
    ):
        assert is_split_into_words is True
        assert truncation is True
        assert padding == "max_length"
        assert return_tensors == "np"

        input_ids = []
        attention_mask = []
        word_id_rows = []
        for tokens in tokens_list:
            word_ids = [None, *range(len(tokens)), None]
            ids = [101, *range(1, len(tokens) + 1), 102]
            pad = max_length - len(ids)
            input_ids.append(ids + [0] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
            word_id_rows.append(word_ids + [None] * pad)

        return FakeBatchEncoding(
            np.asarray(input_ids, dtype="int32"),
            np.asarray(attention_mask, dtype="int32"),
            word_id_rows,
        )


def test_extracts_token_label_names_from_sequence_class_label():
    label_feature = FakeSequence(FakeClassLabel(["O", "B-ORG"]))

    assert hf_text_token._extract_class_label_names(label_feature) == ["O", "B-ORG"]


def test_extracts_token_label_names_from_list_class_label():
    label_feature = [FakeClassLabel(["O", "B-ORG"])]

    assert hf_text_token._extract_class_label_names(label_feature) == ["O", "B-ORG"]


def test_token_preprocessor_accepts_list_class_label_features(monkeypatch):
    train_rows = [
        {"tokens": ["Hello", "Curtin"], "ner_tags": [0, 1]},
        {"tokens": ["World"], "ner_tags": [0]},
    ]
    test_rows = [{"tokens": ["Test"], "ner_tags": [0]}]

    monkeypatch.setattr(
        hf_text_token,
        "get_cached_tokenizer",
        lambda **kwargs: (FakeTokenizer(), None, None),
    )

    train, test, meta = hf_text_token.preprocess_hf_text_token(
        (FakeDataset(train_rows, [FakeClassLabel(["O", "B-ORG"])]), None),
        (FakeDataset(test_rows, [FakeClassLabel(["O", "B-ORG"])]), None),
        {"hf_id": "fake/token", "max_length": 6},
        hf_model_id="fake/model",
        tokens_column="tokens",
        label_column="ner_tags",
    )

    x_train, y_train = train
    x_test, y_test = test

    assert x_train["input_ids"].shape == (2, 6)
    assert x_test["attention_mask"].shape == (1, 6)
    assert y_train.tolist() == [
        [-100, 0, 1, -100, -100, -100],
        [-100, 0, -100, -100, -100, -100],
    ]
    assert y_test.tolist() == [[-100, 0, -100, -100, -100, -100]]
    assert meta["num_classes"] == 2
    assert meta["label_mapping"] == {"O": 0, "B-ORG": 1}
    assert meta["label_format"] == "token_index"
