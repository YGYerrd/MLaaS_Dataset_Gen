import numpy as np

from ..scaling import apply_feature_scaler, apply_target_scaler
from .label_schema import attach_label_schema

def preprocess_tabular_scaling(train, test, meta, *, scaler="standard", y_standardize=True):
    (x_train, y_train) = train
    (x_test, y_test) = test

    # Only scale if features are 2D tabular arrays
    if isinstance(x_train, np.ndarray) and x_train.ndim == 2:
        x_train, x_test, scaler_used = apply_feature_scaler(x_train, x_test, scaler)
        meta = dict(meta)
        meta["feature_scaler"] = scaler_used
    else:
        meta = dict(meta)
        meta["feature_scaler"] = None

    # Regression target scaling is optional
    if meta.get("task_type") == "regression":
        y_train, y_test, y_scaler = apply_target_scaler(
            y_train, y_test, "standard" if y_standardize else None
        )
        meta["target_scaler"] = y_scaler
    meta = attach_label_schema(meta, y_train, default_num_labels=meta.get("num_classes"))
    return (x_train, y_train), (x_test, y_test), meta

import numpy as np

def preprocess_image_float01(train, test, meta):
    (x_train, y_train) = train
    (x_test, y_test) = test

    x_train = np.asarray(x_train).astype("float32") / 255.0
    x_test = np.asarray(x_test).astype("float32") / 255.0

    meta = dict(meta)
    meta["image_normalisation"] = "uint8_to_float32_0_1"
    meta = attach_label_schema(meta, y_train, default_num_labels=meta.get("num_classes"))
    return (x_train, y_train), (x_test, y_test), meta