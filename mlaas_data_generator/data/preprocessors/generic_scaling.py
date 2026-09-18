import numpy as np

from ..scaling import apply_target_scaler
from .label_schema import attach_label_schema

def preprocess_tabular_scaling(train, test, meta, *, scaler="standard", y_standardize=True):
    (x_train, y_train) = train
    (x_test, y_test) = test

    meta = dict(meta)
    fitted = dict(meta.get("shared_fitted_state") or {})
    # Fit once on the first family's selected rows, then freeze this state.
    # Later clients and reconstruction transform only their own selected rows.
    if isinstance(x_train, np.ndarray) and x_train.ndim == 2:
        state = fitted.get("feature_scaler")
        if "feature_scaler" not in fitted:
            method = str(scaler or "none").lower()
            if method in {"standard", "minmax"}:
                from sklearn.preprocessing import StandardScaler, MinMaxScaler
                estimator = (StandardScaler() if method == "standard" else MinMaxScaler()).fit(x_train)
                attrs = ("mean_", "scale_", "var_", "n_features_in_", "n_samples_seen_") if method == "standard" else (
                    "min_", "scale_", "data_min_", "data_max_", "data_range_", "n_features_in_", "n_samples_seen_")
                state = {"type": method, "attributes": {k: np.asarray(getattr(estimator, k)).tolist() for k in attrs}}
            elif method not in {"none", ""}:
                raise ValueError(f"Unknown scaler: {scaler}")
            fitted["feature_scaler"] = state
        if state:
            from sklearn.preprocessing import StandardScaler, MinMaxScaler
            estimator = StandardScaler() if state["type"] == "standard" else MinMaxScaler()
            for key, value in state["attributes"].items():
                setattr(estimator, key, np.asarray(value) if isinstance(value, list) else value)
            x_train, x_test = estimator.transform(x_train).astype("float32"), estimator.transform(x_test).astype("float32")
        else:
            x_train, x_test = x_train.astype("float32"), x_test.astype("float32")
        meta["feature_scaler"] = state["type"] if state else None
    else:
        meta = dict(meta)
        meta["feature_scaler"] = None

    # Regression target scaling is optional
    if meta.get("task_type") == "regression":
        if "target_scaler" in fitted:
            y_scaler = fitted["target_scaler"]
            if y_scaler:
                y_train = ((y_train - y_scaler["mean"]) / y_scaler["std"]).astype("float32")
                y_test = ((y_test - y_scaler["mean"]) / y_scaler["std"]).astype("float32")
        else:
            y_train, y_test, y_scaler = apply_target_scaler(y_train, y_test, "standard" if y_standardize else None)
            fitted["target_scaler"] = y_scaler
        meta["target_scaler"] = y_scaler
    meta["shared_fitted_state"] = fitted
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
