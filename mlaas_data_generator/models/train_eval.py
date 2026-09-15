from __future__ import annotations
import inspect
import numpy as np

try:
    from sklearn.exceptions import NotFittedError
except Exception:
    class NotFittedError(Exception):
        """Fallback NotFittedError when sklearn is unavailable."""
        pass


def resolve_fit_kwargs(model, x, y, kwargs, *, legacy_kwargs=None):
    """Negotiate adapter arguments before training can mutate the model.

    Only an incompatible signature permits the caller's explicit legacy form.
    A TypeError raised inside fit is a training failure, never a retry signal.
    Opaque extension callables receive the requested arguments unchanged.
    """
    try:
        signature = inspect.signature(model.fit)
    except (TypeError, ValueError):
        return kwargs
    try:
        signature.bind(x, y, **kwargs)
    except TypeError:
        if legacy_kwargs is None:
            raise
        signature.bind(x, y, **legacy_kwargs)
        return legacy_kwargs
    return kwargs


def train_local_model(model, x, y, epochs=1, batch_size=32, lr=None):
    kwargs = {"epochs": epochs, "batch_size": batch_size, "verbose": 0}
    legacy_kwargs = {"epochs": epochs, "batch_size": batch_size}
    if lr is not None:
        legacy_kwargs["lr"] = lr
    kwargs = resolve_fit_kwargs(model, x, y, kwargs, legacy_kwargs=legacy_kwargs)
    model.fit(x, y, **kwargs)

    if hasattr(model, "get_weights"):
        try:
            weights = model.get_weights()
            if weights is not None and len(weights) > 0:
                return {f"layer_{i}": w for i, w in enumerate(weights)}
        except Exception:
            pass
    return None


def regression_metrics(y_true, y_pred):
    """Named, unweighted errors in the supplied target units (no broadcasting)."""
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    if truth.ndim == 1:
        truth = truth[:, None]
    if prediction.ndim == 1:
        prediction = prediction[:, None]
    if truth.shape != prediction.shape or not truth.size:
        raise ValueError("regression targets and predictions must have matching non-empty shapes")
    error = prediction - truth
    mse = float(np.mean(error ** 2))
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(error)))}


def evaluate_regression_model(model, x_test, y_test):
    """Return explicit loss/MSE/RMSE/MAE; never reinterpret a positional RMSE.

    Keras loss can include regularization, so it is kept separate from MSE.
    Legacy Keras checkpoints without all three named metrics use predictions.
    Other estimators use predictions directly, independent of evaluate order.
    """
    try:
        module = type(model).__module__.lower()
        keras_result = None
        if "keras" in module or "tensorflow" in module:
            keras_result = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
            if all(name in keras_result for name in ("loss", "mse", "rmse", "mae")):
                return {name: float(keras_result[name]) for name in ("loss", "mse", "rmse", "mae")}
        try:
            inspect.signature(model.predict).bind(x_test, verbose=0)
        except (TypeError, ValueError):
            prediction = model.predict(x_test)
        else:
            prediction = model.predict(x_test, verbose=0)
        metrics = regression_metrics(y_test, prediction)
        return {"loss": float(keras_result["loss"]) if keras_result is not None else metrics["mse"], **metrics}
    except NotFittedError:
        return dict.fromkeys(("loss", "mse", "rmse", "mae"), np.nan)


def evaluate_model(model, x_test, y_test, task_type="classification"):
    if task_type == "regression":
        metrics = evaluate_regression_model(model, x_test, y_test)
        return metrics["loss"], metrics["rmse"], metrics["mae"]
    if hasattr(model, "evaluate"):
        try:
            try:
                results = model.evaluate(x_test, y_test, verbose=0)
            except TypeError:
                results = model.evaluate(x_test, y_test)
        except NotFittedError:
            return np.nan, np.nan, np.nan

        if isinstance(results, (list, tuple)) and len(results) == 4:
            loss = float(results[0]) if results[0] is not None else np.nan
            primary = float(results[1]) if results[1] is not None else np.nan
            secondary = float(results[2]) if results[2] is not None else np.nan
            return loss, primary, secondary
        
        if isinstance(results, (list, tuple)):
            loss = float(results[0]) if results else np.nan
            primary = float(results[1]) if len(results) > 1 else loss
        else:
            loss = float(results)
            primary = float(results)

        try:
            try:
                y_pred = model.predict(x_test, verbose=0)
            except TypeError:
                y_pred = model.predict(x_test)
        except (NotFittedError, AttributeError):
            return loss, primary, np.nan
        except Exception:
            return loss, primary, np.nan

        y_pred = np.asarray(y_pred)
        y_pred_classes = y_pred if y_pred.ndim == 1 else np.argmax(y_pred, axis=1)

        f1 = _macro_f1(y_test, y_pred_classes)
        return loss, primary, f1

    from sklearn.metrics import accuracy_score, f1_score
    try:
        y_pred = model.predict(x_test)
    except NotFittedError:
        return np.nan, np.nan, np.nan

    y_pred = np.asarray(y_pred)
    y_hat = y_pred if y_pred.ndim == 1 else np.argmax(y_pred, axis=1)

    acc = float(accuracy_score(y_test, y_hat))
    f1m = float(f1_score(y_test, y_hat, average="macro", zero_division=0))

    return 1.0 - acc, acc, f1m


def _macro_f1(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    if labels.size == 0:
        return 0.0
    scores = []
    for lbl in labels:
        tp = np.sum((y_true == lbl) & (y_pred == lbl))
        fp = np.sum((y_true != lbl) & (y_pred == lbl))
        fn = np.sum((y_true == lbl) & (y_pred != lbl))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores.append(0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores))
