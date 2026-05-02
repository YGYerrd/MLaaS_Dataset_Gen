from __future__ import annotations
import numpy as np

try:
    from sklearn.exceptions import NotFittedError
except Exception:
    class NotFittedError(Exception):
        """Fallback NotFittedError when sklearn is unavailable."""
        pass


def train_local_model(model, x, y, epochs=1, batch_size=32, lr=None):
    try:
        model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)
    except TypeError:
        if lr is None:
            model.fit(x, y, epochs=epochs, batch_size=batch_size)
        else:
            model.fit(x, y, epochs=epochs, batch_size=batch_size, lr=lr)

    if hasattr(model, "get_weights"):
        try:
            weights = model.get_weights()
            if weights is not None and len(weights) > 0:
                return {f"layer_{i}": w for i, w in enumerate(weights)}
        except Exception:
            pass
    return None


def evaluate_model(model, x_test, y_test, task_type="classification"):
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

        if task_type == "regression":
            return loss, primary, np.nan

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

    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
    try:
        y_pred = model.predict(x_test)
    except NotFittedError:
        return np.nan, np.nan, np.nan

    if task_type == "regression":
        y_pred = np.asarray(y_pred)
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        return rmse, rmse, np.nan
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
