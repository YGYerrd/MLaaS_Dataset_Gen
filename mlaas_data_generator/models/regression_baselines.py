"""Diagnostic controls fitted only on the evaluated client's training subset."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from .train_eval import regression_metrics


def regression_baselines(x_train, y_train, x_test, y_test, *, ridge_alpha=1.0):
    """Compare local mean, least squares, and ridge on the same held-out rows.

    These are diagnostic fits, not generated services or manifest members.
    Ridge alpha is fixed in advance; benchmark labels never select parameters.
    """
    x_train, x_test = np.asarray(x_train), np.asarray(x_test)
    y_train = np.asarray(y_train)
    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError("linear regression controls require tabular matrices")
    predictions = {"local_mean": np.broadcast_to(y_train.mean(axis=0), np.asarray(y_test).shape)}
    for name, estimator in (("linear", LinearRegression()), ("ridge", Ridge(alpha=ridge_alpha))):
        estimator.fit(x_train, y_train)
        predictions[name] = estimator.predict(x_test)
    return {"role": "diagnostic_only", "training_samples": len(y_train),
            "benchmark_samples": len(y_test), "ridge_alpha": ridge_alpha,
            "metrics": {name: regression_metrics(y_test, prediction)
                        for name, prediction in predictions.items()}}
