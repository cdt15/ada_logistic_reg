import os
import sys
import numpy as np

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

from ada_logistic_reg import predict_adaptive_logistic_regression


def test_predict_adaptive_logistic_regression():
    # causal direction: x0 --> x1 --> x3
    x0 = np.random.uniform(size=1000)
    x1 = 2.0 * x0 + np.random.uniform(size=1000)
    x2 = np.random.uniform(size=1000)
    x3 = 4.0 * x1 + np.random.uniform(size=1000)
    X = np.vstack([x0, x1, x2, x3]).T

    class_num = 4
    delims = np.percentile(X[:, 3], [int(100 / class_num * n) for n in range(1, class_num)])
    y_ = np.ones(1000)
    for delim in delims:
        y_ += (delim <= X[:, 3]).astype(int)
    X[:, 3] = y_

    try:
        result = predict_adaptive_logistic_regression(X, predictors=[0, 1, 2], target=3)
    except RuntimeError as e:
        if str(e) == "Rscript is not found." or str(e) == "glmnet is not installed.":
            return

    assert result.shape == (class_num, X.shape[1] - 1)
