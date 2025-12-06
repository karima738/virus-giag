import numpy as np
from core.logistic_regression import LogisticRegressionModel

def test_logistic_regression_train():
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, 20)

    model = LogisticRegressionModel()
    model.train(X, y)

    assert model.loss is not None
    assert model.loss >= 0
