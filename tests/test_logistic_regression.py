import numpy as np
from core.logistic_regression import LogisticRegressionModel


def test_logistic_regression_train():
    """Test que le modele s'entraine correctement"""
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, 20)

    model = LogisticRegressionModel()
    model.train(X, y)

    assert model.loss is not None
    assert model.loss >= 0
    assert model.model is not None


def test_logistic_regression_predict():
    """Test que le modele produit des predictions valides"""
    X = np.array([[0.1, 0.2, 0.3],
                  [0.5, 0.6, 0.7],
                  [0.2, 0.4, 0.6],
                  [0.8, 0.9, 0.1]])
    y = np.array([0, 1, 0, 1])

    model = LogisticRegressionModel()
    model.train(X, y)

    preds = model.predict(X)

    assert preds.shape == (4,)
    assert len(preds) == len(X)
    assert np.all(preds >= 0) and np.all(preds <= 1)
