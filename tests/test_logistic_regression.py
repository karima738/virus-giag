import numpy as np
from core.logistic_regression import LogisticRegressionModel


def test_logistic_regression_train():
    """Test que le modèle s'entraîne correctement et calcule une loss valide"""
    # Données de test simples
    X = np.random.rand(20, 3)
    y = np.random.randint(0, 2, 20)

    # Créer et entraîner le modèle
    model = LogisticRegressionModel()
    model.train(X, y)

    # Vérifications
    assert model.loss is not None, "La loss ne doit pas être None après l'entraînement"
    assert model.loss >= 0, "La loss doit être positive ou nulle"
    assert model.model is not None, "Le modèle sklearn doit être initialisé"


def test_logistic_regression_predict():
    """Test que le modèle produit des prédictions valides"""
    # Données de test
    X = np.array([[0.1, 0.2, 0.3],
                  [0.5, 0.6, 0.7],
                  [0.2, 0.4, 0.6],
                  [0.8, 0.9, 0.1]])
    y = np.array([0, 1, 0, 1])

    # Entraîner le modèle
    model = LogisticRegressionModel()
    model.train(X, y)

    # Faire des prédictions
    preds = model.predict(X)

    # Vérifications
    assert preds.shape == (4,), "Les prédictions doivent avoir la bonne forme"
    assert len(preds) == len(X), "Nombre de prédictions doit égaler nombre d'exemples"
    assert np.all(preds >= 0) and np.all(preds <= 1), "Les probabilités doivent être entre 0 et 1"