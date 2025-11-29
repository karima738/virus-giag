import numpy as np


def binary_cross_entropy_loss(y_true, y_pred):
    """
    Calcule la perte logistique (binary cross-entropy)
    """
    epsilon = 1e-10  # éviter log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) +
                    (1 - y_true) * np.log(1 - y_pred))
    return loss
