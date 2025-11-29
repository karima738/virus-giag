from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_model(y_true, y_pred_binary):
    """
    Évalue le modèle avec plusieurs métriques.
    """
    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred_binary
        ),
        "precision": precision_score(
            y_true,
            y_pred_binary
        ),
        "recall": recall_score(
            y_true,
            y_pred_binary
        ),
        "f1_score": f1_score(
            y_true,
            y_pred_binary
        ),
    }
