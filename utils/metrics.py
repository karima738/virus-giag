from sklearn import metrics



def evaluate_model(y_true, y_pred_binary):
    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred_binary),
        "precision": metrics.precision_score(y_true, y_pred_binary),
        "recall": metrics.recall_score(y_true, y_pred_binary),
        "f1_score": metrics.f1_score(y_true, y_pred_binary),
    }

