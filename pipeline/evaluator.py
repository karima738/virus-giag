from core.loss import binary_cross_entropy_loss

def evaluate_model_with_loss(model, X_test, y_test):
    y_pred = model.predict(X_test)
    loss = binary_cross_entropy_loss(y_test, y_pred)
    print(f"Binary Cross Entropy Loss sur le test set: {loss:.4f}")
    return loss
