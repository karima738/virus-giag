from core.logistic_regression import LogisticRegressionModel


def train_model(X_train, y_train):
    model = LogisticRegressionModel()
    model.train(X_train, y_train)
    return model
