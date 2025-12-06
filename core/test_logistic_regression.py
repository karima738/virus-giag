from sklearn.linear_model import LogisticRegression
from core.model import Model
from core.loss import binary_cross_entropy_loss


class LogisticRegressionModel(Model):
    def __init__(self):
        self.model = LogisticRegression()
        self.loss = None

    def train(self, X, y):
        self.model.fit(X, y)
        y_pred = self.model.predict_proba(X)[:, 1]
        self.loss = binary_cross_entropy_loss(y, y_pred)
        print(f"✅ Modèle entraîné avec succès. Loss = {self.loss:.4f}")

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


class LogisticRegressionGD:
    pass
