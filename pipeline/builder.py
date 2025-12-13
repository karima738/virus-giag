# pipeline/builder.py

import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from pipeline.system import VirusDiagnosisSystem


class VirusModelBuilder:
    """
    Builder du système de diagnostic viral.
    Construit le pipeline étape par étape.
    """

    def __init__(self):
        self.system = VirusDiagnosisSystem()
        self.X = None
        self.y = None

    def load_data(self, path):
        data = pd.read_csv(path)
        self.X = data.drop("target", axis=1)
        self.y = data["target"]
        return self

    def preprocess(self):
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        self.system.preprocessor = scaler
        return self

    def train_model(self):
        model = LogisticRegression()
        model.fit(self.X, self.y)
        self.system.model = model
        return self

    def evaluate(self):
        predictions = self.system.model.predict(self.X)
        acc = accuracy_score(self.y, predictions)
        self.system.metrics["accuracy"] = acc
        return self

    def save(self, path="models/virus_model.pkl"):
        joblib.dump(self.system.model, path)
        return self

    def build(self):
        return self.system
