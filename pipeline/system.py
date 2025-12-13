# pipeline/system.py

class VirusDiagnosisSystem:
    """
    Système final de diagnostic.
    Il contient le modèle, le préprocesseur et les métriques.
    """

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.metrics = {}

    def predict(self, X):
        """
        Effectue un diagnostic sur de nouvelles données.
        """
        if self.preprocessor:
            X = self.preprocessor.transform(X)

        return self.model.predict(X)
