import numpy as np


class ClinicalPredictor:
    def __init__(self, trained_model):
        self.model = trained_model

    def diagnose_patient(self, patient_data):
        """
        patient_data : dict, ex: {"temperature": 38.2, "toux": 1}
        """
        X = np.array([[patient_data["temperature"], patient_data["toux"]]])
        prediction = self.model.predict(X)[0]
        return "infecté" if prediction >= 0.5 else "sain"
