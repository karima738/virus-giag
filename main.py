import pandas as pd
from utils.preprocessing import preprocess_data
from utils.metrics import evaluate_model
from app.interface_clinique import ClinicalPredictor
from pipeline.trainer import train_model
from pipeline.evaluator import evaluate_model_with_loss

# Charger les données
dataset = pd.read_csv("data/patient_data.csv")
X = dataset[['temperature', 'toux']]
y = dataset['Infection']

# Prétraitement
X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

# Entraînement du modèle
model = train_model(X_train, y_train)

# Évaluation
y_pred_binary = (model.predict(X_test) >= 0.5).astype(int)
scores = evaluate_model(y_test, y_pred_binary)
print("📈 Évaluation du modèle :", scores)

# Évaluation avec la loss
evaluate_model_with_loss(model, X_test, y_test)

# Test clinique
predictor = ClinicalPredictor(model)
patient = {"temperature": 38.9, "toux": 1}
result = predictor.diagnose_patient(patient)
print("🩺 Diagnostic :", result)
