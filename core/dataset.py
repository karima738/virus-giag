import pandas as pd

class PatientDataset:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def get_features_and_labels(self):
        X = self.data[['temperature', 'toux']]
        y = self.data['Infection']
        return X, y
