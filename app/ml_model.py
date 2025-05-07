# app/ml_model.py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

class SensorModel:
    def __init__(self):
        self.take_keep_model = None
        self.toxic_gas_model = None
        self.pca = None
        self.mu = None
        self.train_models()

    def train_models(self):
        n = 300
        rng = np.random.default_rng(1)

        # ----- Take/Keep Model -----
        gas = rng.integers(100, 1025, size=n)
        distance = rng.integers(1, 28, size=n)
        weight = 10 + 40 * rng.random(n)

        labels = []
        for g, d, w in zip(gas, distance, weight):
            if g > 400 or (d < 6 and w > 25) or (d <= 10 and w > 25):
                labels.append(1)
            else:
                labels.append(0)

        data1 = np.stack([gas, distance, weight], axis=1)
        labels1 = np.array(labels)

        data1_norm = (data1 - np.mean(data1, axis=0)) / np.std(data1, axis=0)
        pca1 = PCA(n_components=2)
        pca_data1 = pca1.fit_transform(data1_norm)

        X1_train, _, y1_train, _ = train_test_split(pca_data1, labels1, test_size=0.3, stratify=labels1)
        self.take_keep_model = SVC(kernel='rbf').fit(X1_train, y1_train)

        # ----- Toxic Gas Model -----
        labels2 = np.concatenate([np.zeros(n//3), np.zeros(n//3), np.ones(n//3)])
        data2 = np.vstack([
            np.random.randn(n//3, 3) * [2,1,0.5] + [20,10,5],
            np.random.randn(n//3, 3) * [2,1,0.5] + [40,20,6],
            np.random.randn(n//3, 3) * [2,1,0.5] + [25,15,15]
        ])

        self.mu = np.mean(data2, axis=0)
        self.pca = PCA(n_components=2)
        data2_centered = data2 - self.mu
        pca_data2 = self.pca.fit_transform(data2_centered)

        X2_train, _, y2_train, _ = train_test_split(pca_data2, labels2, test_size=0.3, stratify=labels2)
        self.toxic_gas_model = SVC(kernel='rbf').fit(X2_train, y2_train)

    def predict(self, raw_vector):
        # Apply PCA transformation
        centered = np.array(raw_vector) - self.mu
        pca_input = self.pca.transform([centered])

        action_pred = self.take_keep_model.predict(pca_input)[0]
        gas_pred = self.toxic_gas_model.predict(pca_input)[0]

        action = "Take" if action_pred == 1 else "Keep"
        gas = "Toxic" if gas_pred == 1 else "Normal"

        return {
            "action": action,
            "gas": gas,
            "message": f"Action: {action}, Gas: {gas}"
        }
