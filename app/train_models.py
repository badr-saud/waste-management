import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


def train_take_keep_classifier():
    """
    Train a binary classifier for the Take/Keep decision based on sensor data.
    """
    # Set random seed for reproducibility
    np.random.seed(1)

    # Simulate sensor data
    n_samples = 300
    gas_value = np.random.randint(
        100, 1025, size=n_samples
    )  # Gas value between 100 and 1024
    distance = np.random.uniform(
        0, 28, size=n_samples
    )  # Distance between 1.0 and 27.999..
    weight = 0 + 40 * np.random.random(size=n_samples)  # Weight between 10 and 50

    # Initialize labels
    labels = []

    # Apply classification logic
    for i in range(n_samples):
        if gas_value[i] > 400:
            labels.append("take")
        elif distance[i] < 6:
            if weight[i] > 48:
                labels.append("take")
            elif weight[i] > 25:
                labels.append("take")
            else:
                labels.append("keep")
        elif distance[i] <= 10 and weight[i] > 25:
            labels.append("take")
        else:
            labels.append("keep")

    # Convert labels to binary
    binary_labels = np.array([1 if label == "take" else 0 for label in labels])

    # Create feature matrix
    data = np.column_stack((gas_value, distance, weight))

    # Normalize data
    scaler = StandardScaler()
    data_norm = scaler.fit_transform(data)

    # Perform PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(data_norm)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features_pca, binary_labels, test_size=0.3, random_state=42
    )

    # Train SVM classifier
    model = SVC(kernel="rbf", probability=True)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Binary Classification Accuracy (Take vs Keep): {accuracy*100:.2f}%")

    # Save model and PCA parameters
    model_data = {
        "model": model,
        "pca": pca,
        "scaler": scaler,
        "feature_names": ["gas_value", "distance", "weight"],
    }

    with open("take_keep_classifier.pkl", "wb") as f:
        pickle.dump(model_data, f)

    return model_data


def train_toxic_gas_classifier():
    """
    Train a binary classifier for toxic gas detection.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Simulate sensor data
    n_samples = 300
    labels = np.concatenate(
        [
            np.ones(n_samples // 3),
            2 * np.ones(n_samples // 3),
            3 * np.ones(n_samples // 3),
        ]
    )

    # Features: [Fill_Level, Weight, Gas_Concentration]
    normal_data = np.column_stack(
        [
            np.random.normal(20, 2, n_samples // 3),
            np.random.normal(10, 1, n_samples // 3),
            np.random.normal(5, 0.5, n_samples // 3),
        ]
    )

    full_data = np.column_stack(
        [
            np.random.normal(40, 2, n_samples // 3),
            np.random.normal(20, 1, n_samples // 3),
            np.random.normal(6, 0.5, n_samples // 3),
        ]
    )

    gas_data = np.column_stack(
        [
            np.random.normal(25, 2, n_samples // 3),
            np.random.normal(15, 1, n_samples // 3),
            np.random.normal(15, 0.5, n_samples // 3),
        ]
    )

    data = np.vstack([normal_data, full_data, gas_data])

    # Convert to binary labels (1 = Toxic Gas, 0 = Normal/Full)
    binary_labels = (labels == 3).astype(int)

    # Center the data and perform PCA
    mu = np.mean(data, axis=0)
    data_centered = data - mu

    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(data)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features_pca, binary_labels, test_size=0.3, random_state=42
    )

    # Train SVM classifier
    model = SVC(kernel="rbf", probability=True)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Binary Classification Accuracy (Toxic Gas Detection): {accuracy*100:.2f}%")

    # Save model and PCA parameters
    model_data = {
        "model": model,
        "pca": pca,
        "mu": mu,
        "feature_names": ["fill_level", "weight", "gas_concentration"],
    }

    with open("toxic_gas_model.pkl", "wb") as f:
        pickle.dump(model_data, f)

    return model_data


if __name__ == "__main__":
    print("Training Take/Keep Classifier...")
    take_keep_model = train_take_keep_classifier()

    print("\nTraining Toxic Gas Classifier...")
    toxic_gas_model = train_toxic_gas_classifier()

    print("\nTraining complete! Models saved to disk.")
