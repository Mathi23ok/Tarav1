import joblib
import numpy as np

# Load trained model
anomaly_model = joblib.load("anomaly_model.pkl")


def get_anomaly_score(embedding: np.ndarray):
    """
    Returns normalized anomaly score (0 = normal, 1 = highly anomalous)
    """

    embedding = embedding.reshape(1, -1)

    # IsolationForest decision_function:
    # Positive = normal
    # Negative = anomaly
    raw_score = anomaly_model.decision_function(embedding)[0]

    # Convert to anomaly probability
    # Lower raw_score => more anomalous
    anomaly_score = -raw_score

    # Normalize safely
    anomaly_score = max(0.0, min(anomaly_score, 1.0))

    return float(anomaly_score)
