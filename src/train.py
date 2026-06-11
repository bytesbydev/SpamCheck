import os
import joblib
from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train):

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    return model


def save_model(model, vectorizer, model_path="models/spam_detector.pkl"):

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "vectorizer": vectorizer
        },
        model_path
    )


def load_model(model_path="models/spam_detector.pkl"):

    saved = joblib.load(model_path)

    return saved["model"], saved["vectorizer"]