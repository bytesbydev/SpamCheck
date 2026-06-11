from src.preprocessing import preprocess_text
from src.train import load_model


def predict_message(message, model=None, vectorizer=None):

    if model is None or vectorizer is None:
        model, vectorizer = load_model("models/spam_detector.pkl")

    message = preprocess_text(message)

    vector = vectorizer.transform([message])

    prediction = model.predict(vector)[0]
    probability = max(model.predict_proba(vector)[0])

    return {
        "label": prediction,
        "confidence": round(probability, 4)
    }