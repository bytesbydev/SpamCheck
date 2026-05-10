import nltk
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessing import preprocess_text
from src.feature_extraction import (
    build_vectorizer,
    fit_vectorizer,
    transform_messages
)
from src.train import train_model, save_model
from src.evaluate import evaluate_model
from src.predict import predict_message


nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)


def main():

    # Load dataset
    df = load_data("data/SMSSpamCollection")

    # Preprocess messages
    df["message"] = df["message"].apply(preprocess_text)

    # Train-test split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    # TF-IDF features
    vectorizer = build_vectorizer()
    X_train = fit_vectorizer(vectorizer, X_train_text)
    X_test = transform_messages(vectorizer, X_test_text)

    # Train model
    model = train_model(X_train, y_train)

    # Save model
    save_model(
        model,
        vectorizer,
        "models/spam_detector.pkl"
    )

    # Evaluate model
    evaluate_model(
        model,
        X_test,
        y_test
    )

    # Sample predictions
    messages = [
        "Congratulations! You've won a FREE iPhone. Click here to claim now!",
        "Hey, are we still meeting for lunch tomorrow at 1pm?",
        "URGENT: Your account has been compromised. Verify immediately.",
        "Please find attached the report from yesterday's meeting."
    ]

    print("\nSample Predictions")
    print("-" * 25)

    for msg in messages:
        result = predict_message(
            msg,
            model=model,
            vectorizer=vectorizer
        )

        print(f"\nMessage    : {msg}")
        print(f"Prediction : {result['label'].upper()}")


if __name__ == "__main__":
    main()