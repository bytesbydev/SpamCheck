from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label="spam")
    recall = recall_score(y_test, predictions, pos_label="spam")
    f1 = f1_score(y_test, predictions, pos_label="spam")

    print("\nModel Evaluation")
    print("-" * 30)
    print(f"Accuracy Score  : {accuracy:.4f}")
    print(f"Precision Score : {precision:.4f}")
    print(f"Recall Score    : {recall:.4f}")
    print(f"F1 Score        : {f1:.4f}")

    print("\nClassification Report")
    print("-" * 30)
    print(classification_report(y_test, predictions))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }