from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer():
    return TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )


def fit_vectorizer(vectorizer, train_messages):
    return vectorizer.fit_transform(train_messages)


def transform_messages(vectorizer, messages):
    return vectorizer.transform(messages)