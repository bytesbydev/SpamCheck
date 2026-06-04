# 📧 Spam Detector

A Machine Learning-based Spam Mail Detector that classifies SMS messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) techniques and a Logistic Regression model. The project demonstrates the complete machine learning pipeline, including data preprocessing, feature extraction, model training, evaluation, and prediction.

---

## 🚀 Features

- Detects spam and legitimate messages
- Text preprocessing and cleaning
- TF-IDF feature extraction
- Logistic Regression classification
- Model evaluation using Accuracy, Precision, Recall, and F1-Score
- Predicts unseen messages
- Modular and maintainable project structure

---

## 📂 Project Structure

```text
spam-detector/
│
├── data/
│   └── SMSSpamCollection
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── spam_detector.pkl
│
├── reports/
│   ├── metrics.txt
│   └── confusion_matrix.png
│
├── tests/
│   └── test_preprocessing.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── main.py
```

---

## 📊 Dataset

This project uses the **SMS Spam Collection Dataset**, a public dataset containing labeled SMS messages.

- **Ham** → Legitimate message
- **Spam** → Unwanted or promotional message

Example:

```text
ham    I'll call you later.
spam   Congratulations! You won a free prize.
```

---

## ⚙️ Workflow

```text
SMS Dataset
      ↓
Data Loading
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Train-Test Split
      ↓
Logistic Regression Model
      ↓
Model Training
      ↓
Performance Evaluation
      ↓
Spam/Ham Prediction
```

---

## 🧠 Machine Learning Pipeline

### 1. Data Loading
Load SMS messages and their corresponding labels from the dataset.

### 2. Text Preprocessing
- Convert text to lowercase
- Remove punctuation and special characters
- Remove stopwords
- Tokenize text

### 3. Feature Extraction
Convert textual data into numerical features using **TF-IDF Vectorization**.

### 4. Model Training
Train a **Logistic Regression** classifier using the processed dataset.

### 5. Model Evaluation
Evaluate performance using:
- Accuracy
- Precision
- Recall
- F1 Score

### 6. Prediction
Classify new messages as:
- Spam
- Ham

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- VS Code
- Git & GitHub

---

## 📈 Evaluation Metrics

The model is evaluated using:

- **Accuracy** – Overall correctness of predictions
- **Precision** – Percentage of predicted spam messages that are actually spam
- **Recall** – Percentage of actual spam messages correctly identified
- **F1 Score** – Balance between precision and recall

---

## 🎯 Learning Outcomes

Through this project, you will gain experience in:

- Data preprocessing
- Natural Language Processing (NLP)
- Feature extraction using TF-IDF
- Machine Learning classification
- Model evaluation
- Real-world spam detection systems
- Git and GitHub project management

---

## 🔮 Future Enhancements

- Web interface using Flask
- Real-time email classification
- Support for multiple machine learning models
- Model deployment on cloud platforms
- Advanced NLP techniques for improved accuracy

---

## 🏆 Author

Developed by **Bytesbydev** as a Machine Learning and NLP project focused on spam message classification.

---
⭐ If you found this project useful, consider giving it a star!