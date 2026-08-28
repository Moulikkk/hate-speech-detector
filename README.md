```markdown
# Hate Speech Detector

A machine learning based web application that classifies text as **Hate Speech**, **Offensive Language**, or **Normal/Safe Speech**, built using TF-IDF vectorization and Logistic Regression, with an interactive Streamlit interface for real-time predictions.

## Overview

This project addresses the challenge of automated content moderation on social media by classifying user-submitted text into one of three categories. Special attention was given to handling severe class imbalance in the training data (Hate Speech makes up only ~6% of the dataset), using balanced class weighting to improve detection of the underrepresented category.

## Features

- Real-time text classification via an interactive Streamlit interface
- TF-IDF vectorization with stopword removal and bigram features
- Logistic Regression classifier with balanced class weighting
- Model evaluation via classification report and confusion matrix

## Tech Stack

- Python
- Scikit-learn (TF-IDF, Logistic Regression)
- Streamlit
- Pandas / NumPy
- Matplotlib

## Dataset

Uses `labeled_data.csv`, containing labeled social media posts across three categories:

| Class Label | Description |
|---|---|
| 0 | Hate Speech |
| 1 | Offensive Language |
| 2 | Normal / Safe Speech |

## Setup & Installation

1. Clone the repository:
```
git clone https://github.com/YOUR_USERNAME/hate-speech-detector.git
cd hate-speech-detector
```

2. Create and activate a virtual environment:
```
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```
pip install streamlit pandas scikit-learn matplotlib
```

## Usage

Run the app:
```
streamlit run app1.py
```

This opens the app in your browser at `http://localhost:8501`. Enter any text in the input box and click **Predict** to see the classification.

## Evaluation

To evaluate model performance and generate a confusion matrix:
```
python eval.py
```

This outputs a classification report to the terminal and saves `ml_confusion_matrix.png` and `model_summary.txt`.

## Results

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Hate Speech | 0.29 | 0.60 | 0.40 |
| Offensive | 0.97 | 0.85 | 0.90 |
| Normal | 0.76 | 0.92 | 0.83 |

**Overall Accuracy:** 85%

Balanced class weighting significantly improved recall on the Hate Speech class, though at the cost of precision — reflecting the inherent difficulty of learning from a small, underrepresented set of examples.

## Project Structure

```
hate_speech_project/
├── app1.py                  # Streamlit app
├── eval.py                  # Model evaluation script
├── labeled_data.csv         # Training dataset
├── ml_confusion_matrix.png  # Confusion matrix output
└── README.md
```

## Future Improvements

- Explore transformer-based models (e.g. BERT) for improved contextual understanding
- Add confidence scores to predictions in the UI
- Experiment with additional classifiers (SVM, Random Forest) for comparison
- Expand to multilingual datasets

## Author

Moulik Satija
```