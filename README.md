# Spam_Mail_Detection

A machine learning project for classifying emails as spam or not spam using a dataset of word frequencies. This repository contains code for data preprocessing, model training, evaluation, and deployment. A machine learning application to detect spam emails using Streamlit.

# Project Overview

This project uses supervised learning to build a spam email classifier. It processes a dataset of email content represented as word frequencies and applies various machine learning techniques for classification.

# Dataset
Dataset used for this project is from kaggle, you can find the dataset [here](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv)

# Key Features:

Data Preprocessing: Includes handling missing values, normalization, feature selection, and balancing.

Machine Learning Models: Logistic Regression

Performance Evaluation: Uses metrics such as accuracy, precision, recall, and F1-score.

Deployment Ready: Can be easily integrated into web applications.

# Results

Model Used: Random Forest Classifier

Accuracy: 97% on test data.

Evaluation Metrics: Precision, Recall, F1-Score, and Confusion Matrix.

## Setup

1. Clone the repository
```bash
git clone https://github.com/Sarthak-77/Spam_Mail_Detection.git
cd Spam_Mail_Detection
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
streamlit run app.py
```

## Project Structure
- `app.py`: Main Streamlit application
- `Spam_Detection.ipynb`: Model development notebook
- `model.pickle`: Trained model file

## Technologies Used
- Python
- Streamlit
- scikit-learn
- NLTK
- Pandas
- NumPy

## Deployment
This application is deployed on Render. Visit [link] to try it out.
