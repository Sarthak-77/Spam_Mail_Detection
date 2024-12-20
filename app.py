from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Set secret key from environment variable
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key')

# Load the model and vectorizer with better error handling
try:
    with open('spam_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully!")
    
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    print("Vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        email_text = request.form['email_text']
        email_transformed = vectorizer.transform([email_text])
        prediction = model.predict(email_transformed)[0]
        probability = model.predict_proba(email_transformed)[0][1]
        
        result = {
            'is_spam': bool(prediction),
            'probability': float(probability),
            'email_text': email_text
        }
        return render_template('result.html', result=result)
    except Exception as e:
        return render_template('error.html', error=str(e))

# Add port configuration
port = int(os.environ.get('PORT', 5000))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)