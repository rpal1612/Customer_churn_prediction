from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import numpy as np
import os
import pickle
import json
from model_trainer import (
    predict_churn, identify_risk_factors, generate_recommendations
)
from train_models import train_models_from_path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.figure import Figure

app = Flask(__name__)
app.secret_key = 'telecom_churn_prediction_secret_key'

# Global variables to store models and data
models = {}
label_encoders = {}
numerical_cols = []
categorical_cols = []
feature_names = []
model_comparison_data = {}

# Routes
@app.route('/')
def index():
    # Check if models are trained
    models_exist = os.path.exists('models/model_data.pkl')
    return render_template('index.html', models_exist=models_exist)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global models, label_encoders, numerical_cols, categorical_cols, feature_names
    
    # Load models if not already loaded
    if not models:
        try:
            load_saved_models()
        except:
            flash('Models not found. Please train models first using the backend script.', 'warning')
            return redirect(url_for('index'))
    
    if request.method == 'POST':
        try:
            # Get form data
            user_input = {}
            
            # Get selected model
            model_type = request.form.get('model_type')
            if model_type == 'rf':
                active_model = models['rf']
                model_name = "Bayesian Network"
            else:
                active_model = models['bn']
                model_name = "Markov Model"
            
            # Process numerical features
            for col in numerical_cols:
                user_input[col] = float(request.form.get(col))
            
            # Process categorical features
            for col in categorical_cols:
                # Get the selected option (which is the encoded value)
                user_input[col] = int(request.form.get(col))
            
            # Make prediction
            prediction, retention_probability = predict_churn(active_model, user_input, feature_names)
            
            # Calculate retention percentage
            retention_percentage = round(retention_probability * 100, 2)
            
            # Determine risk level
            if retention_percentage >= 70:
                risk_level = "LOW"
            elif retention_percentage >= 40:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            
            # Identify risk factors
            risk_factors = identify_risk_factors(user_input, feature_names, active_model)
            
            # Generate recommendations
            recommendations = generate_recommendations(user_input, risk_factors, retention_probability, model_name)
            
            # Store results in session
            session['prediction_results'] = {
                'prediction': int(prediction),
                'retention_probability': retention_percentage,
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'recommendations': recommendations,
                'model_name': model_name
            }
            
            return redirect(url_for('results'))
            
        except Exception as e:
            flash(f'Error making prediction: {str(e)}', 'danger')
    
    # Prepare data for the form
    form_data = {
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'label_encoders': {col: list(encoder.classes_) for col, encoder in label_encoders.items()}
    }
    
    return render_template('predict.html', form_data=form_data)

@app.route('/results')
def results():
    global model_comparison_data
    
    if 'prediction_results' not in session:
        flash('No prediction results found. Please make a prediction first.', 'warning')
        return redirect(url_for('predict'))
    
    results = session['prediction_results']
    
    # Load model comparison data if not already loaded
    if not model_comparison_data:
        try:
            load_saved_models()
        except Exception as e:
            flash(f'Error loading models: {str(e)}', 'danger')
            return redirect(url_for('index'))
    
    return render_template('results.html', results=results, model_data=model_comparison_data)

@app.template_global()
def enumerate(iterable, start=0):
    return __builtins__.enumerate(iterable, start)

@app.route('/compare')
def compare():
    global model_comparison_data
    
    # Load model comparison data if not already loaded
    if not model_comparison_data:
        try:
            load_saved_models()
        except:
            flash('Models not found. Please train models first using the backend script.', 'warning')
            return redirect(url_for('index'))
    
    # Generate comparison charts
    accuracy_chart = generate_comparison_chart('Accuracy')
    auc_chart = generate_comparison_chart('AUC')
    
    return render_template('compare.html', 
                      model_comparison_data=model_comparison_data,
                      model_data=model_comparison_data,
                      accuracy_chart=accuracy_chart,
                      auc_chart=auc_chart)

def generate_comparison_chart(metric_type):
    """Generate a base64 encoded image for the comparison chart"""
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    
    models = list(model_comparison_data.keys())
    model_names = [model_comparison_data[m]['name'] for m in models]
    
    if metric_type == 'Accuracy':
        values = [model_comparison_data[m]['accuracy'] for m in models]
    else:  # AUC
        values = [model_comparison_data[m]['auc'] for m in models]
    
    colors = ['#6366F1', '#EC4899']  # Indigo and Pink
    ax.bar(model_names, values, color=colors)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(metric_type)
    ax.set_title(f'Model Comparison - {metric_type}')
    
    # Add value labels on top of bars
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
    buf.seek(0)
    
    # Encode the image to base64 string
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{img_str}'

def load_saved_models():
    """Load saved models and data from disk"""
    global models, label_encoders, numerical_cols, categorical_cols, feature_names, model_comparison_data
    
    # Load model data
    with open('models/model_data.pkl', 'rb') as f:
        data = pickle.load(f)
        label_encoders = data['label_encoders']
        numerical_cols = data['numerical_cols']
        categorical_cols = data['categorical_cols']
        feature_names = data['feature_names']
        model_comparison_data = data['model_comparison_data']
    
    # Load models
    with open('models/rf_model.pkl', 'rb') as f:
        models['rf'] = pickle.load(f)
    
    with open('models/bn_model.pkl', 'rb') as f:
        models['bn'] = pickle.load(f)

if __name__ == '__main__':
    app.run(debug=True)