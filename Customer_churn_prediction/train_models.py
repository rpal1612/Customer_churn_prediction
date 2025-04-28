import pandas as pd
import numpy as np
import os
import pickle
from model_trainer import (
    load_and_explore_data, preprocess_data, build_and_evaluate_models
)

def train_models_from_path(dataset_path):
    """
    Train models from a dataset path and save them for later use.
    
    Args:
        dataset_path (str): Path to the CSV dataset file
    
    Returns:
        dict: Training results and model information
    """
    print(f"Loading dataset from: {dataset_path}")
    
    try:
        # Load and preprocess data
        df = load_and_explore_data(dataset_path)
        X_train, X_test, y_train, y_test, encoders, num_cols, cat_cols = preprocess_data(df)
        
        # Store for later use
        label_encoders = encoders
        numerical_cols = num_cols
        categorical_cols = cat_cols
        feature_names = list(X_train.columns)
        
        # Build and evaluate models
        results = build_and_evaluate_models(X_train, X_test, y_train, y_test, numerical_cols, categorical_cols)
        
        # Store models
        models = {
            'rf': results['rf']['model'],
            'bn': results['bn']['model']
        }
        
        # Store model comparison data
        model_comparison_data = {
            'rf': {
                'name': 'Bayesian Network',
                'accuracy': results['rf']['eval']['accuracy'],
                'auc': results['rf']['eval']['auc'],
                'report': results['rf']['eval']['classification_report'],
                'matrix': results['rf']['eval']['confusion_matrix'].tolist()
            },
            'bn': {
                'name': 'Markov Model',
                'accuracy': results['bn']['eval']['accuracy'],
                'auc': results['bn']['eval']['auc'],
                'report': results['bn']['eval']['classification_report'],
                'matrix': results['bn']['eval']['confusion_matrix'].tolist()
            }
        }
        
        # Save models and data for future use
        import os
        import pickle
        os.makedirs('models', exist_ok=True)
        with open('models/model_data.pkl', 'wb') as f:
            pickle.dump({
                'label_encoders': label_encoders,
                'numerical_cols': numerical_cols,
                'categorical_cols': categorical_cols,
                'feature_names': feature_names,
                'model_comparison_data': model_comparison_data
            }, f)
        
        with open('models/rf_model.pkl', 'wb') as f:
            pickle.dump(models['rf'], f)
        
        with open('models/bn_model.pkl', 'wb') as f:
            pickle.dump(models['bn'], f)
        
        print("Models trained and saved successfully!")
        
        return {
            'success': True,
            'models': models,
            'model_comparison_data': model_comparison_data,
            'feature_info': {
                'label_encoders': label_encoders,
                'numerical_cols': numerical_cols,
                'categorical_cols': categorical_cols,
                'feature_names': feature_names
            }
        }
        
    except Exception as e:
        print(f"Error training models: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    # Example usage: python train_models.py
    # You can modify this to accept command line arguments
    dataset_path = "uploads\customer_churn_dataset.csv"  # Replace with your dataset path
    result = train_models_from_path(dataset_path)
    
    if result['success']:
        print("Training completed successfully!")
        print(f"Bayesian Network Accuracy: {result['model_comparison_data']['rf']['accuracy']:.4f}")
        print(f"Markov Model Accuracy: {result['model_comparison_data']['bn']['accuracy']:.4f}")
    else:
        print(f"Training failed: {result['error']}")