import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import sys

def load_dataset(dataset_path):
    """Load the dataset from JSON file"""
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    return dataset
    

def preprocess_data(dataset):
    """Preprocess the dataset for training"""
    # Convert to DataFrame
    df = pd.DataFrame(dataset)
    
    # Separate features and labels
    X = np.array(df['features'].tolist())
    y = np.array(df['label'].tolist())
    
    return X, y

def train_xgboost_model(X, y, test_size=0.2, random_state=42):
    """Train XGBoost model with hyperparameter tuning"""
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
   
    
    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Initialize XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=random_state,
        eval_metric='logloss'
    )
    
    # Perform Grid Search with Cross-Validation
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='f1',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Plagiarized', 'Plagiarized'],
                yticklabels=['Not Plagiarized', 'Plagiarized'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    return best_model, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
    }

def save_model(model, model_path='xgboost_model.pkl'):
    """Save the trained model"""
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def load_model(model_path='xgboost_model.pkl'):
    """Load the trained model and scaler"""
    
    model = joblib.load(model_path)
    return model
    

def predict_plagiarism(model, features):
    """Predict plagiarism for given features"""
    # Ensure features is a 2D array
    if len(features.shape) == 1:
        features = features.reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    return {
        'prediction': int(prediction),
        'probability_not_plagiarized': float(probability[0]),
        'probability_plagiarized': float(probability[1]),
        'confidence': float(max(probability))
    }

def main():
    # Load dataset
    dataset = load_dataset('dataset.json')
    if dataset is None:
        return
    
    # Preprocess data
    X, y = preprocess_data(dataset)
    
    # Train model
    model, metrics = train_xgboost_model(X, y)
    
    # Save model
    save_model(model)
    
    # Save metrics to file
    with open('model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nTraining completed successfully!")
    print("Files saved:")
    print("- plagiarism_model.pkl (trained XGBoost model)")
    print("- scaler.pkl (feature scaler)")
    print("- model_metrics.json (evaluation metrics)")
    print("- confusion_matrix.png (confusion matrix plot)")
    print("- feature_importance.png (feature importance plot)")

if __name__ == "__main__":
    main()