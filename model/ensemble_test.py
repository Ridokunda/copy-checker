import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from tensorflow.keras.models import load_model

# Load dataset
with open('dataset.json', 'r') as f:
    data = json.load(f)

X = np.array([item['features'] for item in data])
y = np.array([item['label'] for item in data])

# Load models
svm = joblib.load('svm_model.pkl')
xgb = joblib.load('xgboost_model.pkl')
rf = joblib.load('model.pkl') 
lr = joblib.load('logistic_regression_model.pkl') 

# Get predictions (probabilities)
svm_probs = svm.predict_proba(X)[:, 1]
xgb_probs = xgb.predict_proba(X)[:, 1]
def bagging_predict_proba(trees, row):
    from collections import Counter
    def predict_single(tree, row):
        while isinstance(tree, dict):
            if row[tree["index"]] < tree["value"]:
                tree = tree["left"]
            else:
                tree = tree["right"]
        return tree
    predictions = [predict_single(t, row) for t in trees]
    positive_votes = predictions.count(1)
    return positive_votes / len(trees)

rf_probs = np.array([bagging_predict_proba(rf, x) for x in X])
lr_probs = lr.predict_proba(X)[:, 1]

""" # Load neural network model
nn = load_model('plagiarism_nn.h5')
# NN model expects input shape (n_samples, n_features)
nn_probs = nn.predict(X, verbose=0).flatten() """


# Simple average ensemble 
ensemble_probs = (svm_probs + xgb_probs + rf_probs + lr_probs) / 4
ensemble_pred = (ensemble_probs > 0.5).astype(int)

print('Ensemble Classification Report:')
print(classification_report(y, ensemble_pred))
print('Ensemble Confusion Matrix:')
print(confusion_matrix(y, ensemble_pred))
