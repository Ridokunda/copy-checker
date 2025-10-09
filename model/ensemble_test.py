import argparse
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

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


def normalize_weights(ws):
    ws = np.array(ws, dtype=float)
    s = ws.sum()
    if s == 0:
        return np.ones_like(ws) / len(ws)
    return ws / s


def evaluate_weights(ws):
    w = normalize_weights(ws)
    probs = w[0]*svm_probs + w[1]*xgb_probs + w[2]*rf_probs + w[3]*lr_probs
    preds = (probs > 0.5).astype(int)
    acc = (preds == y).mean()
    try:
        auc = roc_auc_score(y, probs)
    except Exception:
        auc = float('nan')
    return acc, auc, preds, probs, w


def grid_search_weights(step=0.1):
    # enumerate weights that sum to 1 with given step
    steps = int(round(1.0 / step))
    best = {'acc': -1, 'auc': -1, 'weights': None, 'preds': None, 'probs': None}
    for a in range(0, steps+1):
        for b in range(0, steps+1-a):
            for c in range(0, steps+1-a-b):
                d = steps - a - b - c
                ws = np.array([a, b, c, d], dtype=float) * step
                acc, auc, preds, probs, w = evaluate_weights(ws)
                if acc > best['acc']:
                    best.update({'acc': acc, 'auc': auc, 'weights': w, 'preds': preds, 'probs': probs})
    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ensemble weighted search')
    parser.add_argument('--grid-step', type=float, default=0.1, help='Grid step for weights (default 0.1)')
    args = parser.parse_args()

    print(f'Running grid search with step={args.grid_step}...')
    best = grid_search_weights(step=args.grid_step)
    print(f"Best acc: {best['acc']:.4f}, AUC: {best['auc']:.4f}")
    print('Best weights (SVM, XGB, RF, LR):', best['weights'].tolist())
    print('Classification Report for best weights:')
    print(classification_report(y, best['preds']))
    print('Confusion Matrix:')
    print(confusion_matrix(y, best['preds']))
