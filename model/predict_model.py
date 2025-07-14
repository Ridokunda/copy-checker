import sys
import json
import joblib
import numpy as np
from collections import Counter

# Load the best model and feature names
best_model = joblib.load("best_model.pkl")
with open("feature_keys.json", "r") as f:
    feature_names = json.load(f)["similarity_keys"]

def predict_single(tree, row):
    """For tree-based models only (if best_model is Random Forest)."""
    while isinstance(tree, dict):
        if row[tree["index"]] < tree["value"]:
            tree = tree["left"]
        else:
            tree = tree["right"]
    return tree

def forest_predict(trees, row):
    """Only needed if best_model is Random Forest."""
    votes = [predict_single(t, row) for t in trees]
    counts = Counter(votes)
    prob0 = counts.get(0, 0) / len(votes)
    prob1 = counts.get(1, 0) / len(votes)
    return 1 if prob1 >= prob0 else 0, [prob0, prob1]

def main():
    try:
        payload = json.loads(sys.stdin.read())
        features = payload["features"]
        
        # Generic prediction (works for any sklearn model)
        prediction = best_model.predict([features])[0]
        confidence = best_model.predict_proba([features])[0]
        
        print(json.dumps({
            "prediction": int(prediction),
            "confidence": [float(confidence[0]), float(confidence[1])],
            "model_type": str(type(best_model).__name__)
        }))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()