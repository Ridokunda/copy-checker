import sys
import json
import joblib
from collections import Counter

model = joblib.load("model.pkl")

def predict_single(tree, row):
    while isinstance(tree, dict):
        if row[tree["index"]] < tree["value"]:
            tree = tree["left"]
        else:
            tree = tree["right"]
    return tree  

def forest_predict(trees, row):
    votes = [predict_single(t, row) for t in trees]
    counts = Counter(votes)
    total = len(votes)
    prob0 = counts.get(0, 0) / total
    prob1 = counts.get(1, 0) / total
    prediction = 1 if prob1 >= prob0 else 0
    
    return prediction, [prob0, prob1]

def main():
    try:
        payload = json.loads(sys.stdin.read())
        features = payload["features"]

        pred, confidence = forest_predict(model, features)
        print(json.dumps({
            "prediction": pred, 
            "confidence": confidence
        }))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()