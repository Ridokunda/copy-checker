import sys
import json
import joblib
from collections import Counter

# Load trained Random‑Forest model (list of decision trees stored via joblib)
model = joblib.load("model.pkl")  # list[dict]


def predict_single(tree, row):
    """Traverse a single decision‑tree dictionary and return its class label."""
    while isinstance(tree, dict):
        if row[tree["index"]] < tree["value"]:
            tree = tree["left"]
        else:
            tree = tree["right"]
    return tree  # terminal label (0 or 1)


def forest_predict(trees, row):
    """Return majority vote and per‑class confidence from the forest."""
    votes = [predict_single(t, row) for t in trees]
    counts = Counter(votes)
    total = len(votes)
    # probabilities for classes 0 and 1 in fixed order
    prob0 = counts.get(0, 0) / total
    prob1 = counts.get(1, 0) / total
    prediction = 1 if prob1 >= prob0 else 0
    return prediction, [prob0, prob1]


def main():
    try:
        payload = json.loads(sys.stdin.read())
        features = payload["features"]
        pred, confidence = forest_predict(model, features)
        print(json.dumps({"prediction": pred, "confidence": confidence}))
    except Exception as e:
        print(json.dumps({"error": str(e)}))


if __name__ == "__main__":
    main()
