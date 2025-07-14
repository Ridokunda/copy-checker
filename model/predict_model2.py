import sys
import json
import joblib
from collections import Counter

# Load trained Random‑Forest model (list of decision trees stored via joblib)
model = joblib.load("model.pkl")  # list[dict]
with open("feature_keys.json", "r") as f:
    feature_keys = json.load(f)
feature_names = feature_keys["similarity_keys"]

def predict_single(tree, row):
    """Traverse a single decision‑tree dictionary and return its class label."""
    while isinstance(tree, dict):
        if row[tree["index"]] < tree["value"]:
            tree = tree["left"]
        else:
            tree = tree["right"]
    return tree  


def forest_predict(trees, row):
    """Return majority vote and per‑class confidence from the forest."""
    votes = [predict_single(t, row) for t in trees]
    counts = Counter(votes)
    total = len(votes)
    # probabilities for classes 0 and 1 in fixed order
    prob0 = counts.get(0, 0) / total
    prob1 = counts.get(1, 0) / total
    prediction = 1 if prob1 >= prob0 else 0
    # Get top 3 influential features
    influential = []
    for tree in trees:
        node = tree
        while isinstance(node, dict):
            feat_name = feature_names[node["index"]]
            value = row[node["index"]]
            threshold = node["value"]
            influential.append((feat_name, value, threshold))
            node = node["left"] if value < threshold else node["right"]
    
    # Count most common decision features
    feat_counter = Counter([f[0] for f in influential])
    top_features = feat_counter.most_common(3)
    
    return prediction, [prob0, prob1], top_features


def main():
    try:
        payload = json.loads(sys.stdin.read())
        features = payload["features"]

        # Validate feature vector length matches expected
        expected_length = len(feature_names)
        if len(features) != expected_length:
            print(json.dumps({
                "error": f"Feature vector length mismatch. Expected {expected_length}, got {len(features)}",
                "expected_features": feature_names
            }))
            return

        pred, confidence, top_features = forest_predict(model, features)
        print(json.dumps({"prediction": pred, 
                          "confidence": confidence,
                          "top_features": top_features,
                          "feature_names": feature_names
                          }))
    except Exception as e:
        print(json.dumps({"error": str(e)}))


if __name__ == "__main__":
    main()
