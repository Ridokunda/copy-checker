import sys
import json
import joblib

# --- Prediction Function ---
def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

def predict(node, row):
    if row[node["index"]] < node["value"]:
        return predict(node["left"], row) if isinstance(node["left"], dict) else node["left"]
    else:
        return predict(node["right"], row) if isinstance(node["right"], dict) else node["right"]

# --- Main Execution ---
def main():
    try:
        input_json = sys.stdin.read()
        payload = json.loads(input_json)
        features = payload["features"]

        model = joblib.load("model.pkl")
        prediction = bagging_predict(model, features)

        output = {
            "prediction": prediction
        }
        print(json.dumps(output))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
