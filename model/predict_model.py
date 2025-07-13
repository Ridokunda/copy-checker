import sys
import json
import joblib
from collections import Counter
import os

def predict(node, row):
    if isinstance(node, dict):
        if row[node["index"]] < node["value"]:
            return predict(node["left"], row) if isinstance(node["left"], dict) else node["left"]
        else:
            return predict(node["right"], row) if isinstance(node["right"], dict) else node["right"]
    else:
        return node

def bagging_predict_with_confidence(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    prediction_counts = Counter(predictions)
    
    # Get the most common prediction
    most_common_prediction = prediction_counts.most_common(1)[0][0]
    
    # Calculate confidence scores
    total_predictions = len(predictions)
    confidence_0 = prediction_counts.get(0, 0) / total_predictions  # Not plagiarized
    confidence_1 = prediction_counts.get(1, 0) / total_predictions  # Plagiarized
    
    return most_common_prediction, [confidence_0, confidence_1]

def main():
    try:
        # Read input from stdin
        input_json = sys.stdin.read().strip()
        if not input_json:
            raise ValueError("No input provided")
            
        payload = json.loads(input_json)
        features = payload["features"]
        
        # Debug: Print feature vector info
        print(f"DEBUG: Received {len(features)} features", file=sys.stderr)
        print(f"DEBUG: Feature vector: {features[:5]}...", file=sys.stderr)
        
        # Load the model
        model_path = "model/model.pkl"
        if not os.path.exists(model_path):
            # Try alternative path
            model_path = "model.pkl"
            if not os.path.exists(model_path):
                raise FileNotFoundError("Model file not found")
        
        model = joblib.load(model_path)
        print(f"DEBUG: Model loaded successfully", file=sys.stderr)
        
        # Make prediction with confidence
        prediction, confidence = bagging_predict_with_confidence(model, features)
        
        print(f"DEBUG: Prediction: {prediction}, Confidence: {confidence}", file=sys.stderr)
        
        # Return result
        result = {
            "prediction": int(prediction),
            "confidence": confidence
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()