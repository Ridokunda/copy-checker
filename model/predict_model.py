import sys
import json
import joblib
from collections import Counter
import os
import numpy as np

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
    
    # Calculate confidence scores
    total_predictions = len(predictions)
    confidence_scores = {
        0: prediction_counts.get(0, 0) / total_predictions,  # Not plagiarized
        1: prediction_counts.get(1, 0) / total_predictions   # Plagiarized
    }
    
    # Get the most common prediction and its confidence
    most_common_prediction = max(confidence_scores, key=confidence_scores.get)
    confidence = confidence_scores[most_common_prediction]
    
    return most_common_prediction, confidence, confidence_scores

def calculate_feature_contributions(trees, row):
    """Calculate how much each feature contributed to the final prediction"""
    feature_contributions = np.zeros(len(row))
    
    for tree in trees:
        node = tree
        while isinstance(node, dict):
            feature_idx = node["index"]
            feature_value = row[feature_idx]
            threshold = node["value"]
            
            # Contribution is based on how far the value is from the threshold
            contribution = abs(feature_value - threshold)
            feature_contributions[feature_idx] += contribution
            
            node = node["left"] if feature_value < threshold else node["right"]
    
    # Normalize contributions
    if feature_contributions.sum() > 0:
        feature_contributions /= feature_contributions.sum()
    
    return feature_contributions.tolist()

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
        model_path = "model.pkl"
        if not os.path.exists(model_path):
            # Try alternative path
            model_path = "model.pkl"
            if not os.path.exists(model_path):
                raise FileNotFoundError("Model file not found")
        
        model_data = joblib.load(model_path)
        print(f"DEBUG: Model loaded successfully", file=sys.stderr)
        
        if not isinstance(model_data, dict) or 'forest' not in model_data:
            raise ValueError("Invalid model format - expected dictionary with 'forest' key")
        
        trees = model_data['forest']
        
        # Make prediction with confidence
        prediction, confidence, confidence_scores = bagging_predict_with_confidence(trees, features)
        
        # Calculate feature contributions
        feature_contributions = calculate_feature_contributions(trees, features)
        
        print(f"DEBUG: Prediction: {prediction}, Confidence: {confidence}", file=sys.stderr)
        print(f"DEBUG: Confidence Scores: {confidence_scores}", file=sys.stderr)
        print(f"DEBUG: Feature Contributions: {feature_contributions[:5]}...", file=sys.stderr)
        
        # Return result
        result = {
            "prediction": int(prediction),
            "confidence": [float(confidence_scores[0]), float(confidence_scores[1])],
            "success": True
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        print(json.dumps({
            "error": str(e),
            "success": False
        }))

if __name__ == "__main__":
    main()