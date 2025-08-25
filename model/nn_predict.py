import sys
import json
import numpy as np
from nn_model import CodePlagiarismDetector

def main():
    try:
        if len(sys.argv) != 3:
            print(json.dumps({"error": "Usage: nn_predict.py <original_file> <suspect_file>"}))
            sys.exit(1)

        original_file = sys.argv[1]
        suspect_file = sys.argv[2]

        # Initialize and load models
        detector = CodePlagiarismDetector(root_dir="IR-Plag-Dataset")
        detector.load_models()  # Assumes models are in default paths

        # Predict plagiarism probability
        prob = detector.detect_plagiarism(original_file, suspect_file)
        prediction = 1 if prob > 0.5 else 0
        confidence = [1 - float(prob), float(prob)]

        # Only this line prints to stdout!
        print(json.dumps({
            "prediction": prediction,
            "confidence": confidence
        }))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()