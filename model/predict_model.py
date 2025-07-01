
import sys
import json
import joblib
import numpy as np


model = joblib.load("model/model.pkl")

def main():
    # Read input 
    input_json = sys.stdin.read()
    payload = json.loads(input_json)

    
    features = np.array(payload["features"]).reshape(1, -1)
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0].tolist()

    # Output result
    output = {
        "prediction": int(prediction),
        "confidence": proba
    }
    print(json.dumps(output))

if __name__ == "__main__":
    main()
