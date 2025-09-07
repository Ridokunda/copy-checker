import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load dataset
with open('dataset.json', 'r') as f:
    data = json.load(f)

X = np.array([item['features'] for item in data])
y = np.array([item['label'] for item in data])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
svm.fit(X_train, y_train)

# Evaluate
y_pred = svm.predict(X_test)
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(svm, 'svm_model.pkl')
print('SVM model saved to svm_model.pkl')
