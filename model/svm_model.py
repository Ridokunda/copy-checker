import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
with open('dataset.json', 'r') as f:
    data = json.load(f)

X = np.array([item['features'] for item in data])
y = np.array([item['label'] for item in data])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
svm = SVC(probability=True, class_weight='balanced', random_state=42)
svm.fit(X_train, y_train)

 # Evaluate the model
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Plot and save confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Plag', 'Plag'], yticklabels=['Not Plag', 'Plag'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix')
plt.tight_layout()
plt.savefig('model/confusion_matrix_svm.png')
plt.close()

# Save model
joblib.dump(svm, 'svm_model.pkl')
print('SVM model saved to svm_model.pkl')
