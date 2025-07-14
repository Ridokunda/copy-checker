import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, accuracy_score, 
                            precision_score, recall_score, f1_score,
                            confusion_matrix, roc_auc_score, roc_curve)
import joblib
import seaborn as sns

# Load the dataset
with open("dataset.json", "r") as f:
    data = json.load(f)

X = np.array([item["features"] for item in data])
y = np.array([item["label"] for item in data])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize models to compare
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    #"Gradient Boosting": GradientBoostingClassifier(random_state=42),
    #"SVM": SVC(probability=True, random_state=42),
    #"K-Nearest Neighbors": KNeighborsClassifier(),
    #"Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Dictionary to store results
results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': [],
    'ROC AUC': []
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n{'='*40}")
    print(f"Training and evaluating {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba) if hasattr(model, "predict_proba") else None
    
    # Store results
    results['Model'].append(name)
    results['Accuracy'].append(accuracy)
    results['Precision'].append(precision)
    results['Recall'].append(recall)
    results['F1 Score'].append(f1)
    results['ROC AUC'].append(roc_auc)
    
    # Print classification report
    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.close()
    
    # Plot ROC curve if applicable
    if hasattr(model, "predict_proba"):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(f'roc_curve_{name.lower().replace(" ", "_")}.png')
        plt.close()

# Create comparison table
results_df = pd.DataFrame(results)
print("\n" + "="*40)
print("\nModel Comparison Results:")
print(results_df.to_string(index=False))

# Save results to CSV
results_df.to_csv('model_comparison_results.csv', index=False)

# Cross-validation for best model
best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
best_model = models[best_model_name]
print(f"\nPerforming cross-validation for best model ({best_model_name})...")
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

# Save the best model
joblib.dump(best_model, "best_model.pkl")
print(f"\nBest model ({best_model_name}) saved as best_model.pkl")

# Feature importance for tree-based models
if hasattr(best_model, 'feature_importances_'):
    print("\nFeature Importances:")
    importances = best_model.feature_importances_
    with open("feature_keys.json", "r") as f:
        feature_names = json.load(f)["similarity_keys"]
    
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(feature_imp.head(10).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', 
                data=feature_imp.head(20))
    plt.title('Top 20 Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

print("\nTraining and evaluation complete!")