import json
import random
import joblib
from collections import Counter
from math import sqrt
from sklearn.metrics import precision_score, recall_score, f1_score

# --- Load Dataset and Feature Keys ---
def load_feature_keys(file_path):
    """Load feature keys to understand what each feature represents."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            print(f"✅ Feature keys loaded from {file_path}")
            print(f"   Structure: {type(data)}")
            if isinstance(data, dict):
                print(f"   Keys: {list(data.keys())}")
            return data
    except FileNotFoundError:
        print(f"⚠️  Feature keys file not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"⚠️  Error parsing feature keys file: {e}")
        return None

def load_dataset(file_path):
    """Load dataset from JSON file with validation."""
    with open(file_path, "r") as f:
        dataset = json.load(f)
    if not dataset or not all("features" in item and "label" in item for item in dataset):
        raise ValueError("Dataset must contain 'features' and 'label' keys")
    X = [item["features"] for item in dataset]
    y = [item["label"] for item in dataset]
    return list(zip(X, y))

def validate_dataset(dataset, feature_info):
    """Validate that dataset matches expected structure."""
    if not dataset:
        raise ValueError("Dataset is empty")
    
    actual_feature_count = len(dataset[0][0])
    
    if feature_info:
        # Handle both dict and list structures
        if isinstance(feature_info, dict):
            expected_feature_count = feature_info.get('feature_count', len(feature_info.get('similarity_keys', [])))
        else:
            # If it's a list, assume it's the similarity_keys
            expected_feature_count = len(feature_info)
            
        if expected_feature_count > 0 and actual_feature_count != expected_feature_count:
            print(f"⚠️  Feature count mismatch: expected {expected_feature_count}, got {actual_feature_count}")
        
        print(f"✅ Dataset validation passed: {len(dataset)} samples, {actual_feature_count} features")
    else:
        print(f"✅ Dataset loaded: {len(dataset)} samples, {actual_feature_count} features")
    
    # Check label distribution
    labels = [label for _, label in dataset]
    label_counts = Counter(labels)
    for label, count in label_counts.items():
        percentage = (count / len(dataset)) * 100
        label_name = "Plagiarized" if label == 1 else "Non-plagiarized"
        print(f"   {label_name}: {count} samples ({percentage:.1f}%)")

# --- Decision Tree Implementation ---
def gini(groups, classes):
    """Calculate Gini impurity for a split."""
    total = sum(len(group) for group in groups)
    if total == 0:
        return 0.0
    score = 0.0
    for group in groups:
        if len(group) == 0:
            continue
        proportion = Counter(row[1] for row in group)
        group_score = sum((count / len(group)) ** 2 for count in proportion.values())
        score += (1 - group_score) * (len(group) / total)
    return score

def split(index, value, dataset):
    """Split dataset based on feature index and value."""
    left, right = [], []
    for row in dataset:
        (left if row[0][index] < value else right).append(row)
    return left, right

def get_split(dataset, n_features, impurity_reduction=None):
    """Find the best split using a random subset of features, tracking impurity reduction."""
    best_index, best_value, best_score, best_groups = None, None, 999, None
    features = random.sample(range(len(dataset[0][0])), n_features)
    
    # Calculate parent impurity once
    parent_gini = gini([dataset], [0, 1])
    
    for index in features:
        values = sorted(set(row[0][index] for row in dataset))
        if len(values) < 2:
            continue
            
        for i in range(len(values) - 1):
            value = (values[i] + values[i + 1]) / 2
            groups = split(index, value, dataset)
            score = gini(groups, [0, 1])
            
            if score < best_score:
                best_index, best_value, best_score, best_groups = index, value, score, groups
                if impurity_reduction is not None:
                    # Properly calculate impurity reduction
                    impurity_reduction[index] = (parent_gini - score) * len(dataset)
    
    if best_groups is None:
        return {"terminal": to_terminal(dataset)}
    return {"index": best_index, "value": best_value, "groups": best_groups}

def to_terminal(group):
    """Create a terminal node with the most common label."""
    if not group:
        return 0  # Default to non-plagiarized if empty
    outcomes = [row[1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split_node(node, max_depth, min_size, depth, n_features, impurity_reduction=None):
    """Recursively split nodes to build a decision tree."""
    if "terminal" in node:
        return
    
    left, right = node["groups"]
    del node["groups"]
    
    # Check stopping conditions
    if not left or not right:
        node["left"] = node["right"] = to_terminal(left + right)
        return
    
    if depth >= max_depth:
        node["left"], node["right"] = to_terminal(left), to_terminal(right)
        return
    
    # Process left branch
    if len(left) <= min_size:
        node["left"] = to_terminal(left)
    else:
        node["left"] = get_split(left, n_features, impurity_reduction)
        split_node(node["left"], max_depth, min_size, depth + 1, n_features, impurity_reduction)
    
    # Process right branch
    if len(right) <= min_size:
        node["right"] = to_terminal(right)
    else:
        node["right"] = get_split(right, n_features, impurity_reduction)
        split_node(node["right"], max_depth, min_size, depth + 1, n_features, impurity_reduction)

def build_tree(train, max_depth, min_size, n_features):
    """Build a decision tree with proper impurity reduction tracking."""
    if not train:
        return {"terminal": 0}, [0] * n_features
    
    impurity_reduction = [0.0] * len(train[0][0])
    root = get_split(train, n_features, impurity_reduction)
    split_node(root, max_depth, min_size, 1, n_features, impurity_reduction)
    return root, impurity_reduction

def predict(node, row):
    """Predict the label for a single row."""
    if "terminal" in node:
        return node["terminal"]
    
    if node["index"] is None or node["value"] is None:
        return 0  # Default prediction
    
    if row[node["index"]] < node["value"]:
        return predict(node["left"], row) if isinstance(node["left"], dict) else node["left"]
    else:
        return predict(node["right"], row) if isinstance(node["right"], dict) else node["right"]

# --- Random Forest ---
def subsample(dataset, ratio):
    """Create a random subsample with replacement."""
    sample_size = max(1, round(len(dataset) * ratio))
    return [random.choice(dataset) for _ in range(sample_size)]

def random_forest(train, max_depth, min_size, sample_size, n_trees, n_features):
    """Train a Random Forest and compute OOB error."""
    trees = []
    oob_predictions = {i: [] for i in range(len(train))}
    feature_importance = [0.0] * len(train[0][0])
    
    print(f"🌲 Training {n_trees} trees...")
    
    for tree_idx in range(n_trees):
        if (tree_idx + 1) % 10 == 0:
            print(f"   Trees completed: {tree_idx + 1}/{n_trees}")
        
        # Create bootstrap sample
        sample = subsample(train, sample_size)
        sample_indices = set(id(row) for row in sample)
        
        # Build tree
        tree, tree_importance = build_tree(sample, max_depth, min_size, n_features)
        trees.append(tree)
        
        # Collect OOB predictions
        for i, (row, _) in enumerate(train):
            if id(row) not in sample_indices:
                pred = predict(tree, row)
                oob_predictions[i].append(pred)
        
        # Accumulate feature importance
        for i, imp in enumerate(tree_importance):
            feature_importance[i] += imp
    
    # Calculate OOB error
    oob_error = 0
    oob_count = 0
    for i, preds in oob_predictions.items():
        if preds:
            pred = max(set(preds), key=preds.count)
            oob_error += pred != train[i][1]
            oob_count += 1
    
    oob_error = oob_error / oob_count if oob_count > 0 else 0.0
    print(f"✅ OOB Error: {oob_error:.4f}")
    
    # Normalize feature importance by number of trees
    feature_importance = [imp / n_trees for imp in feature_importance]
    
    return trees, feature_importance

def bagging_predict(trees, row):
    """Predict using majority voting across trees."""
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

def balance_dataset(dataset):
    """Balance the dataset by undersampling the majority class."""
    # Separate classes
    plagiarized = [item for item in dataset if item[1] == 1]
    non_plagiarized = [item for item in dataset if item[1] == 0]
    
    print(f"📊 Original class distribution:")
    print(f"   Plagiarized: {len(plagiarized)} samples")
    print(f"   Non-plagiarized: {len(non_plagiarized)} samples")
    
    # Balance by undersampling majority class
    min_class_size = min(len(plagiarized), len(non_plagiarized))
    
    # If one class is too small, use a minimum threshold
    if min_class_size < 20:
        # Use original dataset if too small to balance
        print("⚠️  Classes too small to balance effectively, using original dataset")
        return dataset
    
    # Randomly sample from each class
    balanced_plagiarized = random.sample(plagiarized, min_class_size)
    balanced_non_plagiarized = random.sample(non_plagiarized, min_class_size)
    
    balanced_dataset = balanced_plagiarized + balanced_non_plagiarized
    random.shuffle(balanced_dataset)
    
    print(f"📊 Balanced class distribution:")
    print(f"   Plagiarized: {min_class_size} samples")
    print(f"   Non-plagiarized: {min_class_size} samples")
    
    return balanced_dataset
    """Analyze feature importance with knowledge of feature types."""
    if not feature_info:
        print("📊 Feature Importance (Generic):")
        for i, imp in enumerate(importance):
            if imp > 0.001:  # Only show important features
                print(f"   Feature {i}: {imp:.4f}")
        return
    
    # Handle both dict and list structures
    if isinstance(feature_info, dict):
        similarity_keys = feature_info.get('similarity_keys', [])
    else:
        # If it's a list, assume it's the similarity_keys
        similarity_keys = feature_info if isinstance(feature_info, list) else []
    
    if not similarity_keys:
        print("📊 Feature Importance (No feature names available):")
        for i, imp in enumerate(importance):
            if imp > 0.001:
                print(f"   Feature {i}: {imp:.4f}")
        return
    
    print("📊 Feature Importance Analysis:")
    
    # Group features by type
    similarity_features = []
    ratio_features = []
    diff_features = []
    
    for i, imp in enumerate(importance):
        if i < len(similarity_keys):
            feature_name = similarity_keys[i]
            if feature_name.startswith('ratio_'):
                ratio_features.append((feature_name, imp))
            elif feature_name.startswith('diff_'):
                diff_features.append((feature_name, imp))
            else:
                similarity_features.append((feature_name, imp))
        else:
            # Handle case where there are more features than names
            similarity_features.append((f"feature_{i}", imp))
    
    # Sort by importance
    similarity_features.sort(key=lambda x: x[1], reverse=True)
    ratio_features.sort(key=lambda x: x[1], reverse=True)
    diff_features.sort(key=lambda x: x[1], reverse=True)
    
    print("   📏 Similarity Metrics:")
    for name, imp in similarity_features:
        if imp > 0.001:
            print(f"     {name}: {imp:.4f}")
    
    if ratio_features:
        print("   📐 Ratio Features (top 10):")
        for name, imp in ratio_features[:10]:
            if imp > 0.001:
                print(f"     {name}: {imp:.4f}")
    
    if diff_features:
        print("   📊 Difference Features (top 10):")
        for name, imp in diff_features[:10]:
            if imp > 0.001:
                print(f"     {name}: {imp:.4f}")

# --- Train & Evaluate ---
def train_and_evaluate(dataset_path, feature_keys_path="feature_keys.json", 
                      n_trees=100, max_depth=10, min_size=5, sample_size=0.7, test_size=0.2,
                      balance_classes=True, feature_selection=True):
    """Train and evaluate the Random Forest model with feature importance."""
    print("🔄 Loading dataset and feature information...")
    
    # Load feature information
    feature_info = load_feature_keys(feature_keys_path)
    
    # Load and validate dataset
    dataset = load_dataset(dataset_path)
    validate_dataset(dataset, feature_info)
    
    # Balance classes if requested
    if balance_classes:
        dataset = balance_dataset(dataset)
        print(f"✅ Dataset balanced: {len(dataset)} samples")
    
    # Shuffle and split dataset
    random.shuffle(dataset)
    n_features = int(sqrt(len(dataset[0][0])))
    split_index = int(len(dataset) * (1 - test_size))
    train_set, test_set = dataset[:split_index], dataset[split_index:]
    
    print(f"📊 Dataset Split:")
    print(f"   Training samples: {len(train_set)}")
    print(f"   Testing samples: {len(test_set)}")
    print(f"   Features per sample: {len(dataset[0][0])}")
    print(f"   Features per split: {n_features}")
    
    # Train Random Forest
    print("🔄 Training Random Forest...")
    forest, feature_importance = random_forest(
        train_set, max_depth, min_size, sample_size, n_trees, n_features
    )
    
    # Save model
    model_data = {
        'forest': forest,
        'feature_info': feature_info,
        'n_features': n_features
    }
    joblib.dump(model_data, "model.pkl")
    print("✅ Model saved to model.pkl")
    
    # Evaluate on test set
    print("🔄 Evaluating model...")
    predictions = [bagging_predict(forest, row) for row, _ in test_set]
    labels = [label for _, label in test_set]
    
    # Calculate metrics
    accuracy = sum(pred == label for pred, label in zip(predictions, labels)) / len(test_set)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    # Calculate confusion matrix manually
    tp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 1)
    fp = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 0)
    tn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 0)
    fn = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 1)
    
    print(f"📊 Model Performance:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"📊 Confusion Matrix:")
    print(f"   True Positives: {tp}")
    print(f"   False Positives: {fp}")
    print(f"   True Negatives: {tn}")
    print(f"   False Negatives: {fn}")
    
    # Analyze feature importance
    analyze_feature_importance(feature_importance, feature_info)
    
    return forest, feature_importance

def load_and_predict(model_path, features):
    """Load saved model and make predictions."""
    model_data = joblib.load(model_path)
    forest = model_data['forest']
    
    if isinstance(features[0], list):
        # Multiple samples
        return [bagging_predict(forest, row) for row in features]
    else:
        # Single sample
        return bagging_predict(forest, features)

# --- Main Execution ---
if __name__ == "__main__":
    print("🚀 Starting Random Forest Training for Plagiarism Detection")
    print("=" * 60)
    
    try:
        forest, importance = train_and_evaluate(
            dataset_path="dataset.json",
            feature_keys_path="feature_keys.json",
            n_trees=100,
            max_depth=12,
            min_size=3,
            sample_size=0.8,
            test_size=0.2,
            balance_classes=True,
            feature_selection=True
        )
        
        print("=" * 60)
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise