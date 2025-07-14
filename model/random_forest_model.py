import json
import random
import joblib
from collections import Counter
from math import log2

# --- Load Dataset ---
with open("dataset.json", "r") as f:
    dataset = json.load(f)

with open("feature_keys.json", "r") as f:
    feature_keys = json.load(f)

X = [item["features"] for item in dataset]
y = [item["label"] for item in dataset]
feature_names = feature_keys["similarity_keys"]

print("\n🔍 Feature Names (first 10):")
for i, name in enumerate(feature_names[:10]):
    print(f"{i}: {name}")

# --- Decision Tree Implementation ---
def gini(groups, classes):
    total = sum(len(group) for group in groups)
    score = 0.0
    for group in groups:
        if len(group) == 0:
            continue
        proportion = Counter(row[-1] for row in group)
        group_score = sum((count / len(group)) ** 2 for count in proportion.values())
        score += (1 - group_score) * (len(group) / total)
    return score

def split(index, value, dataset):
    left, right = [], []
    for row in dataset:
        (left if row[0][index] < value else right).append(row)
    return left, right

def get_split(dataset):
    best_index, best_value, best_score, best_groups = 999, 999, 999, None
    for index in range(len(dataset[0][0])):
        for row in dataset:
            groups = split(index, row[0][index], dataset)
            score = gini(groups, [0, 1])
            if score < best_score:
                best_index, best_value, best_score, best_groups = index, row[0][index], score, groups
    return {"index": best_index, "value": best_value, "groups": best_groups}

def to_terminal(group):
    outcomes = [row[1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split_node(node, max_depth, min_size, depth):
    left, right = node["groups"]
    del node["groups"]
    if not left or not right:
        node["left"] = node["right"] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node["left"], node["right"] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node["left"] = to_terminal(left)
    else:
        node["left"] = get_split(left)
        split_node(node["left"], max_depth, min_size, depth + 1)
    if len(right) <= min_size:
        node["right"] = to_terminal(right)
    else:
        node["right"] = get_split(right)
        split_node(node["right"], max_depth, min_size, depth + 1)

def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split_node(root, max_depth, min_size, 1)
    return root

def predict(node, row):
    if row[node["index"]] < node["value"]:
        return predict(node["left"], row) if isinstance(node["left"], dict) else node["left"]
    else:
        return predict(node["right"], row) if isinstance(node["right"], dict) else node["right"]

# --- Random Forest ---
def subsample(dataset, ratio):
    return [random.choice(dataset) for _ in range(round(len(dataset) * ratio))]

def random_forest(train, max_depth, min_size, sample_size, n_trees, feature_names):
    trees = []
    # Track feature usage across all trees
    feature_usage = {i: 0 for i in range(len(feature_names))}
    
    for _ in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size)
        trees.append(tree)
        # Record feature usage in this tree
        record_feature_usage(tree, feature_usage)
    
    # Convert usage counts to importance scores
    total_splits = sum(feature_usage.values())
    feature_importance = {
        feature_names[i]: count/total_splits if total_splits > 0 else 0
        for i, count in feature_usage.items()
    }
    
    return trees, feature_importance

def record_feature_usage(node, feature_usage):
    if isinstance(node, dict):
        feature_usage[node["index"]] += 1
        record_feature_usage(node["left"], feature_usage)
        record_feature_usage(node["right"], feature_usage)

def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

# --- Train & Evaluate ---
dataset_combined = list(zip(X, y))
random.shuffle(dataset_combined)
split_index = int(len(dataset_combined) * 0.8)
train_set, test_set = dataset_combined[:split_index], dataset_combined[split_index:]

forest, importance = random_forest(
    train_set, 
    max_depth=20, 
    min_size=1, 
    sample_size=0.8, 
    n_trees=10,
    feature_names=feature_names
    )

# Save the model
joblib.dump(forest, "model.pkl")
print("Model saved to model.pkl")
with open("feature_importance.json", "w") as f:
    json.dump(importance, f, indent=2)

print("\n📊 Feature Importance:")
for feat, score in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{feat}: {score:.4f}")

# Evaluate
correct = 0
for row, label in test_set:
    prediction = bagging_predict(forest, row)
    correct += prediction == label

accuracy = correct / len(test_set)
print(f"Accuracy: {accuracy:.4f}")
