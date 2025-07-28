import json
import random
import joblib
from collections import Counter
from math import log2
import matplotlib.pyplot as plt

# --- Load Dataset ---
with open("dataset.json", "r") as f:
    dataset = json.load(f)


X = [item["features"] for item in dataset]
y = [item["label"] for item in dataset]


# --- Decision Tree Implementation ---
# calculate the score of a split
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
# get the best feature and value to split the dataset
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

def random_forest(train, max_depth, min_size, sample_size, n_trees):
    trees = []
    
    for _ in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size)
        trees.append(tree)
    return trees


def bagging_predict(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

def bagging_predict_proba(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    positive_votes = predictions.count(1)
    return positive_votes / len(trees)

# --- Train & Evaluate ---
dataset_combined = list(zip(X, y))
random.shuffle(dataset_combined)
split_index = int(len(dataset_combined) * 0.8)
train_set, test_set = dataset_combined[:split_index], dataset_combined[split_index:]

forest = random_forest(
    train_set, 
    max_depth=15, 
    min_size=5, 
    sample_size=0.8, 
    n_trees=100,
    )

# Save the model
joblib.dump(forest, "model.pkl")
print("Model saved to model.pkl")


# Evaluate with Precision and Recall
TP = FP = FN = TN = 0

for row, label in test_set:
    prediction = bagging_predict(forest, row)
    
    if prediction == 1 and label == 1:
        TP += 1
    elif prediction == 1 and label == 0:
        FP += 1
    elif prediction == 0 and label == 1:
        FN += 1
    elif prediction == 0 and label == 0:
        TN += 1

accuracy = (TP + TN) / len(test_set)
if (TP + FP) > 0 :
    precision = TP / (TP + FP)
else:
    precision = 0
if (TP + FN) > 0:
    recall = TP / (TP + FN) 
else:
    recall = 0
if (precision + recall) > 0:
    f1 = 2 * precision * recall / (precision + recall)
else:
    f1 = 0

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion Matrix
print("\nConfusion Matrix:")
print(f"           Predicted")
print(f"          0      1")
print(f"Actual 0  {TN}   {FP}")
print(f"Actual 1  {FN}   {TP}")

# ROC Curve Calculation
probs_labels = []
for row, label in test_set:
    prob = bagging_predict_proba(forest, row)
    probs_labels.append((prob, label))

# Sort by probability descending
probs_labels.sort(key=lambda x: x[0], reverse=True)

thresholds = [i / 100 for i in range(100, -1, -1)]
roc_points = []

P = sum(1 for _, label in probs_labels if label == 1)
N = sum(1 for _, label in probs_labels if label == 0)

for thresh in thresholds:
    TP = FP = 0
    for prob, label in probs_labels:
        if prob >= thresh:
            if label == 1:
                TP += 1
            else:
                FP += 1
    TPR = TP / P if P > 0 else 0
    FPR = FP / N if N > 0 else 0
    roc_points.append((FPR, TPR))

# AUC Calculation
auc = 0
for i in range(1, len(roc_points)):
    x1, y1 = roc_points[i - 1]
    x2, y2 = roc_points[i]
    auc += (x2 - x1) * (y1 + y2) / 2

print(f"\nAUC: {auc:.4f}")


# Plot ROC Curve 
fprs = [p[0] for p in roc_points]
tprs = [p[1] for p in roc_points]

plt.figure(figsize=(6, 6))
plt.plot(fprs, tprs, marker='.', color='blue', label=f'ROC Curve (AUC={auc:.4f})')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend()

plt.savefig("roc_curve.png") 
plt.show()