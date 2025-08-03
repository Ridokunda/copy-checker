import os
import re
import numpy as np
from glob import glob
import random
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class CodePlagiarismDetector:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.w2v_model = None
        self.nn_model = None
        
    def preprocess_code(self, file_path):
        """Clean and tokenize source code"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        
        code = re.sub(r'\/\/.*', '', code)  
        code = re.sub(r'\/\*.*?\*\/', '', code, flags=re.DOTALL)
        
        # Normalize whitespace and special tokens
        code = re.sub(r'\s+', ' ', code).strip()
        
        # Tokenize
        tokens = word_tokenize(code)
        
        return tokens
    
    def train_word2vec(self, code_files):
        """Train Word2Vec model on code tokens"""
        processed_code = [self.preprocess_code(f) for f in code_files]
        
        model = Word2Vec(
            sentences=processed_code,
            vector_size=100,
            window=5,
            min_count=1,
            workers=4,
            sg=1
        )
        
        return model
    
    def document_embedding(self, tokens, model):
        """Create document embedding by averaging token vectors"""
        vectors = [model.wv[token] for token in tokens if token in model.wv]
        if len(vectors) == 0:
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)
    
    def create_similarity_model(self, input_dim):
        """Create neural network model for plagiarism detection"""
        model = Sequential([
            Dense(256, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_file_paths(self):
        """Load all file paths from the structured directory"""
        cases = sorted(glob(os.path.join(self.root_dir, "case-*")))
        
        data = {
            "original": [],
            "non_plagiarized": [],
            "plagiarized": []
        }
        
        for case in cases:
            # Original file
            orig_files = glob(os.path.join(case, "original", "*"))
            if orig_files:
                data["original"].append(orig_files[0])
            
            # Non-plagiarized files
            non_plag = glob(os.path.join(case, "non-plagiarized", "*", "*"))
            data["non_plagiarized"].extend(non_plag)
            
            # Plagiarized files (all levels)
            plagiarized = glob(os.path.join(case, "plagiarized", "*", "*", "*"))
            data["plagiarized"].extend(plagiarized)
        
        return data
    
    def generate_pairs(self, data, num_negative_pairs=5):
        """Generate positive and negative training pairs"""
        positive_pairs = []
        negative_pairs = []
        
        # Positive pairs: original + plagiarized versions
        for orig in data["original"]:
            case_dir = os.path.dirname(os.path.dirname(orig))
            plagiarized_files = glob(os.path.join(case_dir, "plagiarized", "*", "*", "*"))
            for plag in plagiarized_files:
                positive_pairs.append((orig, plag, 1))
        
        # Negative pairs: original + non-plagiarized from other cases
        for orig in data["original"]:
            case_num = os.path.basename(os.path.dirname(os.path.dirname(orig)))
            other_non_plag = [f for f in data["non_plagiarized"] 
                            if not f.startswith(os.path.join(self.root_dir, case_num))]
            
            for non_plag in random.sample(other_non_plag, min(num_negative_pairs, len(other_non_plag))):
                negative_pairs.append((orig, non_plag, 0))
        
        # Combine and shuffle
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)
        
        return all_pairs
    
    def train(self):
        """Train the complete plagiarism detection system"""
        # 1. Load data
        data = self.load_file_paths()
        pairs = self.generate_pairs(data)
        
        # 2. Get all code files for Word2Vec training
        all_code_files = data["original"] + data["non_plagiarized"] + data["plagiarized"]
        
        # 3. Train Word2Vec
        print("Training Word2Vec model...")
        self.w2v_model = self.train_word2vec(all_code_files)
        
        # 4. Create training data
        print("Creating training pairs...")
        X = []
        y = []
        
        for file1, file2, label in pairs:
            tokens1 = self.preprocess_code(file1)
            tokens2 = self.preprocess_code(file2)
            
            emb1 = self.document_embedding(tokens1, self.w2v_model)
            emb2 = self.document_embedding(tokens2, self.w2v_model)
            
            # Feature engineering
            abs_diff = np.abs(emb1 - emb2)
            pair_features = np.concatenate([emb1, emb2, abs_diff])
            
            X.append(pair_features)
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # 5. Train neural network
        print("Training neural network...")
        self.nn_model = self.create_similarity_model(X.shape[1])
        history = self.nn_model.fit(
            X, y, 
            epochs=15, 
            batch_size=32, 
            validation_split=0.2,
            class_weight={0: 1., 1: 3.}  # Higher weight for plagiarism class
        )
        
        return history
    
    def evaluate(self):
        """Evaluate the model on all cases"""
        data = self.load_file_paths()
        cases = sorted(glob(os.path.join(self.root_dir, "case-*")))
        
        all_true = []
        all_pred = []
        all_probs = []
        file_info = []
        
        for case in cases:
            # Get original file
            orig_files = glob(os.path.join(case, "original", "*"))
            if not orig_files:
                continue
            orig_file = orig_files[0]
            
            # Get all files to compare against
            test_files = (glob(os.path.join(case, "non-plagiarized", "*", "*")) +
                         glob(os.path.join(case, "plagiarized", "*", "*", "*")))
            
            for test_file in test_files:
                prob = self.detect_plagiarism(orig_file, test_file)
                
                # Determine ground truth
                if "non-plagiarized" in test_file:
                    true_label = 0
                else:
                    true_label = 1
                
                pred_label = 1 if prob > 0.5 else 0
                
                all_true.append(true_label)
                all_pred.append(pred_label)
                all_probs.append(prob)
                
                # Extract level if plagiarized
                level = None
                if "plagiarized" in test_file:
                    parts = test_file.split(os.sep)
                    level = parts[-3]
                
                file_info.append({
                    "case": os.path.basename(case),
                    "file": test_file,
                    "level": level,
                    "prob": prob,
                    "true_label": true_label,
                    "pred_label": pred_label
                })
        
        # Generate classification report
        print("\nClassification Report:")
        print(classification_report(all_true, all_pred, target_names=["Non-plagiarized", "Plagiarized"]))
        
        # Generate confusion matrix
        cm = confusion_matrix(all_true, all_pred)
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=["Non-plagiarized", "Plagiarized"],
                    yticklabels=["Non-plagiarized", "Plagiarized"])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_Matrix.png") 
        plt.show()
        
        return file_info
    
    def detect_plagiarism(self, file1, file2):
        """Detect plagiarism between two files"""
        tokens1 = self.preprocess_code(file1)
        tokens2 = self.preprocess_code(file2)
        
        emb1 = self.document_embedding(tokens1, self.w2v_model)
        emb2 = self.document_embedding(tokens2, self.w2v_model)
        
        # Create input vector
        abs_diff = np.abs(emb1 - emb2)
        pair_vector = np.concatenate([emb1, emb2, abs_diff]).reshape(1, -1)
        
        # Predict
        probability = self.nn_model.predict(pair_vector, verbose=0)[0][0]
        
        return probability
    
    def analyze_by_level(self, evaluation_results):
        """Analyze performance by plagiarism level"""
        level_stats = {}
        
        for result in evaluation_results:
            if result["level"] is not None:
                level = result["level"]
                
                if level not in level_stats:
                    level_stats[level] = {
                        "count": 0,
                        "correct": 0,
                        "probs": []
                    }
                
                level_stats[level]["count"] += 1
                if result["pred_label"] == result["true_label"]:
                    level_stats[level]["correct"] += 1
                level_stats[level]["probs"].append(result["prob"])
        
        # Calculate metrics
        for level in level_stats:
            level_stats[level]["accuracy"] = (
                level_stats[level]["correct"] / level_stats[level]["count"]
            )
            level_stats[level]["avg_prob"] = (
                sum(level_stats[level]["probs"]) / len(level_stats[level]["probs"])
            )
        
        # Print results
        print("\nPerformance by Plagiarism Level:")
        for level, stats in sorted(level_stats.items()):
            print(f"{level}:")
            print(f"  Accuracy: {stats['accuracy']:.2%}")
            print(f"  Avg Probability: {stats['avg_prob']:.2f}")
            print(f"  Samples: {stats['count']}")
        
        return level_stats
    
    def save_models(self, w2v_path="code_w2v.model", nn_path="plagiarism_nn.h5"):
        """Save trained models to disk"""
        self.w2v_model.save(w2v_path)
        self.nn_model.save(nn_path)
        print(f"Models saved to {w2v_path} and {nn_path}")
    
    def load_models(self, w2v_path="code_w2v.model", nn_path="plagiarism_nn.h5"):
        """Load trained models from disk"""
        from gensim.models import Word2Vec
        from tensorflow.keras.models import load_model
        
        self.w2v_model = Word2Vec.load(w2v_path)
        self.nn_model = load_model(nn_path)
        print(f"Models loaded from {w2v_path} and {nn_path}")


if __name__ == "__main__":
    # Initialize detector with your dataset path
    detector = CodePlagiarismDetector(root_dir="IR-Plag-Dataset")
    
    # Train the models
    print("Starting training process...")
    training_history = detector.train()
    
    # Evaluate on all cases
    print("\nEvaluating model...")
    evaluation_results = detector.evaluate()
    
    # Analyze by plagiarism level
    #level_stats = detector.analyze_by_level(evaluation_results)
    
    # Save models for future use
    detector.save_models()
    