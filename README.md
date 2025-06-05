# Copy Checker

**Copy Checker** is an academic plagiarism detection tool designed to identify code similarity in Java source files. This tool is particularly useful for educators and academic institutions to verify the originality of student submissions. The system uses Abstract Syntax Trees (ASTs) and Machine Learning to detect copied or semantically similar code snippets.

## 🚀 Features

- 🔍 Detects code similarity in Java source files
- 🌳 Parses source files into Abstract Syntax Trees (ASTs)
- 🤖 Utilizes machine learning models trained on original and plagiarized Java code
- 📊 Generates similarity reports for each code pair
- 🛠 User-friendly interface for uploading and analyzing files (optional GUI)

## 🧠 How It Works

1. **Parsing Java Code**  
   Each submitted `.java` file is parsed into an Abstract Syntax Tree to capture the structure of the code.

2. **Feature Extraction**  
   The AST is processed to extract relevant features (e.g., statement patterns, control flow, method usage).

3. **Similarity Detection**  
   - A similarity score is computed based on structural and syntactic features.
   - The model classifies the code pair as **original** or **potentially plagiarized**.

4. **Result Output**  
   Results are displayed as a report, highlighting:
   - Degree of similarity
   - Code snippets or sections with high overlap
   - Plagiarism confidence score

## 🏗 Project Structure 
