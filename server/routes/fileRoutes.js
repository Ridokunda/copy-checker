const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { parseJavaFile, extractFeatures, tokenize, getTokenMap,
  linearizeAST,
  levenshteinDistanceAST,
  levenshteinSimilarityAST,
  astLevenshteinDistance,
  astLevenshteinSimilarity } = require('../logic/JavaParser2.js');
const AdmZip = require("adm-zip");

const router = express.Router();
const upload = multer({ dest: 'uploads/' });

router.post('/predict', upload.fields([{ name: 'original' }, { name: 'suspect' }]), async (req, res) => {
  try {
    const originalPath = req.files['original'][0].path;
    const suspectPath = req.files['suspect'][0].path;

    console.log('Parsing files:', originalPath, suspectPath);


    // --- Build normalized feature vector as in build_dataset2.js ---
    const originalCode = fs.readFileSync(originalPath, 'utf-8');
    const suspectCode = fs.readFileSync(suspectPath, 'utf-8');
    const originalTokens = tokenize(originalCode);
    const suspectTokens = tokenize(suspectCode);
    const originalTokenMap = getTokenMap(originalTokens);
    const suspectTokenMap = getTokenMap(suspectTokens);

    const ast1 = parseJavaFile(originalPath);
    const ast2 = parseJavaFile(suspectPath);
    const f1 = extractFeatures(ast1, originalTokens.length);
    const f2 = extractFeatures(ast2, suspectTokens.length);
    f1['num_unique_tokens'] = originalTokenMap.size;
    f2['num_unique_tokens'] = suspectTokenMap.size;
    // add levenshtein distance and similarity of ASTs
    f2['ast_levenshtein_distance'] = astLevenshteinDistance(originalPath, suspectPath);
    f2['ast_levenshtein_similarity'] = astLevenshteinSimilarity(originalPath, suspectPath);

    // Load allKeys from dataset.json for consistent feature order
    const vec1 = Object.values(f1);
    const vec2 = Object.values(f2);
    const featureVector = [...vec1, ...vec2];
    console.log('feature vector:', featureVector);
    const py = spawn('python', ['model/predict_model2.py']);
    let output = '';
    let errorOutput = '';

    py.stdin.write(JSON.stringify({ features: featureVector }));
    py.stdin.end();

    py.stdout.on('data', (data) => {
      output += data.toString();
    });

    py.stderr.on('data', (data) => {
      errorOutput += data.toString();
      console.error(`Python error: ${data}`);
    });

    py.on('close', (code) => {
      if (code !== 0) {
        console.error('Python script failed with code:', code);
        console.error('Error output:', errorOutput);
        return res.status(500).json({ success: false, error: 'Python script failed' });
      }
      
      try {
        console.log('Python output:', output);
        const result = JSON.parse(output);
        
        fs.unlinkSync(originalPath);
        fs.unlinkSync(suspectPath);
        
        res.json({ success: true, ...result });
      } catch (e) {
        console.error('Failed to parse Python output', e);
        console.error('Raw output:', output);
        res.status(500).json({ success: false, error: 'Failed to parse prediction' });
      }
    });

  } catch (err) {
    console.error('Route error:', err);
    res.status(500).json({ success: false, error: 'Server error: ' + err.message });
  }
});

function runPrediction(f1, f2) {
  return new Promise((resolve, reject) => {
    try {

      const vec1 = Object.values(f1);
      const vec2 = Object.values(f2);
      const featureVector = [...vec1, ...vec2];
      const py = spawn("python", ["model/predict_model2.py"]);
      let output = "";
      let errorOutput = "";

      py.stdin.write(JSON.stringify({ features: featureVector }));
      py.stdin.end();

      py.stdout.on("data", (data) => {
        output += data.toString();
      });

      py.stderr.on("data", (data) => {
        errorOutput += data.toString();
        console.error(`Python error: ${data}`);
      });

      py.on("close", (code) => {
        if (code !== 0) {
          console.error("Python script failed:", errorOutput);
          return reject("Python script failed");
        }
        try {
          const result = JSON.parse(output);
          resolve(result);
        } catch (e) {
          console.error("Parse error:", e, output);
          reject("Failed to parse prediction output");
        }
      });
    } catch (err) {
      reject(err);
    }
  });
}

router.post("/batch-upload", upload.single("zipfile"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: "No ZIP file uploaded" });
    }

    // Extract ZIP to temp folder
    const zipPath = req.file.path;
    const extractPath = path.join("uploads", `extract-${Date.now()}`);
    fs.mkdirSync(extractPath);

    const zip = new AdmZip(zipPath);
    zip.extractAllTo(extractPath, true);

    // Collect all .java files
    const javaFiles = fs.readdirSync(extractPath)
      .filter(file => file.endsWith(".java"))
      .map(file => path.join(extractPath, file));

    if (javaFiles.length < 2) {
      return res.status(400).json({ message: "ZIP must contain at least 2 Java files" });
    }

    // Parse + extract features
    const fileData = javaFiles.map(f => {
      const code = fs.readFileSync(f, 'utf-8');
      const tokens = tokenize(code);
      const tokenMap = getTokenMap(tokens);
      const ast = parseJavaFile(f);
      const features = extractFeatures(ast, tokens.length);
      features['num_unique_tokens'] = tokenMap.size;
      // Optionally add AST metrics if needed (for pairwise, see below)
      return {
        filename: path.basename(f),
        path: f,
        features
      };
    });


    // Compare each pair
    const comparisons = [];
    for (let i = 0; i < fileData.length; i++) {
      for (let j = i + 1; j < fileData.length; j++) {
        // Add AST metrics for the second file in the pair
        const f1 = { ...fileData[i].features };
        const f2 = { ...fileData[j].features };
        f2['ast_levenshtein_distance'] = astLevenshteinDistance(fileData[i].path, fileData[j].path);
        f2['ast_levenshtein_similarity'] = astLevenshteinSimilarity(fileData[i].path, fileData[j].path);
        const vec1 = Object.values(f1);
        const vec2 = Object.values(f2);;
        const featureVector = [...vec1, ...vec2];
        // Use runPrediction with normalized vectors
        const result = await new Promise((resolve, reject) => {
          const py = spawn("python", ["model/predict_model2.py"]);
          let output = "";
          let errorOutput = "";
          py.stdin.write(JSON.stringify({ features: featureVector }));
          py.stdin.end();
          py.stdout.on("data", (data) => {
            output += data.toString();
          });
          py.stderr.on("data", (data) => {
            errorOutput += data.toString();
            console.error(`Python error: ${data}`);
          });
          py.on("close", (code) => {
            if (code !== 0) {
              console.error("Python script failed:", errorOutput);
              return reject("Python script failed");
            }
            try {
              const parsed = JSON.parse(output);
              resolve(parsed);
            } catch (e) {
              console.error("Parse error:", e, output);
              reject("Failed to parse prediction output");
            }
          });
        });
        comparisons.push({
          file1: fileData[i].filename,
          file2: fileData[j].filename,
          ...result
        });
      }
    }

    // Cleanup: delete ZIP + extracted files
    fs.unlinkSync(zipPath);
    fileData.forEach(f => fs.unlinkSync(f.path));
    // Use fs.rmSync for recursive directory removal (Node.js >= v14.14.0)
    if (fs.rmSync) {
      fs.rmSync(extractPath, { recursive: true, force: true });
    } else {
      // Fallback for older Node.js
      fs.rmdirSync(extractPath, { recursive: true });
    }

    // Build report
    const report = {
      totalFiles: fileData.length,
      comparisons,
      submittedAt: new Date()
    };

    const reportPath = path.join("uploads", `plagiarism-report-${Date.now()}.json`);
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

    res.json({
      success: true,
      message: "Plagiarism report generated from ZIP",
      report
    });

  } catch (err) {
    console.error("Batch ZIP route error:", err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// Neural Network model prediction route
router.post('/nn-predict', upload.fields([{ name: 'original' }, { name: 'suspect' }]), async (req, res) => {
  try {
    const originalPath = req.files['original'][0].path;
    const suspectPath = req.files['suspect'][0].path;

    // Call Python script for NN model prediction
    const py = spawn('python', [
      'model/nn_predict.py',
      originalPath,
      suspectPath
    ]);
    let output = '';
    let errorOutput = '';

    py.stdout.on('data', (data) => {
      output += data.toString();
    });

    py.stderr.on('data', (data) => {
      errorOutput += data.toString();
      console.error(`Python error: ${data}`);
    });

    py.on('close', (code) => {
      // Clean up uploaded files
      fs.unlinkSync(originalPath);
      fs.unlinkSync(suspectPath);

      if (code !== 0) {
        console.error('Python script failed with code:', code);
        console.error('Error output:', errorOutput);
        return res.status(500).json({ success: false, error: 'Python script failed' });
      }
      try {
        const result = JSON.parse(output);
        res.json({ success: true, ...result });
      } catch (e) {
        console.error('Failed to parse Python output', e);
        res.status(500).json({ success: false, error: 'Failed to parse prediction' });
      }
    });
  } catch (err) {
    console.error('NN Route error:', err);
    res.status(500).json({ success: false, error: 'Server error: ' + err.message });
  }
});

module.exports = router;