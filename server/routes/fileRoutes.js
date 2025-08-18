const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { parseJavaFile, extractFeatures, tokenize } = require('../logic/JavaParser2.js');
const AdmZip = require("adm-zip");

const router = express.Router();
const upload = multer({ dest: 'uploads/' });

router.post('/predict', upload.fields([{ name: 'original' }, { name: 'suspect' }]), async (req, res) => {
  try {
    const originalPath = req.files['original'][0].path;
    const suspectPath = req.files['suspect'][0].path;

    console.log('Parsing files:', originalPath, suspectPath);

    const originalCode = fs.readFileSync(originalPath, 'utf-8');
    const suspectCode = fs.readFileSync(suspectPath, 'utf-8');
    const originalTokens = tokenize(originalCode);
    const suspectTokens = tokenize(suspectCode);

    const ast1 = parseJavaFile(originalPath);
    const ast2 = parseJavaFile(suspectPath);
    const f1 = extractFeatures(ast1, originalTokens);
    const f2 = extractFeatures(ast2, suspectTokens);
    
    const featureVector = [...Object.values(f1), ...Object.values(f2)];

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
      const featureVector = [...Object.values(f1), ...Object.values(f2)];
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
      const ast = parseJavaFile(f);
      const features = extractFeatures(ast);
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
        const result = await runPrediction(fileData[i].features, fileData[j].features);
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
    fs.rmdirSync(extractPath, { recursive: true });

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

module.exports = router;