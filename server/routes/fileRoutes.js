const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { parseJavaFile, extractFeatures, tokenize, Parser } = require('../logic/JavaParser2.js');

const router = express.Router();
const upload = multer({ dest: 'uploads/' });

// Load the feature keys used during training
let FEATURE_KEYS = null;
try {
  FEATURE_KEYS = JSON.parse(fs.readFileSync('./feature_keys.json', 'utf-8'));
  console.log(`Loaded ${FEATURE_KEYS.length} feature keys for consistent vectorization`);
} catch (err) {
  console.error('Warning: Could not load feature_keys.json. Using dynamic feature extraction.');
}

// Convert a feature object into a vector using the training feature keys
function toVector(features, keys) {
  // If keys is an object (like from feature_keys.json), use its similarity_keys property
  if (keys && typeof keys === 'object' && !Array.isArray(keys) && keys.similarity_keys) {
    return keys.similarity_keys.map(k => features[k] || 0);
  }
  // Otherwise assume it's an array
  return Array.isArray(keys) ? keys.map(k => features[k] || 0) : [];
}

// Helper function to normalize feature vectors using training keys
function normalizeFeatures(features) {
  if (FEATURE_KEYS) {
    return toVector(features, FEATURE_KEYS);
  } else {
    // Fallback to dynamic extraction (not recommended for production)
    const baseFeatures = [
      features.num_classes || 0,
      features.num_methods || 0,
      features.num_if || 0,
      features.num_for || 0,
      features.num_while || 0,
      features.num_return || 0,
      features.num_imports || 0,
      features.num_package || 0,
      features.num_expressions || 0,
      features.num_statements || 0,
      features.avg_method_length || 0,
      features.max_depth || 0
    ];
    
    const ngramKeys = Object.keys(features)
      .filter(key => key.startsWith('ngram_'))
      .sort();
    
    const ngramFeatures = ngramKeys.map(key => features[key] || 0);
    
    return [...baseFeatures, ...ngramFeatures];
  }
}

router.post('/predict', upload.fields([{ name: 'original' }, { name: 'suspect' }]), async (req, res) => {
  try {
    const originalPath = req.files['original'][0].path;
    const suspectPath = req.files['suspect'][0].path;

    console.log('Parsing files:', originalPath, suspectPath);

    // Parse and extract features
    const ast1 = parseJavaFile(originalPath);
    const ast2 = parseJavaFile(suspectPath);
    const f1 = extractFeatures(ast1);
    const f2 = extractFeatures(ast2);

    console.log('Features 1:', f1);
    console.log('Features 2:', f2);

    // Normalize features to ensure consistent vector length
    const normalizedF1 = normalizeFeatures(f1);
    const normalizedF2 = normalizeFeatures(f2);
    
    const vector = [...normalizedF1, ...normalizedF2];
    
    console.log('Feature vector length:', vector.length);
    console.log('Feature vector:', vector);

    // Call Python script
    const py = spawn('python', ['model/predict_model2.py']);
    let output = '';
    let errorOutput = '';

    py.stdin.write(JSON.stringify({ features: vector }));
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
        
        // Clean up uploaded files
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

router.post('/compare', (req, res) => {
  const { features1, features2 } = req.body;

  if (!features1 || !features2) {
    return res.status(400).json({ error: 'Missing feature vectors' });
  }

  const input = JSON.stringify({ features: [...features1, ...features2] });

  const py = spawn('python', ['model/predict_model.py']);
  let result = '';

  py.stdout.on('data', (data) => {
    result += data.toString();
  });

  py.stderr.on('data', (data) => {
    console.error('Python error:', data.toString());
  });

  py.on('close', (code) => {
    try {
      const output = JSON.parse(result);
      res.json(output);
    } catch (err) {
      res.status(500).json({ error: 'Failed to parse Python output' });
    }
  });

  py.stdin.write(input);
  py.stdin.end();
});

module.exports = router;