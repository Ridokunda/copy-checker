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


function calculateSimilarityFeatures(f1, f2, keys) {
    const similarities = {};
    
    // Convert to vectors first
    const v1 = keys.map(k => f1[k] || 0);
    const v2 = keys.map(k => f2[k] || 0);
    
    // 1. Cosine similarity
    const dotProduct = v1.reduce((sum, val, i) => sum + val * v2[i], 0);
    const norm1 = Math.sqrt(v1.reduce((sum, val) => sum + val * val, 0));
    const norm2 = Math.sqrt(v2.reduce((sum, val) => sum + val * val, 0));
    similarities.cosine_similarity = (norm1 * norm2) > 0 ? dotProduct / (norm1 * norm2) : 0;
    
    // 2. Euclidean distance (normalized)
    const euclideanDist = Math.sqrt(v1.reduce((sum, val, i) => sum + Math.pow(val - v2[i], 2), 0));
    const maxDist = Math.sqrt(keys.length * Math.max(...v1.concat(v2)) ** 2);
    similarities.euclidean_similarity = maxDist > 0 ? 1 - (euclideanDist / maxDist) : 1;
    
    // 3. Manhattan distance (normalized)
    const manhattanDist = v1.reduce((sum, val, i) => sum + Math.abs(val - v2[i]), 0);
    const maxManhattan = keys.length * Math.max(...v1.concat(v2));
    similarities.manhattan_similarity = maxManhattan > 0 ? 1 - (manhattanDist / maxManhattan) : 1;
    
    // 4. Jaccard similarity for binary features (presence/absence)
    const binary1 = v1.map(x => x > 0 ? 1 : 0);
    const binary2 = v2.map(x => x > 0 ? 1 : 0);
    const intersection = binary1.reduce((sum, val, i) => sum + (val && binary2[i] ? 1 : 0), 0);
    const union = binary1.reduce((sum, val, i) => sum + (val || binary2[i] ? 1 : 0), 0);
    similarities.jaccard_similarity = union > 0 ? intersection / union : 0;
    
    // 5. Individual feature ratios and differences
    keys.forEach(key => {
        const val1 = f1[key] || 0;
        const val2 = f2[key] || 0;
        const maxVal = Math.max(val1, val2);
        const minVal = Math.min(val1, val2);
        
        // Ratio similarity (how similar are the values)
        similarities[`ratio_${key}`] = maxVal > 0 ? minVal / maxVal : 1;
        
        // Absolute difference (normalized)
        const maxPossible = Math.max(val1, val2, 1); // Avoid division by zero
        similarities[`diff_${key}`] = 1 - (Math.abs(val1 - val2) / maxPossible);
    });
    
    return similarities;
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

    // Normalize features to ensure consistent vector length
    /*const normalizedF1 = normalizeFeatures(f1);
    const normalizedF2 = normalizeFeatures(f2);
    
    const vector = [...normalizedF1, ...normalizedF2];*/
    
    // Calculate similarity features using the same method as training
    const allKeys = Object.keys({...f1, ...f2});
    const similarityFeatures = calculateSimilarityFeatures(f1, f2, allKeys);
    
    // Convert to array in same order as training
    const featureVector = FEATURE_KEYS?.similarity_keys?.map(k => similarityFeatures[k] || 0) || 
                         Object.values(similarityFeatures);

    // Call Python script
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