const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { parseJavaFile, extractFeatures } = require('../logic/JavaParser2.js');

const router = express.Router();
const upload = multer({ dest: 'uploads/' });

router.post('/predict', upload.fields([{ name: 'original' }, { name: 'suspect' }]), async (req, res) => {
  try {
    const originalPath = req.files['original'][0].path;
    const suspectPath = req.files['suspect'][0].path;

    console.log('Parsing files:', originalPath, suspectPath);

    const ast1 = parseJavaFile(originalPath);
    const ast2 = parseJavaFile(suspectPath);
    const f1 = extractFeatures(ast1);
    const f2 = extractFeatures(ast2);
    
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

module.exports = router;