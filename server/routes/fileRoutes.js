const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { parseJavaFile, extractFeatures } = require('../logic/javaParser');

const router = express.Router();
const upload = multer({ dest: 'uploads/' });

router.post('/predict', upload.fields([{ name: 'original' }, { name: 'suspect' }]), async (req, res) => {
  try {
    const originalPath = req.files['original'][0].path;
    const suspectPath = req.files['suspect'][0].path;

    // Parse and extract features
    const ast1 = parseJavaFile(originalPath);
    const ast2 = parseJavaFile(suspectPath);
    const f1 = extractFeatures(ast1);
    const f2 = extractFeatures(ast2);

    const vector = [
      f1.total_nodes, f1.tree_depth, f1.num_classes, f1.num_methods, f1.num_if, f1.num_for, f1.num_while,
      f2.total_nodes, f2.tree_depth, f2.num_classes, f2.num_methods, f2.num_if, f2.num_for, f2.num_while,
    ];

    // Call Python script
    const py = spawn('python', ['model/predict_model.py']);
    let output = '';

    py.stdin.write(JSON.stringify({ features: vector }));
    py.stdin.end();

    py.stdout.on('data', (data) => {
      output += data.toString();
    });

    py.stderr.on('data', (data) => {
      console.error(`Python error: ${data}`);
    });

    py.on('close', (code) => {
      try {
        const result = JSON.parse(output);
        res.json({ success: true, ...result });
      } catch (e) {
        console.error('Failed to parse Python output', e);
        res.status(500).json({ success: false, error: 'Failed to parse prediction' });
      }
    });

  } catch (err) {
    console.error(err);
    res.status(500).json({ success: false, error: 'Server error' });
  }
});

module.exports = router;
