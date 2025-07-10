const express = require('express');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { parseJavaFile, extractFeatures } = require('../logic/JavaParser2');

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
      f1.num_classes, f1.num_methods, f1.num_if, f1.num_for, f1.num_while, f1.num_return,
      f1.num_imports, f1.num_package, f1.num_expressions, f1.num_statements, f1.avg_method_length, f1.max_depth,
      f2.num_classes, f2.num_methods, f2.num_if, f2.num_for, f2.num_while, f2.num_return,
      f2.num_imports, f2.num_package, f2.num_expressions, f2.num_statements, f2.avg_method_length, f2.max_depth
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
