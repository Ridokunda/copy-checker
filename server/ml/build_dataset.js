const fs = require('fs');
const path = require('path');
const { parseJavaFile, extractFeatures } = require('../logic/JavaParser2.js');

const BASE_DIR = './IR-Plag-Dataset';
const OUTPUT_FILE = './dataset.json';

// Collect all keys across the dataset to ensure consistent vector length
function collectAllKeys(featureList) {
  const allKeys = new Set();
  featureList.forEach(f => Object.keys(f).forEach(k => allKeys.add(k)));
  return Array.from(allKeys);
}

// Convert a feature object into a vector using a global list of keys
function toVector(features, keys) {
  return keys.map(k => features[k] || 0);
}

function collectJavaFiles(dirPath) {
  let files = [];
  const items = fs.readdirSync(dirPath);
  for (const item of items) {
    const fullPath = path.join(dirPath, item);
    if (fs.statSync(fullPath).isDirectory()) {
      files = files.concat(collectJavaFiles(fullPath));
    } else if (fullPath.endsWith('.java')) {
      files.push(fullPath);
    }
  }
  return files;
}

function buildDataset() {
  const dataset = [];
  const featuresList = []; // collect feature objects
  const cases = fs.readdirSync(BASE_DIR);

  for (const caseFolder of cases) {
    const casePath = path.join(BASE_DIR, caseFolder);
    const originalPath = path.join(casePath, 'original');
    const plagPath = path.join(casePath, 'plagiarized');
    const nonPlagPath = path.join(casePath, 'non-plagiarized');

    const originals = collectJavaFiles(originalPath);
    const plagiarized = collectJavaFiles(plagPath);
    const nonPlagiarized = collectJavaFiles(nonPlagPath);

    for (const orig of originals) {
      const origAst = parseJavaFile(orig);
      const origFeatures = extractFeatures(origAst);

      for (const plag of plagiarized) {
        const plagAst = parseJavaFile(plag);
        const plagFeatures = extractFeatures(plagAst);
        dataset.push({ f1: origFeatures, f2: plagFeatures, label: 1 });
        featuresList.push(origFeatures, plagFeatures);
      }

      for (const nonPlag of nonPlagiarized) {
        const nonAst = parseJavaFile(nonPlag);
        const nonFeatures = extractFeatures(nonAst);
        dataset.push({ f1: origFeatures, f2: nonFeatures, label: 0 });
        featuresList.push(origFeatures, nonFeatures);
      }
    }
  }

  // Create consistent vector keys
  const allKeys = collectAllKeys(featuresList);

  // Convert dataset to numeric vectors
  const final = dataset.map(item => ({
    features: [...toVector(item.f1, allKeys), ...toVector(item.f2, allKeys)],
    label: item.label
  }));

  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(final, null, 2));
  console.log(`✅ Dataset built with ${final.length} samples and ${allKeys.length * 2} features → ${OUTPUT_FILE}`);
}

buildDataset();
