const fs = require('fs');
const path = require('path');
const { parseJavaFile, extractFeatures, tokenize } = require('../logic/JavaParser2.js');

const BASE_DIR = './IR-Plag-Dataset';
const OUTPUT_FILE = './dataset.json';

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

// Collect all possible keys across the dataset to normalize vectors
function getAllFeatureKeys(dataset) {
  const keySet = new Set();
  dataset.forEach((item) => {
    Object.keys(item).forEach((k) => keySet.add(k));
  });
  return Array.from(keySet);
}

// Convert a feature object to a vector
function toVector(featureObj, keys) {
  return keys.map((k) => featureObj[k] || 0);
}

function buildDataset() {
  const rawDataset = [];
  const dataset = [];
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
      const originalCode = fs.readFileSync(orig, 'utf-8');
     
      const originalTokens = tokenize(originalCode);
      
      const origAst = parseJavaFile(orig);
      const origFeatures = extractFeatures(origAst, originalTokens);

      for (const plag of plagiarized) {
        const plagCode = fs.readFileSync(plag, 'utf-8');
        const plagTokens = tokenize(plagCode);
        const plagAst = parseJavaFile(plag);
        const plagFeatures = extractFeatures(plagAst, plagTokens);
        rawDataset.push({ features1: origFeatures, features2: plagFeatures, label: 1 });
      }

      for (const nonPlag of nonPlagiarized) {
        const nonCode = fs.readFileSync(nonPlag, 'utf-8');
        const nonTokens = tokenize(nonCode);
        const nonAst = parseJavaFile(nonPlag);
        const nonFeatures = extractFeatures(nonAst, nonTokens);
        rawDataset.push({ features1: origFeatures, features2: nonFeatures, label: 0 });
      }
    }
  }

  const allKeys = getAllFeatureKeys(
    rawDataset.flatMap(({ features1, features2 }) => [features1, features2])
  );

  // Convert to vectors
  rawDataset.forEach(({ features1, features2, label }) => {
    const vec1 = toVector(features1, allKeys);
    const vec2 = toVector(features2, allKeys);
    dataset.push({ features: [...vec1, ...vec2], label });
  });

  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(dataset, null, 2));
  console.log(`Dataset built: ${dataset.length} examples saved to ${OUTPUT_FILE}`);
}

buildDataset();
