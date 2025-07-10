const fs = require('fs');
const path = require('path');
const { parseJavaFile, extractFeatures } = require('../logic/JavaParser2');

const BASE_DIR = '../../IR-Plag-Dataset';
const OUTPUT_FILE = '../../dataset.json';

function toVector(features) {
  return [
    features.num_classes,
    features.num_methods,
    features.num_if,
    features.num_for,
    features.num_while,
    features.num_return,
    features.num_imports,
    features.num_package,
    features.num_expressions,
    features.num_statements,
    features.avg_method_length,
    features.max_depth
  ];
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
      const origVec = toVector(extractFeatures(origAst));

      for (const plag of plagiarized) {
        const plagAst = parseJavaFile(plag);
        const plagVec = toVector(extractFeatures(plagAst));
        dataset.push({ features: [...origVec, ...plagVec], label: 1 });
      }

      for (const nonPlag of nonPlagiarized) {
        const nonAst = parseJavaFile(nonPlag);
        const nonVec = toVector(extractFeatures(nonAst));
        dataset.push({ features: [...origVec, ...nonVec], label: 0 });
      }
    }
  }

  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(dataset, null, 2));
  console.log(`✅ Dataset built: ${dataset.length} examples saved to ${OUTPUT_FILE}`);
}

buildDataset();
