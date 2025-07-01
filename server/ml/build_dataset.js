

const fs = require("fs");
const path = require("path");
const { parseJavaFile, extractFeatures } = require("../logic/javaParser");

const DATASET_PATH = "./IR-Plag-Dataset";
const OUTPUT = [];

function cosineSimilarity(vec1, vec2) {
  const dot = vec1.reduce((sum, val, i) => sum + val * vec2[i], 0);
  const mag1 = Math.sqrt(vec1.reduce((sum, val) => sum + val * val, 0));
  const mag2 = Math.sqrt(vec2.reduce((sum, val) => sum + val * val, 0));
  return dot / (mag1 * mag2 + 1e-9);
}

function toVector(features) {
  return [
    features.total_nodes,
    features.tree_depth,
    features.num_classes,
    features.num_methods,
    features.num_if,
    features.num_for,
    features.num_while,
  ];
}

function extractJavaFile(dir) {
  const files = fs.readdirSync(dir).filter(f => f.endsWith(".java"));
  if (files.length === 0) return null;
  return path.join(dir, files[0]);
}

function processPair(originalPath, comparePath, label) {
  const originalAST = parseJavaFile(originalPath);
  const compAST = parseJavaFile(comparePath);
  const originalFeatures = extractFeatures(originalAST);
  const compFeatures = extractFeatures(compAST);

  const vec1 = toVector(originalFeatures);
  const vec2 = toVector(compFeatures);

  const similarity = cosineSimilarity(vec1, vec2);
  OUTPUT.push({ features: vec1.concat(vec2), similarity, label });
}

function processCase(casePath) {
  const originalDir = path.join(casePath, "original");
  const originalFile = extractJavaFile(originalDir);
  if (!originalFile) return;

  // Process plagiarized
  const plagRoot = path.join(casePath, "plagiarized");
  if (fs.existsSync(plagRoot)) {
    const levels = fs.readdirSync(plagRoot);
    levels.forEach(level => {
      const subDir = path.join(plagRoot, level);
      fs.readdirSync(subDir).forEach(fileId => {
        const compPath = extractJavaFile(path.join(subDir, fileId));
        if (compPath) processPair(originalFile, compPath, 1); 
      });
    });
  }

  // Process non-plagiarized
  const nonPlagDir = path.join(casePath, "non-plagiarized");
  if (fs.existsSync(nonPlagDir)) {
    fs.readdirSync(nonPlagDir).forEach(sub => {
      const compPath = extractJavaFile(path.join(nonPlagDir, sub));
      if (compPath) processPair(originalFile, compPath, 0); 
    });
  }
}

function run() {
  const cases = fs.readdirSync(DATASET_PATH);
  cases.forEach(caseDir => {
    const casePath = path.join(DATASET_PATH, caseDir);
    if (fs.statSync(casePath).isDirectory()) {
      console.log("Processing:", caseDir);
      processCase(casePath);
    }
  });

  // Save dataset to json
  fs.writeFileSync("dataset.json", JSON.stringify(OUTPUT, null, 2));
  console.log("Saved dataset.json with", OUTPUT.length, "examples");
}

run();
