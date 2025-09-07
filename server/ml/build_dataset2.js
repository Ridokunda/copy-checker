const fs = require('fs');
const path = require('path');
const { parseJavaFile, extractFeatures, tokenize, getTokenMap,linearizeAST, levenshteinDistanceAST,
  levenshteinSimilarityAST, astLevenshteinDistance, astLevenshteinSimilarity } = require('../logic/JavaParser2.js');

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

      const origTokenMap = getTokenMap(originalTokens);

      
      const origAst = parseJavaFile(orig);
      const origFeatures = extractFeatures(origAst, originalTokens.length);
      origFeatures['num_unique_tokens'] = origTokenMap.size;

      for (const plag of plagiarized) {
        const plagCode = fs.readFileSync(plag, 'utf-8');
        const plagTokens = tokenize(plagCode);
        const plagTokenMap = getTokenMap(plagTokens);
         
        //token overlap
        let overlapCount = 0; 
        for (const [token, count] of origTokenMap.entries()) {
          if (plagTokenMap.has(token)) {
            overlapCount += Math.min(count, plagTokenMap.get(token));
          }
        }
        origFeatures['token_overlap'] = overlapCount;    

        const plagAst = parseJavaFile(plag);
        const plagFeatures = extractFeatures(plagAst, plagTokens.length);
        plagFeatures['num_unique_tokens'] = plagTokenMap.size;
        rawDataset.push({ features1: origFeatures, features2: plagFeatures, label: 1 });

        // Compute AST-based similarity metrics
        const astLevDist = astLevenshteinDistance(orig, plag);
        const astLevSim = astLevenshteinSimilarity(orig, plag);

        plagFeatures['ast_levenshtein_distance'] = astLevDist;
        plagFeatures['ast_levenshtein_similarity'] = astLevSim;
      }

      for (const nonPlag of nonPlagiarized) {
        const nonCode = fs.readFileSync(nonPlag, 'utf-8');
        const nonTokens = tokenize(nonCode);
        const nonTokenMap = getTokenMap(nonTokens);

        //token overlap
        let overlapCount = 0;
        for (const [token, count] of origTokenMap.entries()) {
          if (nonTokenMap.has(token)) {
            overlapCount += Math.min(count, nonTokenMap.get(token));
          }
        }
        origFeatures['token_overlap'] = overlapCount;
        

        const nonAst = parseJavaFile(nonPlag);
        const nonFeatures = extractFeatures(nonAst, nonTokens.length);
        nonFeatures['num_unique_tokens'] = nonTokenMap.size;
        rawDataset.push({ features1: origFeatures, features2: nonFeatures, label: 0 });

        // Compute AST-based similarity metrics        
        const astLevDist = astLevenshteinDistance(orig, nonPlag);
        const astLevSim = astLevenshteinSimilarity(orig, nonPlag);

        nonFeatures['ast_levenshtein_distance'] = astLevDist;
        nonFeatures['ast_levenshtein_similarity'] = astLevSim;
      }
    }
  }
  
  // Convert to vectors
  rawDataset.forEach(({ features1, features2, label }) => {
    const vec1 = Object.values(features1);
    const vec2 = Object.values(features2);
    dataset.push({ features: [...vec1, ...vec2], label });
  });

  fs.writeFileSync(OUTPUT_FILE, JSON.stringify(dataset, null, 2));
  console.log(`Dataset built: ${dataset.length} examples saved to ${OUTPUT_FILE}`);
}
//final feature vector looks like
/*
final vector is:
  [ features: [
    num_tokens (orig),
    num_methods (orig),
    num_if (orig),
    num_for (orig),
    num_while (orig),
    num_return (orig),
    num_imports (orig),
    num_package (orig),
    num_expressions (orig),
    num_statements (orig),
    num_systemcall (orig),
    num_javacall (orig),
    num_variables (orig),
    num_var_declarations (orig),
    num_method_calls (orig),
    max_depth (orig),
    avg_method_length (orig),
    num_unique_tokens (orig),
    token_overlap (orig),
    num_tokens (other),
    num_methods (other),
    num_if (other),
    num_for (other),
    num_while (other),
    num_return (other),
    num_imports (other),
    num_package (other),
    num_expressions (other),
    num_statements (other),
    num_systemcall (other),
    num_javacall (other),
    num_variables (other),
    num_var_declarations (other),
    num_method_calls (other),
    max_depth (other),
    avg_method_length (other),
    num_unique_tokens (other),
    ast_levenshtein_distance (other),
    ast_levenshtein_similarity (other)
    ],
    label:[1 or 0]
  ]
*/

buildDataset();
