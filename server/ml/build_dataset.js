// Enhanced feature vector construction for plagiarism detection
const fs = require('fs');
const path = require('path');
const { parseJavaFile, extractFeatures, tokenize, Parser } = require('../logic/JavaParser2.js');

const BASE_DIR = './IR-Plag-Dataset';
const OUTPUT_FILE = './dataset.json';
const FEATURE_KEYS_FILE = './feature_keys.json';

// Calculate similarity metrics between two feature vectors
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

// Normalize feature vectors
function normalizeFeatures(features, keys) {
    const stats = {};
    
    // Calculate mean and std for each feature
    keys.forEach(key => {
        const values = features.map(f => f[key] || 0);
        const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
        const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
        const std = Math.sqrt(variance);
        stats[key] = { mean, std: std > 0 ? std : 1 }; // Avoid division by zero
    });
    
    // Normalize each feature vector
    return features.map(f => {
        const normalized = {};
        keys.forEach(key => {
            const value = f[key] || 0;
            normalized[key] = (value - stats[key].mean) / stats[key].std;
        });
        return normalized;
    });
}

function collectAllKeys(featureList) {
    const allKeys = new Set();
    featureList.forEach(f => Object.keys(f).forEach(k => allKeys.add(k)));
    return Array.from(allKeys).sort();
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
    const featuresList = [];
    const cases = fs.readdirSync(BASE_DIR);

    console.log('📂 Scanning dataset directories...');

    // First pass: collect all features
    for (const caseFolder of cases) {
        const casePath = path.join(BASE_DIR, caseFolder);
        if (!fs.statSync(casePath).isDirectory()) continue;
        
        const originalPath = path.join(casePath, 'original');
        const plagPath = path.join(casePath, 'plagiarized');
        const nonPlagPath = path.join(casePath, 'non-plagiarized');

        if (!fs.existsSync(originalPath)) {
            console.log(`⚠️  Skipping ${caseFolder}: no 'original' directory`);
            continue;
        }

        const originals = collectJavaFiles(originalPath);
        const plagiarized = fs.existsSync(plagPath) ? collectJavaFiles(plagPath) : [];
        const nonPlagiarized = fs.existsSync(nonPlagPath) ? collectJavaFiles(nonPlagPath) : [];

        console.log(`📁 Processing ${caseFolder}: ${originals.length} original, ${plagiarized.length} plagiarized, ${nonPlagiarized.length} non-plagiarized`);

        for (const orig of originals) {
            try {
                const origAst = parseJavaFile(orig);
                const origFeatures = extractFeatures(origAst);
                featuresList.push(origFeatures);

                // Process plagiarized files
                for (const plag of plagiarized) {
                    try {
                        const plagAst = parseJavaFile(plag);
                        const plagFeatures = extractFeatures(plagAst);
                        featuresList.push(plagFeatures);
                        dataset.push({ f1: origFeatures, f2: plagFeatures, label: 1 });
                    } catch (err) {
                        console.error(`❌ Error processing ${plag}:`, err.message);
                    }
                }

                // Process non-plagiarized files
                for (const nonPlag of nonPlagiarized) {
                    try {
                        const nonAst = parseJavaFile(nonPlag);
                        const nonFeatures = extractFeatures(nonAst);
                        featuresList.push(nonFeatures);
                        dataset.push({ f1: origFeatures, f2: nonFeatures, label: 0 });
                    } catch (err) {
                        console.error(`❌ Error processing ${nonPlag}:`, err.message);
                    }
                }
            } catch (err) {
                console.error(`❌ Error processing ${orig}:`, err.message);
            }
        }
    }

    if (dataset.length === 0) {
        console.error('❌ No valid samples found in dataset!');
        return;
    }

    // Get all unique feature keys
    const allKeys = collectAllKeys(featuresList);
    console.log(`🔑 Found ${allKeys.length} unique features`);

    // Normalize features
    console.log('🔄 Normalizing features...');
    const normalizedFeatures = normalizeFeatures(featuresList, allKeys);
    
    // Create feature map for quick lookup
    const featureMap = new Map();
    featuresList.forEach((original, index) => {
        featureMap.set(original, normalizedFeatures[index]);
    });

    // Build final dataset with similarity features
    console.log('🔄 Computing similarity features...');
    const finalDataset = dataset.map(item => {
        const f1Normalized = featureMap.get(item.f1);
        const f2Normalized = featureMap.get(item.f2);
        
        // Calculate similarity features
        const similarities = calculateSimilarityFeatures(f1Normalized, f2Normalized, allKeys);
        
        return {
            features: Object.values(similarities),
            label: item.label
        };
    });

    // Save feature keys (similarity feature names)
    const similarityKeys = Object.keys(calculateSimilarityFeatures(
        normalizedFeatures[0], 
        normalizedFeatures[1], 
        allKeys
    ));
    
    fs.writeFileSync(FEATURE_KEYS_FILE, JSON.stringify({
        original_keys: allKeys,
        similarity_keys: similarityKeys,
        feature_count: similarityKeys.length
    }, null, 2));

    // Calculate dataset statistics
    const plagiarizedCount = finalDataset.filter(item => item.label === 1).length;
    const nonPlagiarizedCount = finalDataset.filter(item => item.label === 0).length;

    console.log(`📊 Dataset Statistics:`);
    console.log(`   Total samples: ${finalDataset.length}`);
    console.log(`   Plagiarized: ${plagiarizedCount} (${(plagiarizedCount/finalDataset.length*100).toFixed(1)}%)`);
    console.log(`   Non-plagiarized: ${nonPlagiarizedCount} (${(nonPlagiarizedCount/finalDataset.length*100).toFixed(1)}%)`);
    console.log(`   Features per sample: ${similarityKeys.length}`);
    console.log(`   Original features: ${allKeys.length}`);
    console.log(`   Similarity features: ${similarityKeys.length}`);

    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(finalDataset, null, 2));
    console.log(`✅ Dataset built → ${OUTPUT_FILE}`);
    console.log(`💾 Feature keys saved to ${FEATURE_KEYS_FILE}`);
}

buildDataset();