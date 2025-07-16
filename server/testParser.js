const path = require('path');
const fs = require('fs');
const { parseJavaFile, extractFeatures } = require('./logic/JavaParser2.js');

const filePath = path.join(__dirname, './uploads/Test1.java');
const outputPath = path.join(__dirname, './output/result.json');

// Parse the Java file
const result = parseJavaFile(filePath);

// Create output directory if it doesn't exist
const outputDir = path.dirname(outputPath);
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

// Write result to file
fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
console.log(`Result successfully written to ${outputPath}`);
