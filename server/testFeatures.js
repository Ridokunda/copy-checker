const path = require('path');
const { parseJavaFile, extractFeatures, tokenize, Parser } = require('./logic/JavaParser2');

const filePath = path.join(__dirname, './uploads/Test1.java'); 
const ast = parseJavaFile(filePath);
const features = extractFeatures(ast);

console.log(JSON.stringify(features, null, 2));
