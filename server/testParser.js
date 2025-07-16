const path = require('path');
const { parseJavaFile, extractFeatures } = require('./logic/JavaParser2.js');

const filePath = path.join(__dirname, './uploads/T4.java');
const result = parseJavaFile(filePath);
console.log(JSON.stringify(result, null, 2));
