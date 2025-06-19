const path = require('path');
const { parseJavaFile, extractFeatures } = require('./logic/javaParser');

const filePath = path.join(__dirname, './uploads/Test1.java');
const result = parseJavaFile(filePath);
console.log(JSON.stringify(result, null, 2));
