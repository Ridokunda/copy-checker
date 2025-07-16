const { parseJavaFile, extractFeatures, tokenize, Parser } = require('./logic/JavaParser2');
const path = require('path');
const fs = require('fs');
const filePath = path.join(__dirname, './uploads/Test1.java');

const code = fs.readFileSync(filePath, "utf-8");
const tokens = tokenize(code);

console.log(tokens);