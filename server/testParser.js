const path = require('path');
const { parseJavaFile } = require('./logic/JavaParser2');

const filePath = path.join(__dirname, './uploads/Test1.java');
const result = parseJavaFile(filePath);
console.log(JSON.stringify(result, null, 2));
