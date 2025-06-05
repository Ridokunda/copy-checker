// server/testParser.js
const parseJavaFile = require('./logic/javaParser');

const result = parseJavaFile('./uploads/Test1.java');
console.log(JSON.stringify(result, null, 2));
