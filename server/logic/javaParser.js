const fs = require('fs');

// Define simple keywords to look for
const BLOCK_KEYWORDS = ['class', 'if', 'else', 'for', 'while', 'switch', 'try', 'catch', 'method'];

function parseJavaFile(filePath) {
  const code = fs.readFileSync(filePath, 'utf-8');
  const lines = code.split('\n');

  const ast = [];
  const stack = [];

  lines.forEach((rawLine, index) => {
    const line = rawLine.trim();

    // Skip empty lines and comments
    if (line === '' || line.startsWith('//') || line.startsWith('*')) return;

    let node = null;

    if (line.startsWith('class ')) {
      node = { type: 'class', name: extractName(line, 'class'), children: [], line: index + 1 };
    } else if (line.match(/(public|private|protected)?\s*(static)?\s*\w+\s+\w+\(.*\)\s*\{/)) {
      node = { type: 'method', name: extractMethodName(line), children: [], line: index + 1 };
    } else if (line.startsWith('if')) {
      node = { type: 'if', line: index + 1, children: [] };
    } else if (line.startsWith('else')) {
      node = { type: 'else', line: index + 1, children: [] };
    } else if (line.startsWith('for')) {
      node = { type: 'for', line: index + 1, children: [] };
    } else if (line.startsWith('while')) {
      node = { type: 'while', line: index + 1, children: [] };
    }

    if (node) {
      if (stack.length > 0) {
        stack[stack.length - 1].children.push(node);
      } else {
        ast.push(node);
      }
      stack.push(node);
    }

    // Close block on brace
    if (line.endsWith('}')) {
      stack.pop();
    }
  });

  return ast;
}

// Extract class or method name
function extractName(line, keyword) {
  const parts = line.split(' ');
  const index = parts.indexOf(keyword);
  return parts[index + 1]?.replace('{', '');
}

function extractMethodName(line) {
  const match = line.match(/\w+\s+(\w+)\(.*\)/);
  return match ? match[1] : 'anonymous';
}

module.exports = parseJavaFile;
