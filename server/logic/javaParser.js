const fs = require('fs');




function parseJavaFile(filePath) {
  const code = fs.readFileSync(filePath, 'utf-8');
  const lines = code.split('\n');

  const ast = [];
  const stack = [];

  lines.forEach((rawLine, index) => {
    const line = rawLine.trim();

    if (line === '' || line.startsWith('//') || line.startsWith('*')) return;

    let node = null;

    if (line.startsWith('class ') || line.startsWith('public class ')) {
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

    
    if (line.endsWith('}')) {
      stack.pop();
    }
  });

  return ast;
}


function extractName(line, keyword) {
  const parts = line.split(' ');
  const index = parts.indexOf(keyword);
  return parts[index + 1]?.replace('{', '');
}

function extractMethodName(line) {
  const match = line.match(/\w+\s+(\w+)\(.*\)/);
  return match ? match[1] : 'anonymous';
}

function extractFeatures(ast) {
  let totalNodes = 0;
  let typeCounts = {};
  let preorder = [];

  let maxDepth = 0;

  function traverse(node, depth) {
    totalNodes++;
    preorder.push(node.type);
    typeCounts[node.type] = (typeCounts[node.type] || 0) + 1;

    maxDepth = Math.max(maxDepth, depth);

    if (node.children) {
      node.children.forEach(child => traverse(child, depth + 1));
    }
  }

  ast.forEach(node => traverse(node, 1));

  return {
    total_nodes: totalNodes,
    tree_depth: maxDepth,
    num_classes: typeCounts.class || 0,
    num_methods: typeCounts.method || 0,
    num_if: typeCounts.if || 0,
    num_for: typeCounts.for || 0,
    num_while: typeCounts.while || 0,
    node_type_freq: typeCounts,
    preorder_sequence: preorder
  };
}
module.exports = {
  parseJavaFile,
  extractFeatures
};

