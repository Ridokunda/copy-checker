// Recursive Descent Parser for a Subset of Java - Returns AST
// Focused on: classes, methods, if/for/while blocks, import/package declarations

const fs = require("fs");

// --- Tokenizer ---
function tokenize(code) {
  code = code.replace(/\/\/.*|\/\*[\s\S]*?\*\//g, '');

  const tokenPattern = /\b(package|import|class|public|private|protected|void|int|final|boolean|char|byte|short|long|float|double|String|if|for|while|static|return|try|catch|new|throws|throw)\b(?:\[\])*|(\w+(?:\[\])*)|(\d+\.\d+)|(\+\+|--|<=|>=|==|!=)|(?:"(?:\\.|[^"\\])*")|(?:'(?:\\.|[^'\\])*')|\{|\}|\(|\)|\.|;|,|[a-zA-Z_][a-zA-Z0-9_]*|\S/g;

  const tokens = [];
  let match;
  while ((match = tokenPattern.exec(code)) !== null) {
    tokens.push(match[0]);
  }
  return tokens;
}


// --- Parser Class ---
class Parser {
  constructor(tokens) {
    this.tokens = tokens;
    this.pos = 0;
  }

  current() {
    return this.tokens[this.pos];
  }

  next() {
    this.pos++;
    return this.current();
  }

  match(...expected) {
    const token = this.current();
    if (expected.includes(token)) {
      this.next();
      return true;
    }
    return false;
  }

  parse() {
    const ast = { type: "Program", body: [] };
    console.log(`Starting parse at position ${this.pos}`);
    while (this.match("package", "import")) {
      const keyword = this.tokens[this.pos - 1];
      const value = this.parseQualifiedName();
      if (this.match(";")) {
        ast.body.push({ type: keyword === "package" ? "PackageDeclaration" : "ImportDeclaration", value });
      }
    }
    console.log(`starting to parse class`);
    while (this.pos < this.tokens.length) {
      const classNode = this.parseClass();
      if (classNode) ast.body.push(classNode);
      else this.next();
    }
    return ast;
  }

  parseQualifiedName() {
    const parts = [];
    while (/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(this.current())) {
      parts.push(this.current());
      this.next();
      if (!this.match(".")) break;
    }
    return parts.join(".");
  }

  parseClass() {
    this.match("public", "private", "protected");
    if (!this.match("class")) return null;
    const name = this.current();
    this.next();
    if (!this.match("{")) return null;

    const classNode = {
      type: "ClassDeclaration",
      name,
      body: []
    };

    while (!this.match("}")) {
      if (this.pos >= this.tokens.length) break;
      const member = this.parseMethodOrField();
      if (member) classNode.body.push(member);
    }

    return classNode;
  }

  parseMethodOrField() {
    const start = this.pos;
    const modifiers = [];
    console.log(`Starting to parse method or field`);
    //collect modifiers
    while (this.match("public", "private", "protected", "static", "final", "synchronized", "volatile", "transient")) {
      modifiers.push(this.tokens[this.pos - 1]);
    }
    //if it is a method
    if (this.match("void", "int", "String", "boolean", "double", "float", "char", "byte", "short", "long")) {
      const returnType = this.tokens[this.pos - 1];
      const name = this.current();
      this.next();

      if (this.match("(")){
        const params = [];
        while (!this.match(")")) {
          console.log(`Starting to parse parameters`);
          if (this.pos >= this.tokens.length) break;
          const paramType = this.current();
          if (["final"].includes(paramType)) {
            this.next();
            continue;
          }
          this.next();
          const paramName = this.current();
          this.next();
          params.push({ type: paramType, name: paramName });
          this.match(",");
        }
        console.log(params);
        if (!this.match("{")) return null;
        const body = this.parseBlock();
        return {
          type: "MethodDeclaration",
          modifiers,
          returnType,
          name,
          params,
          body
        };
      }
    }  
    //if is field declaration
    this.pos = start;
    const varDecl = this.parseVariableDeclaration();
    if (varDecl) {
      if (modifiers.length > 0) {
        varDecl.modifiers = modifiers;
      }
      return varDecl;
    }

    this.pos = start;
    return this.parseStatement();
  }

  parseVariableDeclaration() {
    const startPos = this.pos;
    const isFinal = this.match("final");
    const type = this.current();
    
    if (!["int", "String", "boolean", "double", "float", "char", "byte", "short", "long"].includes(type)) {
      return null;
    }
    this.next();
    
    const declarations = [];
    let name = this.current();
    this.next();
    
    let value = null;
    if (this.match("=")) {
      value = this.parseExpression();
    }
    
    declarations.push({ name, value });
    
    // Handle multiple variables declared together 
    while (this.match(",")) {
      name = this.current();
      this.next();
      value = null;
      if (this.match("=")) {
        value = this.parseExpression();
      }
      declarations.push({ name, value });
    }
    
    this.match(";");
    
    return {
      type: "VariableDeclaration",
      kind: isFinal ? "final" : "typed",
      dataType: type,
      declarations
    };
  }

  parseExpression() {
    const tokens = [];
    while (![",", ";", ")", "}", "]"].includes(this.current())) {
      if (this.pos >= this.tokens.length) break;
      tokens.push(this.current());
      this.next();
    }
    return tokens.length > 0 ? tokens.join(" ") : null;
  }
  

  parseStatement() {
    const varDecl = this.parseVariableDeclaration();
    if (varDecl) return varDecl;

     if (this.match("if")) {
      const test = this.parseCondition();
      const consequent = this.parseBlock();
      let alternate = null;
      if (this.match("else")) {
        if (this.current() === "if") {
          alternate = this.parseStatement();
        } else {
          alternate = this.parseBlock();
        }
      }
      return { type: "IfStatement", test, consequent, alternate };
    }
    if (this.match("for")) {
      const test = this.parseCondition();
      const body = this.parseBlock();
      return { type: "ForStatement", test, body };
    }
    if (this.match("while")) {
      const test = this.parseCondition();
      const body = this.parseBlock();
      return { type: "WhileStatement", test, body };
    }
    if (this.match("return")) {
      let value = null;
      if (![";", "}"].includes(this.current())) {
        value = this.parseExpression();
      }
      this.match(";");
      return { type: "ReturnStatement", value };
    }
    if (this.match("{")) {
      return this.parseBlock();
    }

    const expression = this.parseExpression();
    this.match(";");
    return expression ? { type: "ExpressionStatement", expression } : null;
  }

  parseBlock() {
    console.log(`Starting to parse block`);
    const body = [];
    while (!this.match("}")) {
      if (this.pos >= this.tokens.length) break;
      const stmt = this.parseStatement();
      if (stmt) body.push(stmt);
    }
    return { type: "BlockStatement", body };
  }

  parseCondition() {
    const tokens = [];
    if (!this.match("(")) return null;
    while (!this.match(")")) {
      if (this.pos >= this.tokens.length) break;
      tokens.push(this.current());
      this.next();
    }
    return { type: "Condition", tokens };
  }
}

// --- Main Parse Function ---
function parseJavaFile(filepath) {
  const code = fs.readFileSync(filepath, "utf-8");
  const tokens = tokenize(code);
  const parser = new Parser(tokens);
  return parser.parse();
}

// --- Feature Extraction ---
function extractFeatures(ast) {
  const stats = {
    num_classes: 0,
    num_methods: 0,
    num_if: 0,
    num_for: 0,
    num_while: 0,
    num_return: 0,
    num_imports: 0,
    num_package: 0,
    num_expressions: 0,
    num_statements: 0,
    num_var_declarations: 0,
    total_method_lengths: 0,
    max_depth: 0
  };

  const sequence = []; // collect node types
  let methodCount = 0;

  function traverse(node, depth = 0) {
    if (!node || typeof node !== 'object') return;
    if (node.type) sequence.push(node.type);
    if (depth > stats.max_depth) stats.max_depth = depth;

    switch (node.type) {
      case "ClassDeclaration": stats.num_classes++; break;
      case "MethodDeclaration":
        stats.num_methods++;
        methodCount++;
        if (node.body && node.body.body) {
          stats.total_method_lengths += node.body.body.length;
        }
        break;
      case "IfStatement": stats.num_if++; stats.num_statements++; break;
      case "ForStatement": stats.num_for++; stats.num_statements++; break;
      case "WhileStatement": stats.num_while++; stats.num_statements++; break;
      case "ReturnStatement": stats.num_return++; stats.num_statements++; break;
      case "ImportDeclaration": stats.num_imports++; break;
      case "PackageDeclaration": stats.num_package++; break;
      case "ExpressionStatement": stats.num_expressions++; stats.num_statements++; break;
    }

    for (const key in node) {
      const child = node[key];
      if (Array.isArray(child)) child.forEach(n => traverse(n, depth + 1));
      else if (typeof child === 'object') traverse(child, depth + 1);
    }
  }

  traverse(ast);
  stats.avg_method_length = methodCount > 0 ? stats.total_method_lengths / methodCount : 0;
  delete stats.total_method_lengths;

  // --- n-gram extraction ---
  function generateNGrams(seq, n) {
    const grams = {};
    for (let i = 0; i <= seq.length - n; i++) {
      const gram = seq.slice(i, i + n).join('_');
      grams[gram] = (grams[gram] || 0) + 1;
    }
    return grams;
  }

  function topKGrams(grams, k = 10) {
    return Object.entries(grams)
      .sort((a, b) => b[1] - a[1])
      .slice(0, k)
      .reduce((acc, [k, v]) => {
        acc[`ngram_${k}`] = v;
        return acc;
      }, {});
  }

  const ngrams2 = topKGrams(generateNGrams(sequence, 2), 10);
  const ngrams3 = topKGrams(generateNGrams(sequence, 3), 10);
  Object.assign(stats, ngrams2, ngrams3);

  return stats;
}


module.exports = {
  parseJavaFile,
  extractFeatures,
  tokenize
};
