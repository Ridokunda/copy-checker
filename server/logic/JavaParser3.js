// Enhanced Recursive Descent Parser for Java - Returns AST
// Improved tokenization, error handling, and language coverage

const fs = require("fs");

// --- Enhanced Tokenizer ---
function tokenize(code) {
  const tokens = [];
  const tokenPatterns = [
    // Keywords
    { type: 'KEYWORD', regex: /\b(package|import|class|interface|enum|public|private|protected|static|final|abstract|void|int|long|double|float|boolean|char|String|if|else|for|while|do|switch|case|default|break|continue|return|try|catch|finally|throw|throws|new|this|super|extends|implements)\b/ },
    
    // Literals
    { type: 'STRING', regex: /"(?:\\.|[^"\\])*"/ },
    { type: 'CHAR', regex: /'(?:\\.|[^'\\])*'/ },
    { type: 'NUMBER', regex: /\b\d+(\.\d+)?([eE][+-]?\d+)?[fFdDlL]?\b/ },
    { type: 'BOOLEAN', regex: /\b(true|false)\b/ },
    { type: 'NULL', regex: /\bnull\b/ },
    
    // Operators
    { type: 'OPERATOR', regex: /(\+\+|--|==|!=|<=|>=|&&|\|\||<<|>>|>>>|\+=|-=|\*=|\/=|%=|&=|\|=|\^=|<<=|>>=|>>>=|[+\-*/%<>=!&|^~?:])/},
    
    // Punctuation
    { type: 'PUNCTUATION', regex: /[{}()\[\];,.]/ },
    
    // Identifiers
    { type: 'IDENTIFIER', regex: /[a-zA-Z_$][a-zA-Z0-9_$]*/ },
    
    // Whitespace and comments
    { type: 'WHITESPACE', regex: /\s+/ },
    { type: 'COMMENT', regex: /\/\/.*|\/\*[\s\S]*?\*\// },
    
    // Annotations
    { type: 'ANNOTATION', regex: /@[a-zA-Z_$][a-zA-Z0-9_$]*/ },
  ];

  let pos = 0;
  let line = 1;
  let column = 1;

  while (pos < code.length) {
    let matched = false;
    
    for (const pattern of tokenPatterns) {
      const regex = new RegExp(pattern.regex.source, 'g');
      regex.lastIndex = pos;
      const match = regex.exec(code);
      
      if (match && match.index === pos) {
        const value = match[0];
        
        // Skip whitespace and comments for parsing
        if (pattern.type !== 'WHITESPACE' && pattern.type !== 'COMMENT') {
          tokens.push({
            type: pattern.type,
            value,
            line,
            column
          });
        }
        
        // Update position tracking
        const newlines = value.match(/\n/g);
        if (newlines) {
          line += newlines.length;
          column = value.length - value.lastIndexOf('\n');
        } else {
          column += value.length;
        }
        
        pos += value.length;
        matched = true;
        break;
      }
    }
    
    if (!matched) {
      throw new Error(`Unexpected character '${code[pos]}' at line ${line}, column ${column}`);
    }
  }
  
  return tokens;
}

// --- Enhanced Parser Class ---
class Parser {
  constructor(tokens) {
    this.tokens = tokens;
    this.pos = 0;
    this.errors = [];
  }

  current() {
    return this.pos < this.tokens.length ? this.tokens[this.pos] : null;
  }

  peek(offset = 1) {
    const index = this.pos + offset;
    return index < this.tokens.length ? this.tokens[index] : null;
  }

  next() {
    this.pos++;
    return this.current();
  }

  match(...expected) {
    const token = this.current();
    if (token && expected.includes(token.value)) {
      this.next();
      return token;
    }
    return null;
  }

  matchType(...types) {
    const token = this.current();
    if (token && types.includes(token.type)) {
      this.next();
      return token;
    }
    return null;
  }

  expect(value) {
    const token = this.current();
    if (!token || token.value !== value) {
      this.error(`Expected '${value}' but found '${token?.value || 'EOF'}'`);
      return null;
    }
    return this.next();
  }

  error(message) {
    const token = this.current();
    const error = {
      message,
      line: token?.line || 'EOF',
      column: token?.column || 'EOF',
      token: token?.value || 'EOF',
      context: this.getContext()
    };
    this.errors.push(error);
    console.error(`Parse Error: ${message} at line ${error.line}, column ${error.column}`);
    console.error(`Token: '${error.token}', Context: ${error.context}`);
  }

  getContext() {
    const start = Math.max(0, this.pos - 5);
    const end = Math.min(this.tokens.length, this.pos + 5);
    const context = this.tokens.slice(start, end)
      .map((token, index) => {
        const marker = (start + index === this.pos) ? ' >>> ' : ' ';
        return `${marker}${token.value}`;
      })
      .join('');
    return context;
  }

  parse() {
    const ast = { 
      type: "Program", 
      body: [],
      errors: this.errors 
    };

    // Parse package declaration
    if (this.match("package")) {
      const packageDecl = this.parsePackageDeclaration();
      if (packageDecl) ast.body.push(packageDecl);
    }

    // Parse import declarations
    while (this.match("import")) {
      const importDecl = this.parseImportDeclaration();
      if (importDecl) ast.body.push(importDecl);
    }

    // Parse type declarations (classes, interfaces, enums)
    while (this.pos < this.tokens.length) {
      const typeDecl = this.parseTypeDeclaration();
      if (typeDecl) {
        ast.body.push(typeDecl);
      } else {
        this.next(); // Skip unrecognized tokens
      }
    }

    return ast;
  }

  parsePackageDeclaration() {
    const name = this.parseQualifiedName();
    this.expect(";");
    return {
      type: "PackageDeclaration",
      name
    };
  }

  parseImportDeclaration() {
    const isStatic = this.match("static");
    const name = this.parseQualifiedName();
    const isWildcard = this.match("*");
    this.expect(";");
    
    return {
      type: "ImportDeclaration",
      name,
      static: !!isStatic,
      wildcard: !!isWildcard
    };
  }

  parseQualifiedName() {
    const parts = [];
    let identifier = this.matchType("IDENTIFIER");
    
    if (!identifier) {
      this.error("Expected identifier");
      return "";
    }
    
    parts.push(identifier.value);
    
    while (this.match(".")) {
      identifier = this.matchType("IDENTIFIER");
      if (!identifier) {
        this.error("Expected identifier after '.'");
        break;
      }
      parts.push(identifier.value);
    }
    
    return parts.join(".");
  }

  parseTypeDeclaration() {
    const modifiers = this.parseModifiers();
    
    if (this.match("class")) {
      return this.parseClassDeclaration(modifiers);
    } else if (this.match("interface")) {
      return this.parseInterfaceDeclaration(modifiers);
    } else if (this.match("enum")) {
      return this.parseEnumDeclaration(modifiers);
    }
    
    return null;
  }

  parseModifiers() {
    const modifiers = [];
    const modifierKeywords = ["public", "private", "protected", "static", "final", "abstract"];
    
    while (this.current() && modifierKeywords.includes(this.current().value)) {
      modifiers.push(this.current().value);
      this.next();
    }
    
    return modifiers;
  }

  parseClassDeclaration(modifiers) {
    const nameToken = this.matchType("IDENTIFIER");
    if (!nameToken) {
      this.error("Expected class name");
      return null;
    }

    const classNode = {
      type: "ClassDeclaration",
      name: nameToken.value,
      modifiers,
      superClass: null,
      interfaces: [],
      body: []
    };

    // Parse extends clause
    if (this.match("extends")) {
      classNode.superClass = this.parseQualifiedName();
    }

    // Parse implements clause
    if (this.match("implements")) {
      classNode.interfaces.push(this.parseQualifiedName());
      while (this.match(",")) {
        classNode.interfaces.push(this.parseQualifiedName());
      }
    }

    if (!this.expect("{")) return null;

    // Parse class body
    while (!this.match("}")) {
      if (this.pos >= this.tokens.length) {
        this.error("Unexpected end of file in class body");
        break;
      }
      
      const member = this.parseClassMember();
      if (member) classNode.body.push(member);
    }

    return classNode;
  }

  parseInterfaceDeclaration(modifiers) {
    const nameToken = this.matchType("IDENTIFIER");
    if (!nameToken) {
      this.error("Expected interface name");
      return null;
    }

    const interfaceNode = {
      type: "InterfaceDeclaration",
      name: nameToken.value,
      modifiers,
      body: []
    };

    if (!this.expect("{")) return null;

    while (!this.match("}")) {
      if (this.pos >= this.tokens.length) {
        this.error("Unexpected end of file in interface body");
        break;
      }
      
      const member = this.parseInterfaceMember();
      if (member) interfaceNode.body.push(member);
    }

    return interfaceNode;
  }

  parseEnumDeclaration(modifiers) {
    const nameToken = this.matchType("IDENTIFIER");
    if (!nameToken) {
      this.error("Expected enum name");
      return null;
    }

    const enumNode = {
      type: "EnumDeclaration",
      name: nameToken.value,
      modifiers,
      constants: [],
      body: []
    };

    if (!this.expect("{")) return null;

    // Parse enum constants
    while (this.current() && this.current().type === "IDENTIFIER") {
      enumNode.constants.push(this.current().value);
      this.next();
      if (!this.match(",")) break;
    }

    if (this.match(";")) {
      // Parse enum body members
      while (!this.match("}")) {
        if (this.pos >= this.tokens.length) break;
        const member = this.parseClassMember();
        if (member) enumNode.body.push(member);
      }
    } else {
      this.expect("}");
    }

    return enumNode;
  }

  parseClassMember() {
    const modifiers = this.parseModifiers();
    
    // Constructor
    if (this.current() && this.current().type === "IDENTIFIER" && this.peek() && this.peek().value === "(") {
      return this.parseConstructor(modifiers);
    }
    
    // Method or field
    const type = this.parseType();
    if (!type) {
      this.next(); // Skip unrecognized tokens
      return null;
    }
    
    const nameToken = this.matchType("IDENTIFIER");
    if (!nameToken) {
      this.error("Expected member name");
      return null;
    }

    if (this.match("(")) {
      return this.parseMethod(modifiers, type, nameToken.value);
    } else {
      return this.parseField(modifiers, type, nameToken.value);
    }
  }

  parseInterfaceMember() {
    const modifiers = this.parseModifiers();
    const type = this.parseType();
    
    if (!type) {
      this.next();
      return null;
    }
    
    const nameToken = this.matchType("IDENTIFIER");
    if (!nameToken) {
      this.error("Expected member name");
      return null;
    }

    if (this.match("(")) {
      const method = this.parseMethod(modifiers, type, nameToken.value);
      method.abstract = true;
      return method;
    } else {
      return this.parseField(modifiers, type, nameToken.value);
    }
  }

  parseType() {
    const primitiveTypes = ["void", "int", "long", "double", "float", "boolean", "char"];
    
    let type = null;
    
    if (primitiveTypes.includes(this.current()?.value)) {
      type = this.current().value;
      this.next();
    } else if (this.current()?.type === "IDENTIFIER") {
      type = this.parseQualifiedName();
    } else {
      return null;
    }
    
    // Handle array types
    while (this.match("[")) {
      this.expect("]");
      type += "[]";
    }
    
    return type;
  }

  parseConstructor(modifiers) {
    const nameToken = this.current();
    this.next(); // Skip constructor name
    
    const parameters = this.parseParameterList();
    
    let body = null;
    if (this.match("{")) {
      body = this.parseBlock();
    } else {
      this.expect(";");
    }

    return {
      type: "ConstructorDeclaration",
      name: nameToken.value,
      modifiers,
      parameters,
      body
    };
  }

  parseMethod(modifiers, returnType, name) {
    const parameters = this.parseParameterList();
    
    // Consume the closing parenthesis
    if (!this.expect(")")) {
      return null;
    }
    
    // Handle throws clause
    let throwsClause = null;
    if (this.match("throws")) {
      throwsClause = [];
      throwsClause.push(this.parseQualifiedName());
      while (this.match(",")) {
        throwsClause.push(this.parseQualifiedName());
      }
    }
    
    let body = null;
    if (this.match("{")) {
      body = this.parseBlock();
    } else {
      this.expect(";");
    }

    return {
      type: "MethodDeclaration",
      name,
      modifiers,
      returnType,
      parameters,
      throwsClause,
      body
    };
  }

  parseField(modifiers, type, name) {
    let initializer = null;
    
    if (this.match("=")) {
      initializer = this.parseExpression();
    }
    
    this.expect(";");
    
    return {
      type: "FieldDeclaration",
      name,
      modifiers,
      fieldType: type,
      initializer
    };
  }

  parseParameterList() {
    const parameters = [];
    
    // Handle empty parameter list
    if (this.current() && this.current().value === ")") {
      return parameters;
    }
    
    while (this.current() && this.current().value !== ")") {
      if (this.pos >= this.tokens.length) {
        this.error("Unexpected end of file in parameter list");
        break;
      }
      
      const paramType = this.parseType();
      if (!paramType) {
        this.error("Expected parameter type");
        break;
      }
      
      const paramName = this.matchType("IDENTIFIER");
      if (!paramName) {
        this.error("Expected parameter name");
        break;
      }
      
      parameters.push({
        type: "Parameter",
        paramType,
        name: paramName.value
      });
      
      // Check for comma (more parameters) or closing paren
      if (this.current() && this.current().value === ",") {
        this.next(); // consume comma
      } else if (this.current() && this.current().value === ")") {
        break; // end of parameter list
      } else {
        this.error("Expected ',' or ')' in parameter list");
        break;
      }
    }
    
    return parameters;
  }

  parseStatement() {
    if (this.match("if")) {
      return this.parseIfStatement();
    } else if (this.match("for")) {
      return this.parseForStatement();
    } else if (this.match("while")) {
      return this.parseWhileStatement();
    } else if (this.match("do")) {
      return this.parseDoWhileStatement();
    } else if (this.match("switch")) {
      return this.parseSwitchStatement();
    } else if (this.match("return")) {
      return this.parseReturnStatement();
    } else if (this.match("break")) {
      this.expect(";");
      return { type: "BreakStatement" };
    } else if (this.match("continue")) {
      this.expect(";");
      return { type: "ContinueStatement" };
    } else if (this.match("try")) {
      return this.parseTryStatement();
    } else if (this.match("throw")) {
      return this.parseThrowStatement();
    } else if (this.match("{")) {
      return this.parseBlock();
    } else {
      return this.parseExpressionStatement();
    }
  }

  parseIfStatement() {
    this.expect("(");
    const condition = this.parseExpression();
    this.expect(")");
    
    const consequent = this.parseStatement();
    let alternate = null;
    
    if (this.match("else")) {
      alternate = this.parseStatement();
    }
    
    return {
      type: "IfStatement",
      condition,
      consequent,
      alternate
    };
  }

  parseForStatement() {
    this.expect("(");
    
    const init = this.parseExpression();
    this.expect(";");
    
    const condition = this.parseExpression();
    this.expect(";");
    
    const update = this.parseExpression();
    this.expect(")");
    
    const body = this.parseStatement();
    
    return {
      type: "ForStatement",
      init,
      condition,
      update,
      body
    };
  }

  parseWhileStatement() {
    this.expect("(");
    const condition = this.parseExpression();
    this.expect(")");
    
    const body = this.parseStatement();
    
    return {
      type: "WhileStatement",
      condition,
      body
    };
  }

  parseDoWhileStatement() {
    const body = this.parseStatement();
    this.expect("while");
    this.expect("(");
    const condition = this.parseExpression();
    this.expect(")");
    this.expect(";");
    
    return {
      type: "DoWhileStatement",
      body,
      condition
    };
  }

  parseSwitchStatement() {
    this.expect("(");
    const discriminant = this.parseExpression();
    this.expect(")");
    this.expect("{");
    
    const cases = [];
    
    while (!this.match("}")) {
      if (this.match("case")) {
        const test = this.parseExpression();
        this.expect(":");
        const consequent = [];
        
        while (!this.current() || (this.current().value !== "case" && this.current().value !== "default" && this.current().value !== "}")) {
          const stmt = this.parseStatement();
          if (stmt) consequent.push(stmt);
        }
        
        cases.push({
          type: "SwitchCase",
          test,
          consequent
        });
      } else if (this.match("default")) {
        this.expect(":");
        const consequent = [];
        
        while (!this.current() || (this.current().value !== "case" && this.current().value !== "default" && this.current().value !== "}")) {
          const stmt = this.parseStatement();
          if (stmt) consequent.push(stmt);
        }
        
        cases.push({
          type: "SwitchCase",
          test: null,
          consequent
        });
      } else {
        this.next();
      }
    }
    
    return {
      type: "SwitchStatement",
      discriminant,
      cases
    };
  }

  parseReturnStatement() {
    let argument = null;
    
    if (this.current() && this.current().value !== ";") {
      argument = this.parseExpression();
    }
    
    this.expect(";");
    
    return {
      type: "ReturnStatement",
      argument
    };
  }

  parseTryStatement() {
    const block = this.parseBlock();
    const handlers = [];
    let finalizer = null;
    
    while (this.match("catch")) {
      this.expect("(");
      const paramType = this.parseType();
      const paramName = this.matchType("IDENTIFIER");
      this.expect(")");
      const body = this.parseBlock();
      
      handlers.push({
        type: "CatchClause",
        param: {
          type: "Parameter",
          paramType,
          name: paramName ? paramName.value : null
        },
        body
      });
    }
    
    if (this.match("finally")) {
      finalizer = this.parseBlock();
    }
    
    return {
      type: "TryStatement",
      block,
      handlers,
      finalizer
    };
  }

  parseThrowStatement() {
    const argument = this.parseExpression();
    this.expect(";");
    
    return {
      type: "ThrowStatement",
      argument
    };
  }

  parseBlock() {
    const body = [];
    
    while (!this.match("}")) {
      if (this.pos >= this.tokens.length) {
        this.error("Unexpected end of file in block");
        break;
      }
      
      const stmt = this.parseStatement();
      if (stmt) body.push(stmt);
    }
    
    return {
      type: "BlockStatement",
      body
    };
  }

  parseExpressionStatement() {
    // Handle variable declarations that might look like expressions
    if (this.current() && this.current().type === "IDENTIFIER") {
      const lookahead = this.peek();
      if (lookahead && this.current().value !== "this" && this.current().value !== "super") {
        // Check if this might be a variable declaration
        const possibleType = this.current().value;
        if (lookahead.type === "IDENTIFIER" && !["(", ".", "["].includes(lookahead.value)) {
          // This looks like: Type variableName
          return this.parseVariableDeclaration();
        }
      }
    }
    
    const expression = this.parseExpression();
    
    if (!this.expect(";")) {
      // If we can't find a semicolon, this might not be an expression statement
      return null;
    }
    
    return {
      type: "ExpressionStatement",
      expression
    };
  }
  
  parseVariableDeclaration() {
    const type = this.current().value;
    this.next();
    
    const declarations = [];
    
    do {
      const nameToken = this.matchType("IDENTIFIER");
      if (!nameToken) {
        this.error("Expected variable name");
        break;
      }
      
      let initializer = null;
      if (this.match("=")) {
        initializer = this.parseExpression();
      }
      
      declarations.push({
        type: "VariableDeclarator",
        name: nameToken.value,
        initializer
      });
      
    } while (this.match(","));
    
    this.expect(";");
    
    return {
      type: "VariableDeclaration",
      declarationType: type,
      declarations
    };
  }

  parseExpression() {
    // Improved expression parsing with proper parentheses handling
    const tokens = [];
    let parenDepth = 0;
    let bracketDepth = 0;
    let braceDepth = 0;
    
    while (this.current()) {
      const token = this.current();
      
      // Check for stopping conditions before consuming the token
      if (parenDepth === 0 && bracketDepth === 0 && braceDepth === 0) {
        if (token.value === ";" || 
            token.value === ")" || 
            token.value === "}" ||
            token.value === "," ||
            token.value === ":") {
          break;
        }
      }
      
      // Track nesting depth after checking stop conditions
      if (token.value === "(") parenDepth++;
      else if (token.value === ")") {
        parenDepth--;
        // If we're closing a paren and back to depth 0, we might be done
        if (parenDepth < 0) {
          break; // Don't consume this closing paren
        }
      }
      else if (token.value === "[") bracketDepth++;
      else if (token.value === "]") {
        bracketDepth--;
        if (bracketDepth < 0) break;
      }
      else if (token.value === "{") braceDepth++;
      else if (token.value === "}") {
        braceDepth--;
        if (braceDepth < 0) break;
      }
      
      tokens.push(token.value);
      this.next();
      
      // After consuming a token, check if we should stop
      if (parenDepth === 0 && bracketDepth === 0 && braceDepth === 0) {
        // Look ahead to see if next token is a delimiter
        const nextToken = this.current();
        if (nextToken && (nextToken.value === ";" || 
                         nextToken.value === ")" || 
                         nextToken.value === "}" ||
                         nextToken.value === "," ||
                         nextToken.value === ":")) {
          break;
        }
      }
    }
    
    return {
      type: "Expression",
      tokens
    };
  }
}

// --- Main Parse Function ---
function parseJavaFile(filepath) {
  const code = fs.readFileSync(filepath, "utf-8");
  const tokens = tokenize(code);
  const parser = new Parser(tokens);
  return parser.parse();
}

// --- Enhanced Feature Extraction ---
function extractFeatures(ast) {
  const stats = {
    // Basic counts
    num_classes: 0,
    num_interfaces: 0,
    num_enums: 0,
    num_methods: 0,
    num_constructors: 0,
    num_fields: 0,
    
    // Statement counts
    num_if: 0,
    num_for: 0,
    num_while: 0,
    num_do_while: 0,
    num_switch: 0,
    num_try: 0,
    num_return: 0,
    num_break: 0,
    num_continue: 0,
    num_throw: 0,
    
    // Declaration counts
    num_imports: 0,
    num_package: 0,
    num_expressions: 0,
    num_statements: 0,
    
    // Complexity metrics
    total_method_lengths: 0,
    max_depth: 0,
    cyclomatic_complexity: 0,
    
    // Modifiers
    num_public: 0,
    num_private: 0,
    num_protected: 0,
    num_static: 0,
    num_final: 0,
    num_abstract: 0,
    
    // Inheritance
    num_extends: 0,
    num_implements: 0,
    
    // Error tracking
    num_errors: 0
  };

  const sequence = [];
  let methodCount = 0;

  function countModifiers(modifiers) {
    if (!modifiers) return;
    modifiers.forEach(mod => {
      switch (mod) {
        case 'public': stats.num_public++; break;
        case 'private': stats.num_private++; break;
        case 'protected': stats.num_protected++; break;
        case 'static': stats.num_static++; break;
        case 'final': stats.num_final++; break;
        case 'abstract': stats.num_abstract++; break;
      }
    });
  }

  function calculateCyclomaticComplexity(node) {
    let complexity = 1; // Base complexity
    
    function traverse(n) {
      if (!n || typeof n !== 'object') return;
      
      switch (n.type) {
        case 'IfStatement':
        case 'ForStatement':
        case 'WhileStatement':
        case 'DoWhileStatement':
        case 'SwitchCase':
        case 'CatchClause':
          complexity++;
          break;
      }
      
      for (const key in n) {
        const child = n[key];
        if (Array.isArray(child)) {
          child.forEach(traverse);
        } else if (typeof child === 'object') {
          traverse(child);
        }
      }
    }
    
    traverse(node);
    return complexity;
  }

  function traverse(node, depth = 0) {
    if (!node || typeof node !== 'object') return;
    
    if (node.type) sequence.push(node.type);
    if (depth > stats.max_depth) stats.max_depth = depth;

    switch (node.type) {
      case "ClassDeclaration":
        stats.num_classes++;
        countModifiers(node.modifiers);
        if (node.superClass) stats.num_extends++;
        if (node.interfaces && node.interfaces.length > 0) stats.num_implements++;
        break;
        
      case "InterfaceDeclaration":
        stats.num_interfaces++;
        countModifiers(node.modifiers);
        break;
        
      case "EnumDeclaration":
        stats.num_enums++;
        countModifiers(node.modifiers);
        break;
        
      case "MethodDeclaration":
        stats.num_methods++;
        methodCount++;
        countModifiers(node.modifiers);
        if (node.body && node.body.body) {
          stats.total_method_lengths += node.body.body.length;
          stats.cyclomatic_complexity += calculateCyclomaticComplexity(node.body);
        }
        break;
        
      case "ConstructorDeclaration":
        stats.num_constructors++;
        countModifiers(node.modifiers);
        if (node.body && node.body.body) {
          stats.total_method_lengths += node.body.body.length;
          stats.cyclomatic_complexity += calculateCyclomaticComplexity(node.body);
        }
        break;
        
      case "FieldDeclaration":
        stats.num_fields++;
        countModifiers(node.modifiers);
        break;
        
      case "IfStatement": stats.num_if++; stats.num_statements++; break;
      case "ForStatement": stats.num_for++; stats.num_statements++; break;
      case "WhileStatement": stats.num_while++; stats.num_statements++; break;
      case "DoWhileStatement": stats.num_do_while++; stats.num_statements++; break;
      case "SwitchStatement": stats.num_switch++; stats.num_statements++; break;
      case "TryStatement": stats.num_try++; stats.num_statements++; break;
      case "ReturnStatement": stats.num_return++; stats.num_statements++; break;
      case "BreakStatement": stats.num_break++; stats.num_statements++; break;
      case "ContinueStatement": stats.num_continue++; stats.num_statements++; break;
      case "ThrowStatement": stats.num_throw++; stats.num_statements++; break;
      case "ImportDeclaration": stats.num_imports++; break;
      case "PackageDeclaration": stats.num_package++; break;
      case "ExpressionStatement": stats.num_expressions++; stats.num_statements++; break;
    }

    for (const key in node) {
      const child = node[key];
      if (Array.isArray(child)) {
        child.forEach(n => traverse(n, depth + 1));
      } else if (typeof child === 'object') {
        traverse(child, depth + 1);
      }
    }
  }

  traverse(ast);
  
  // Calculate averages
  stats.avg_method_length = methodCount > 0 ? stats.total_method_lengths / methodCount : 0;
  stats.avg_cyclomatic_complexity = methodCount > 0 ? stats.cyclomatic_complexity / methodCount : 0;
  delete stats.total_method_lengths;
  
  // Count errors
  if (ast.errors) {
    stats.num_errors = ast.errors.length;
  }

  // Generate n-grams
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
  tokenize,
  Parser
};