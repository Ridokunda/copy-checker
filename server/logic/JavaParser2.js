
const fs = require("fs");

// --- Tokenizer ---
function tokenize(code) {
  code = code.replace(/\/\/.*|\/\*[\s\S]*?\*\//g, '');

  const tokenPattern = /\b(package|import|class|public|private|protected|void|int|final|boolean|char|byte|short|long|float|double|String|if|for|while|static|return|try|catch|new|throws|throw)\b(?:\[\])*|(\w+(?:\[\])*)|(\d+\.\d+)|(\+\+|--|<=|>=|==|!=)|(?:"(?:\\.|[^"\\])*")|(?:'(?:\\.|[^'\\])*')|\{|\}|\(|\)|\.|;|,|[a-zA-Z_][a-zA-Z0-9_]*(?:\[\])*|\S/g;

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
    //console.log(`Starting parse at position ${this.pos}`);
    while (this.match("package", "import")) {
      const keyword = this.tokens[this.pos - 1];
      const value = this.parseQualifiedName();
      if (this.match(";")) {
        ast.body.push({ type: keyword === "package" ? "PackageDeclaration" : "ImportDeclaration", value });
      }
    }
    //console.log(`starting to parse class`);
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
      if (member){
        classNode.body.push(member);
      }else{
        this.next();
      }
    }

    return classNode;
  }

  parseMethodOrField() {
    const start = this.pos;
    const modifiers = [];
    //console.log(`Starting to parse method or field`);
    
    // Collect modifiers
    while (this.match("public", "private", "protected", "static", "final", "synchronized", "volatile", "transient")) {
      modifiers.push(this.tokens[this.pos - 1]);
    }
    
    const currentToken = this.current();
    const isPrimitiveType = ["void","int", "String", "boolean", "double", "float", "char", "byte", "short", "long"].includes(currentToken);
    const isClassType = /^[A-Z][a-zA-Z0-9_]*(\[\])*$/.test(currentToken);
    const isGenericType = /^[A-Z][a-zA-Z0-9_]*<.*>(\[\])*$/.test(currentToken);
    
    if (isPrimitiveType || isClassType || isGenericType) {
      let returnType = currentToken;
      this.next();
        
      // Handle generic types 
      if (currentToken.includes("<") && !currentToken.includes(">")) {
        while (this.current() && !this.current().includes(">")) {
          returnType += this.current();
          this.next();
        }
        if (this.current() && this.current().includes(">")) {
          returnType += this.current();
          this.next();
        }
      }
      
      const name = this.current();
      this.next();

      if (this.match("(")) {
        const params = [];
        while (!this.match(")")) {
          //console.log(`Starting to parse parameters`);
          if (this.pos >= this.tokens.length) break;
          
          if (["final"].includes(this.current())) {
            this.next();
          }
          
          const paramType = this.current();
          this.next();
          

          
          const paramName = this.current();
          this.next();
          params.push({ type: paramType, name: paramName });
          this.match(",");
        }
        //console.log(params);
        
        // Handle throws clause
        if (this.match("throws")) {
          while (this.current() !== "{" && this.pos < this.tokens.length) {
            this.next();
          }
        }
        
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
      

      this.pos = start;
    }
    
    // Parse as variable declaration
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
    //console.log("starting parsing variable declaration");
    const startPos = this.pos;
    
    const modifiers = [];
    while (this.match("public", "private", "protected", "static", "final")) {
      modifiers.push(this.tokens[this.pos - 1]);
    }
    
    const type = this.current();
    
    const isPrimitiveType = ["int[][]", "String[][]", "boolean[][]", "double[][]", "float[][]", "char[][]", "byte[][]", "short[][]", "long[][]",
                              "int[]", "String[]", "boolean[]", "double[]", "float[]", "char[]", "byte[]", "short[]", "long[]",
                             "int", "String", "boolean", "double", "float", "char", "byte", "short", "long"].includes(type);
    const isClassType = /^[A-Z][a-zA-Z0-9_]*(\[\])*$/.test(type); 
    const isGenericType = /^[A-Z][a-zA-Z0-9_]*<.*>(\[\])*$/.test(type); 
    
    if (!isPrimitiveType && !isClassType && !isGenericType) {
      this.pos = startPos;
      return null;
    }
    let fullType = type;
    this.next();

    if (type.includes("<") && !type.includes(">")) {
      while (this.current() && !this.current().includes(">")) {
        fullType += this.current();
        this.next();
      }
      if (this.current() && this.current().includes(">")) {
        fullType += this.current();
        this.next();
      }
    }
    
    const declarations = [];
    let name = this.current();
    if (!name || !/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(name)) {
      this.pos = startPos; 
      return null;
    }
    this.next();
    
    let value = null;
    if (this.match("=")) {
      value = this.parseAssignedValue();
    }
    //console.log(this.current());
    declarations.push({ name, value });
    
    // Handle multiple variables declared together 
    while (this.match(",")) {
      name = this.current();
      this.next();
      value = null;
      if (this.match("=")) {
        value = this.parseAssignedValue();
      }
      declarations.push({ name, value });
    }
    
    if (!this.match(";")) {
      //console.log("failde");
      this.pos = startPos;
      return null;
    }
    
    return {
      type: "VariableDeclaration",
      kind: modifiers.includes("final") ? "final" : "typed",
      dataType: fullType,
      declarations,
      modifiers
    };
  }
  parseAssignedValue(){
    const tokens = [];
    while (![";"].includes(this.current())) {
      if (this.pos >= this.tokens.length) break;
      tokens.push(this.current());
      this.next();
    }
    return tokens.length > 0 ? tokens.join(" ") : null;
  }

  parseExpression() {
    const tokens = [];
    while (![",", ";", "}", "]"].includes(this.current())) {
      if (this.pos >= this.tokens.length) break;
      tokens.push(this.current());
      this.next();
    }
    return tokens.length > 0 ? tokens.join(" ") : null;
  }
  

  parseStatement() {
    //console.log("startin parsing statement");
    
    if(this.match("System")){
      let value = this.parseSystemCall();
      return {type: "SystemCall", value};
    }
    if(this.match("java")){
      let value = this.parseJavaCall();
      return {type: "JavaCall", value};
    }
    if (this.match("if")) {
      const test = this.parseCondition();
      const consequent = this.parseStatementOrBlock();
      let alternate = null;
      if (this.match("else")) {
        if (this.current() === "if") {
          alternate = this.parseStatement();
        } else {
          alternate = this.parseStatementOrBlock();
        }
      }
      return { type: "IfStatement", test, consequent, alternate };
    }
    if (this.match("for")) {
      const test = this.parseCondition();
      const body = this.parseStatementOrBlock();
      return { type: "ForStatement", test, body };
    }
    if (this.match("while")) {
      const test = this.parseCondition();
      const body = this.parseStatementOrBlock();
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
    if(this.match("try")){
      const test = this.parseCondition();
      const body = this.parseStatementOrBlock();
      let alt = null
      if(this.match("catch")){
        alt = this.parseStatement();
      }
      return {type: "tryStatement", test, body, alt};
    }
    if (this.match("{")) {
      return this.parseBlock();
    }

    const varDecl = this.parseVariableDeclaration();
    if (varDecl) return varDecl;

    if (/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(this.current()) && this.tokens[this.pos + 1] === "(") {
      return this.parseMethodCall();
    }


    const expression = this.parseExpression();
    this.match(";");
    return expression ? { type: "ExpressionStatement", expression } : null;
  }
  
  parseStatementOrBlock() {
    if (this.current() === "{") {
      this.next();
      return this.parseBlock();
    } else {
      return this.parseStatement();
    }
  }

  parseSystemCall(){
    let value = []
    while(!this.match(";")){
      value.push(this.current());
      this.next();
    }
    return value.join("");
  }
  parseJavaCall(){
    let value = [];
    while(!this.match(";")){
      value.push(this.current());
      this.next();
    }
    return value.join("");
  }
  parseMethodCall(){
    const name = this.current();
    this.next();
    let args = [];
    if(this.match("(")){
      while (this.current() !== ")" && this.pos < this.tokens.length) {
        if (this.current() !== ",") {
          args.push(this.current());
        }
        this.next();
      }
      this.match(")");
    }
    this.match(";");

    return {
      type: "MethodCall",
      name,
      arguments: args
    };
  }

  parseBlock() {
    //console.log(`Starting to parse block`);
    const body = [];
    while (!this.match("}")) {
      if (this.pos >= this.tokens.length) break;
      const stmt = this.parseStatement();
      if (stmt) {
        body.push(stmt);
      } else {
        this.next();
      } 
    }
    //console.log("end parse block");
    return { type: "BlockStatement", body };
  }

  parseCondition() {
    //console.log("starting parsing condition");
    const tokens = [];
    if (!this.match("(")) return null;
    while (!this.match(")")) {
      if (this.pos >= this.tokens.length) break;
      tokens.push(this.current());
      this.next();
    }
    this.match(")");
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
function extractFeatures(ast, num_tokens) {
  const stats = {
    num_tokens:num_tokens,
    num_methods: 0,
    num_if: 0,
    num_for: 0,
    num_while: 0,
    num_return: 0,
    num_imports: 0,
    num_package: 0,
    num_expressions: 0,
    num_statements: 0,
    num_systemcall: 0,
    num_javacall: 0,
    num_variables: 0,
    num_var_declarations: 0,
    num_method_calls:0,
    total_method_lengths: 0,
    max_depth: 0
  };

  const sequence = [];
  let methodCount = 0;

  function traverse(node, depth = 0) {
    if (!node || typeof node !== 'object') return;
    if (node.type) sequence.push(node.type);
    if (depth > stats.max_depth) stats.max_depth = depth;

    switch (node.type) {
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
      case "SystemCall": stats.num_systemcall++; stats.num_statements++; break;
      case "JavaCall": stats.num_javacall++; stats.num_statements++; break;
      case "VariableDeclaration": stats.num_variables++; break;
      case "PackageDeclaration": stats.num_package++; break;
      case "ExpressionStatement": stats.num_expressions++; stats.num_statements++; break;
      case "MethodCall": stats.num_method_calls++; stats.num_statements++; break;
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
    let grams = {}
    if(n === 2){
      grams = {
        system_if:0,
        vardec_while:0,
        vardec_if:0,
        expression_expression:0,
        expression_if:0,
      };
    }
    if(n === 4){
      grams = {
        for_cond_block_if:0,
        vardec_for_cond_block:0,
        if_cond_block_system_:0,
        if_cond_block_if:0,
        if_cond_block_while:0,
        if_cond_block_vardec:0,
        while_cond_block_if:0,
        vardec_if_cond_block:0,
        while_cond_block_vardec:0,
        expression_expression_expression_expression:0
      };
    }
    

    for (let i = 0; i <= seq.length - n; i++) {
      const gram = seq.slice(i, i + n).join('_');
      //console.log(gram);
      if(n === 2){
        if(gram === "SystemCall_IfStatement") grams.system_if++;
        if(gram === "VariableDeclaration_WhileStatement") grams.vardec_while++;
        if(gram === "VariableDeclaration_IfStatement") grams.vardec_if++;
        if(gram === "ExpressionStatement_ExpressionStatement") grams.expression_expression++;
        if(gram === "ExpressionStatement_IfStatement") grams.expression_if++;

      }else if(n === 4){
        if(gram === "ForStatement_Condition_BlockStatement_IfStatement") grams.for_cond_block_if++;
        if(gram === "VariableDeclaration_ForStatement_Condition_BlockStatement") grams.vardec_for_cond_block++;
        if(gram === "IfStatement_Condition_BlockStatement_SystemCall") grams.if_cond_block_system_++;
        if(gram === "IfStatement_Condition_BlockStatement_IfStatement") grams.if_cond_block_if++;
        if(gram === "IfStatement_Condition_BlockStatement_WhileStatement") grams.if_cond_block_while++;
        if(gram === "IfStatement_Condition_BlockStatement_VariableDeclaration") grams.if_cond_block_vardec++;
        if(gram === "WhileStatement_Condition_BlockStatement_IfStatement") grams.while_cond_block_if++;
        if(gram === "VariableDeclaration_IfStatement_Condition_BlockStatement") grams.vardec_if_cond_block++;
        if(gram === "WhileStatement_Condition_BlockStatement_VariableDeclaration") grams.while_cond_block_vardec++;
        if(gram === "ExpressionStatement_ExpressionStatement_ExpressionStatement_ExpressionStatement") grams.expression_expression_expression_expression++;
      }
      
    }
    return grams;
  }

 

  const ngrams2 = generateNGrams(sequence, 2);
  const ngrams4 = generateNGrams(sequence, 4);
  Object.assign(stats, ngrams2, ngrams4);
  console.log(stats)
  return stats;
}


module.exports = {
  parseJavaFile,
  extractFeatures,
  tokenize
};