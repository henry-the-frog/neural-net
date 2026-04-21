// constrained-decoding.js — Constrained / Structured Output Decoding
// Used by: Outlines, Guidance, vLLM structured output, OpenAI JSON mode
//
// Constrain the model's output to follow a grammar or schema by masking
// logits at each step to only allow valid next tokens.

/**
 * Apply a token mask to logits: set disallowed tokens to -Infinity.
 *
 * @param {Float64Array} logits - raw model logits
 * @param {Set<number>|number[]} allowedTokens - set of allowed token IDs
 * @returns {Float64Array} masked logits
 */
export function applyTokenMask(logits, allowedTokens) {
  const allowed = allowedTokens instanceof Set ? allowedTokens : new Set(allowedTokens);
  const result = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    result[i] = allowed.has(i) ? logits[i] : -Infinity;
  }
  return result;
}

/**
 * Simple JSON grammar constraint.
 * Tracks JSON structure state and returns allowed tokens at each step.
 *
 * This is a simplified version — real implementations use finite automata
 * or pushdown automata to track arbitrary grammar states.
 */
export class JSONConstraint {
  constructor(tokenizer) {
    this.tokenizer = tokenizer;
    this.state = 'start'; // start, object, key, colon, value, comma, end
    this.depth = 0;
    this.buffer = '';
  }

  /**
   * Given current state, return set of allowed next characters/tokens.
   * For simplicity, works at character level.
   */
  allowedChars() {
    switch (this.state) {
      case 'start': return new Set(['{', '[']);
      case 'object': return new Set(['"', '}']);
      case 'key': return new Set([...'abcdefghijklmnopqrstuvwxyz_"']);
      case 'colon': return new Set([':']);
      case 'value': return new Set(['"', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 't', 'f', 'n', '{', '[']);
      case 'string_value': return new Set([...'abcdefghijklmnopqrstuvwxyz0123456789 _-"']);
      case 'comma': return new Set([',', '}', ']']);
      case 'end': return new Set([]);
      default: return new Set();
    }
  }

  /**
   * Advance state based on character consumed.
   */
  consume(char) {
    this.buffer += char;

    switch (this.state) {
      case 'start':
        if (char === '{') { this.state = 'object'; this.depth++; }
        break;
      case 'object':
        if (char === '"') this.state = 'key';
        else if (char === '}') { this.depth--; this.state = this.depth > 0 ? 'comma' : 'end'; }
        break;
      case 'key':
        if (char === '"') this.state = 'colon';
        break;
      case 'colon':
        if (char === ':') this.state = 'value';
        break;
      case 'value':
        if (char === '"') this.state = 'string_value';
        else if (char === '{') { this.state = 'object'; this.depth++; }
        else this.state = 'comma'; // number/bool/null
        break;
      case 'string_value':
        if (char === '"') this.state = 'comma';
        break;
      case 'comma':
        if (char === ',') this.state = 'object';
        else if (char === '}') { this.depth--; this.state = this.depth > 0 ? 'comma' : 'end'; }
        break;
    }
  }

  /**
   * Check if the current buffer is valid JSON.
   */
  isComplete() {
    return this.state === 'end';
  }

  reset() {
    this.state = 'start';
    this.depth = 0;
    this.buffer = '';
  }
}

/**
 * Regex-based constraint: only allow tokens that match a regex pattern.
 */
export class RegexConstraint {
  constructor(pattern) {
    this.pattern = new RegExp(pattern);
    this.buffer = '';
  }

  /**
   * Check if adding a character keeps the buffer as a valid partial match.
   */
  isValidExtension(char) {
    const extended = this.buffer + char;
    // Check if extended could be a prefix of a match
    // Simple approach: check if partial match is possible
    try {
      return this.pattern.test(extended) || this._couldMatch(extended);
    } catch {
      return false;
    }
  }

  consume(char) {
    this.buffer += char;
  }

  isComplete() {
    return this.pattern.test(this.buffer);
  }

  _couldMatch(partial) {
    // Heuristic: check if any extension could match
    // This is simplified — real implementation uses automaton
    for (const ext of ['', 'a', '0', '"', '}', ']']) {
      if (this.pattern.test(partial + ext)) return true;
    }
    return false;
  }
}
