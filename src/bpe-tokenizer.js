// bpe-tokenizer.js — Byte-Pair Encoding tokenizer
// Based on the original BPE algorithm by Sennrich et al. (2016)

/**
 * Byte-Pair Encoding tokenizer.
 * 
 * Training:
 * 1. Start with character-level tokens
 * 2. Count all adjacent pairs
 * 3. Merge the most frequent pair into a new token
 * 4. Repeat until vocab_size reached
 * 
 * Encoding:
 * 1. Split text into characters
 * 2. Apply learned merges in order
 * 
 * Decoding:
 * 1. Look up token strings and concatenate
 */
export class BPETokenizer {
  constructor({ vocabSize = 256 } = {}) {
    this.vocabSize = vocabSize;
    this.merges = [];      // Array of [pair, newToken] in order learned
    this.vocab = new Map(); // token string → token id
    this.inverseVocab = new Map(); // token id → token string
    this.specialTokens = new Map(); // e.g., <|endoftext|>
  }

  /**
   * Train BPE on a text corpus.
   * @param {string} text - Training text
   * @param {number} numMerges - Number of merge operations (vocabSize - baseVocabSize)
   */
  train(text, numMerges = null) {
    // Step 1: Initialize with byte-level vocabulary (all unique characters)
    const chars = new Set(text);
    let nextId = 0;
    
    // Add special tokens first
    for (const [tok, id] of this.specialTokens) {
      this.vocab.set(tok, id);
      this.inverseVocab.set(id, tok);
      nextId = Math.max(nextId, id + 1);
    }
    
    // Add all unique characters as base vocabulary
    for (const ch of chars) {
      if (!this.vocab.has(ch)) {
        this.vocab.set(ch, nextId);
        this.inverseVocab.set(nextId, ch);
        nextId++;
      }
    }
    
    const baseVocabSize = nextId;
    const targetMerges = numMerges ?? (this.vocabSize - baseVocabSize);
    
    // Step 2: Split text into words (whitespace-separated), keeping whitespace as prefix
    // Each word is represented as a list of tokens (initially characters)
    const words = this._splitIntoWords(text);
    let wordTokens = words.map(w => [...w]); // array of arrays of single chars
    
    // Step 3: Iterative merging
    for (let merge = 0; merge < targetMerges; merge++) {
      // Count all adjacent pairs across all words
      const pairCounts = new Map();
      for (const tokens of wordTokens) {
        for (let i = 0; i < tokens.length - 1; i++) {
          const pair = tokens[i] + '\0' + tokens[i + 1];
          pairCounts.set(pair, (pairCounts.get(pair) || 0) + 1);
        }
      }
      
      if (pairCounts.size === 0) break;
      
      // Find most frequent pair
      let bestPair = null;
      let bestCount = 0;
      for (const [pair, count] of pairCounts) {
        if (count > bestCount) {
          bestCount = count;
          bestPair = pair;
        }
      }
      
      if (bestCount < 2) break; // No pair appears more than once
      
      const [left, right] = bestPair.split('\0');
      const merged = left + right;
      
      // Add merged token to vocabulary
      this.vocab.set(merged, nextId);
      this.inverseVocab.set(nextId, merged);
      this.merges.push({ left, right, merged, id: nextId });
      nextId++;
      
      // Apply merge to all words
      wordTokens = wordTokens.map(tokens => this._applyMerge(tokens, left, right, merged));
    }
    
    return {
      vocabSize: this.vocab.size,
      numMerges: this.merges.length,
      baseVocabSize,
    };
  }

  /**
   * Encode text to token IDs.
   * @param {string} text - Input text
   * @returns {number[]} Array of token IDs
   */
  encode(text) {
    if (text.length === 0) return [];
    
    const words = this._splitIntoWords(text);
    const allTokens = [];
    
    for (const word of words) {
      let tokens = [...word]; // Start with characters
      
      // Apply all learned merges in order
      for (const { left, right, merged } of this.merges) {
        tokens = this._applyMerge(tokens, left, right, merged);
      }
      
      // Convert to IDs
      for (const tok of tokens) {
        const id = this.vocab.get(tok);
        if (id !== undefined) {
          allTokens.push(id);
        } else {
          // Unknown character — encode as individual bytes
          for (const ch of tok) {
            const chId = this.vocab.get(ch);
            if (chId !== undefined) allTokens.push(chId);
          }
        }
      }
    }
    
    return allTokens;
  }

  /**
   * Decode token IDs back to text.
   * @param {number[]} ids - Array of token IDs
   * @returns {string} Decoded text
   */
  decode(ids) {
    return ids.map(id => this.inverseVocab.get(id) || '').join('');
  }

  /**
   * Get the vocabulary as an array of [token, id] pairs.
   * @returns {Array} Vocabulary entries
   */
  getVocab() {
    return [...this.vocab.entries()].sort((a, b) => a[1] - b[1]);
  }

  /**
   * Add a special token.
   * @param {string} token - Special token string
   * @param {number} id - Token ID (optional, auto-assigned if not provided)
   */
  addSpecialToken(token, id = null) {
    if (id === null) {
      id = Math.max(0, ...this.vocab.values()) + 1;
    }
    this.specialTokens.set(token, id);
    this.vocab.set(token, id);
    this.inverseVocab.set(id, token);
  }

  /**
   * Serialize tokenizer state to JSON.
   */
  toJSON() {
    return {
      vocabSize: this.vocabSize,
      merges: this.merges,
      vocab: [...this.vocab.entries()],
      specialTokens: [...this.specialTokens.entries()],
    };
  }

  /**
   * Load tokenizer state from JSON.
   */
  static fromJSON(json) {
    const tok = new BPETokenizer({ vocabSize: json.vocabSize });
    tok.merges = json.merges;
    tok.vocab = new Map(json.vocab);
    tok.inverseVocab = new Map([...tok.vocab.entries()].map(([k, v]) => [v, k]));
    tok.specialTokens = new Map(json.specialTokens || []);
    return tok;
  }

  // --- Private methods ---

  /**
   * Split text into "words" — keeping whitespace attached to the following word.
   * This matches GPT-2's pre-tokenization pattern.
   */
  _splitIntoWords(text) {
    // Simple split: each word includes its leading whitespace
    const words = [];
    let current = '';
    for (let i = 0; i < text.length; i++) {
      if (text[i] === ' ' && current.length > 0) {
        words.push(current);
        current = ' ';
      } else {
        current += text[i];
      }
    }
    if (current.length > 0) words.push(current);
    return words;
  }

  /**
   * Apply a single merge operation to a token sequence.
   */
  _applyMerge(tokens, left, right, merged) {
    if (tokens.length < 2) return tokens;
    const result = [];
    let i = 0;
    while (i < tokens.length) {
      if (i < tokens.length - 1 && tokens[i] === left && tokens[i + 1] === right) {
        result.push(merged);
        i += 2;
      } else {
        result.push(tokens[i]);
        i++;
      }
    }
    return result;
  }
}
