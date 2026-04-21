// bpe.js — Byte Pair Encoding (BPE) Tokenizer
// Used by GPT-2/3/4, Llama, etc. for subword tokenization.
// BPE iteratively merges the most frequent pair of adjacent tokens.
//
// Training: given a corpus, learn merge rules by frequency
// Encoding: apply learned merge rules to tokenize new text
// Decoding: reverse token IDs back to text

/**
 * BPE Tokenizer
 *
 * Training algorithm:
 * 1. Start with character-level vocabulary (each char is a token)
 * 2. Count all adjacent token pairs in the corpus
 * 3. Merge the most frequent pair into a new token
 * 4. Repeat until vocab size reached
 *
 * Encoding:
 * 1. Split text into characters
 * 2. Apply merge rules in priority order
 * 3. Map merged tokens to IDs
 */
export class BPETokenizer {
  constructor() {
    this.merges = [];        // [{pair: [a, b], merged: "ab"}] in priority order
    this.vocab = new Map();  // token string → ID
    this.idToToken = [];     // ID → token string
    this.specialTokens = new Map(); // special tokens like <|endoftext|>
  }

  /**
   * Train BPE on a corpus.
   * @param {string} text - training text
   * @param {number} vocabSize - target vocabulary size
   * @param {string[]} [specialTokens] - special tokens to reserve
   */
  train(text, vocabSize, specialTokens = ['<|endoftext|>', '<|pad|>']) {
    // Initialize with special tokens
    this.vocab.clear();
    this.idToToken = [];
    this.merges = [];
    this.specialTokens.clear();

    for (const st of specialTokens) {
      const id = this.idToToken.length;
      this.vocab.set(st, id);
      this.idToToken.push(st);
      this.specialTokens.set(st, id);
    }

    // Initialize vocabulary with all unique bytes/characters in the corpus
    const charSet = new Set(text);
    for (const ch of charSet) {
      if (!this.vocab.has(ch)) {
        const id = this.idToToken.length;
        this.vocab.set(ch, id);
        this.idToToken.push(ch);
      }
    }

    // Split corpus into words (whitespace-delimited), then into character sequences
    // Each word is represented as a list of tokens
    const words = text.split(/(\s+)/); // keep whitespace as separate "words"
    let wordTokens = words.map(w => [...w]); // character-level

    const initialVocabSize = this.vocab.size;
    const numMerges = vocabSize - initialVocabSize;

    for (let step = 0; step < numMerges; step++) {
      // Count all adjacent pairs
      const pairCounts = new Map();
      for (const tokens of wordTokens) {
        for (let i = 0; i < tokens.length - 1; i++) {
          const key = tokens[i] + '\0' + tokens[i + 1];
          pairCounts.set(key, (pairCounts.get(key) || 0) + 1);
        }
      }

      if (pairCounts.size === 0) break;

      // Find most frequent pair
      let bestPair = null, bestCount = 0;
      for (const [key, count] of pairCounts) {
        if (count > bestCount) {
          bestCount = count;
          bestPair = key;
        }
      }

      const [a, b] = bestPair.split('\0');
      const merged = a + b;

      // Add merge rule
      this.merges.push({ pair: [a, b], merged });

      // Add to vocabulary
      if (!this.vocab.has(merged)) {
        const id = this.idToToken.length;
        this.vocab.set(merged, id);
        this.idToToken.push(merged);
      }

      // Apply merge to all words
      wordTokens = wordTokens.map(tokens => applyMerge(tokens, a, b, merged));
    }
  }

  /**
   * Encode text to token IDs.
   * @param {string} text
   * @returns {number[]} token IDs
   */
  encode(text) {
    // Split into characters
    let tokens = [...text];

    // Apply all merge rules in order
    for (const { pair: [a, b], merged } of this.merges) {
      tokens = applyMerge(tokens, a, b, merged);
    }

    // Map to IDs
    return tokens.map(t => {
      if (this.vocab.has(t)) return this.vocab.get(t);
      // Unknown token: use first special token as fallback
      return 0;
    });
  }

  /**
   * Decode token IDs back to text.
   * @param {number[]} ids
   * @returns {string}
   */
  decode(ids) {
    return ids.map(id => this.idToToken[id] || '').join('');
  }

  /**
   * Get vocabulary size.
   */
  get size() {
    return this.vocab.size;
  }

  /**
   * Export merge rules and vocabulary for serialization.
   */
  export() {
    return {
      merges: this.merges,
      vocab: Object.fromEntries(this.vocab),
      specialTokens: Object.fromEntries(this.specialTokens),
    };
  }

  /**
   * Import from exported data.
   */
  static import(data) {
    const tok = new BPETokenizer();
    tok.merges = data.merges;
    tok.vocab = new Map(Object.entries(data.vocab).map(([k, v]) => [k, Number(v)]));
    tok.idToToken = new Array(tok.vocab.size);
    for (const [token, id] of tok.vocab) tok.idToToken[id] = token;
    tok.specialTokens = new Map(Object.entries(data.specialTokens || {}).map(([k, v]) => [k, Number(v)]));
    return tok;
  }
}

/**
 * Apply a single merge rule to a token sequence.
 * Replace all occurrences of [a, b] with [merged].
 */
function applyMerge(tokens, a, b, merged) {
  const result = [];
  let i = 0;
  while (i < tokens.length) {
    if (i < tokens.length - 1 && tokens[i] === a && tokens[i + 1] === b) {
      result.push(merged);
      i += 2;
    } else {
      result.push(tokens[i]);
      i++;
    }
  }
  return result;
}
