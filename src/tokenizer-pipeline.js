// tokenizer-pipeline.js — Full Tokenizer Pipeline
// Combines BPE with pre/post-processing for a complete text ↔ tokens workflow.
// Inspired by SentencePiece and HuggingFace tokenizers.

import { BPETokenizer } from './bpe-tokenizer.js';

/**
 * Full tokenizer pipeline with special tokens and encoding/decoding.
 */
export class TokenizerPipeline {
  constructor() {
    this.bpe = new BPETokenizer();
    this.specialTokens = new Map();
    this.idToSpecial = new Map();
    this.vocab = new Map(); // token string → id
    this.idToToken = new Map(); // id → token string
    this.nextId = 0;
    
    // Add default special tokens
    this.addSpecialToken('<pad>', 0);
    this.addSpecialToken('<unk>', 1);
    this.addSpecialToken('<bos>', 2);
    this.addSpecialToken('<eos>', 3);
  }

  addSpecialToken(token, id = null) {
    if (id === null) id = this.nextId++;
    else if (id >= this.nextId) this.nextId = id + 1;
    
    this.specialTokens.set(token, id);
    this.idToSpecial.set(id, token);
    this.vocab.set(token, id);
    this.idToToken.set(id, token);
    return id;
  }

  /**
   * Train BPE on a corpus and build vocabulary.
   * @param {Array<string>} texts - Training texts
   * @param {number} vocabSize - Target vocabulary size
   */
  train(texts, vocabSize = 256) {
    this.bpe.train(texts, vocabSize - this.specialTokens.size);
    
    // Build vocab from BPE (returns [string, id] pairs)
    const bpeVocab = this.bpe.getVocab();
    for (const [token, bpeId] of bpeVocab) {
      if (!this.vocab.has(token)) {
        const id = this.nextId++;
        this.vocab.set(token, id);
        this.idToToken.set(id, token);
      }
    }
    
    // Build reverse lookup: BPE id → our id
    this._bpeIdToId = new Map();
    for (const [token, bpeId] of bpeVocab) {
      this._bpeIdToId.set(bpeId, this.vocab.get(token));
    }
    
    // Build our id → BPE token lookup
    this._idToStr = new Map();
    for (const [token, bpeId] of bpeVocab) {
      this._idToStr.set(this.vocab.get(token), token);
    }
  }

  /**
   * Encode text to token IDs.
   * @param {string} text
   * @param {boolean} addSpecial - Whether to add BOS/EOS
   * @returns {Array<number>} Token IDs
   */
  encode(text, addSpecial = true) {
    const bpeIds = this.bpe.encode(text);
    const ids = [];
    
    if (addSpecial) ids.push(this.specialTokens.get('<bos>'));
    
    for (const bpeId of bpeIds) {
      const id = this._bpeIdToId.get(bpeId);
      ids.push(id !== undefined ? id : this.specialTokens.get('<unk>'));
    }
    
    if (addSpecial) ids.push(this.specialTokens.get('<eos>'));
    
    return ids;
  }

  /**
   * Decode token IDs back to text.
   * @param {Array<number>} ids
   * @param {boolean} skipSpecial
   * @returns {string}
   */
  decode(ids, skipSpecial = true) {
    const tokens = [];
    for (const id of ids) {
      if (skipSpecial && this.idToSpecial.has(id)) continue;
      const token = this._idToStr ? this._idToStr.get(id) : this.idToToken.get(id);
      if (token) tokens.push(token);
    }
    return tokens.join('');
  }

  /**
   * Pad a batch of sequences to the same length.
   * @param {Array<Array<number>>} sequences
   * @param {number} maxLen - Maximum length (null = longest in batch)
   * @returns {{ padded: Array<Array<number>>, attentionMask: Array<Array<number>> }}
   */
  pad(sequences, maxLen = null) {
    if (maxLen === null) maxLen = Math.max(...sequences.map(s => s.length));
    const padId = this.specialTokens.get('<pad>');
    
    const padded = [];
    const attentionMask = [];
    
    for (const seq of sequences) {
      const p = [...seq];
      const mask = new Array(maxLen).fill(0);
      for (let i = 0; i < Math.min(seq.length, maxLen); i++) mask[i] = 1;
      
      while (p.length < maxLen) p.push(padId);
      if (p.length > maxLen) p.length = maxLen;
      
      padded.push(p);
      attentionMask.push(mask);
    }
    
    return { padded, attentionMask };
  }

  vocabSize() {
    return this.vocab.size;
  }
}
