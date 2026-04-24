// grouped-query-attention.js — Grouped Query Attention (Ainslie et al., 2023)
//
// Used in Llama 2, Llama 3, Mistral, and many modern LLMs.
// Key idea: use fewer K/V heads than Q heads to reduce KV cache size
// during inference while maintaining quality close to full MHA.
//
// If numKVHeads == numHeads: equivalent to Multi-Head Attention (MHA)
// If numKVHeads == 1: equivalent to Multi-Query Attention (MQA)
// Otherwise: Grouped Query Attention (GQA)
//
// Memory savings: KV cache reduces by factor of numHeads/numKVHeads

import { Matrix } from './matrix.js';
import { flashAttention } from './flash-attention.js';

function extractCols(m, offset, count) {
  const result = new Matrix(m.rows, count);
  for (let i = 0; i < m.rows; i++)
    for (let j = 0; j < count; j++)
      result.set(i, j, m.get(i, offset + j));
  return result;
}

export class GroupedQueryAttention {
  /**
   * @param {number} dModel — model dimension
   * @param {number} numHeads — number of query heads
   * @param {number} [numKVHeads=numHeads] — number of key/value heads (must divide numHeads)
   * @param {Object} [opts]
   * @param {boolean} [opts.causal=false]
   * @param {number} [opts.blockSize=32] — flash attention block size
   */
  constructor(dModel, numHeads, numKVHeads = numHeads, { causal = false, blockSize = 32 } = {}) {
    if (dModel % numHeads !== 0) throw new Error('dModel must be divisible by numHeads');
    if (numHeads % numKVHeads !== 0) throw new Error('numHeads must be divisible by numKVHeads');
    
    this.dModel = dModel;
    this.numHeads = numHeads;
    this.numKVHeads = numKVHeads;
    this.headDim = dModel / numHeads;
    this.kvDim = numKVHeads * this.headDim;
    this.groupSize = numHeads / numKVHeads; // Q heads per KV head
    this.causal = causal;
    this.blockSize = blockSize;
    
    const s = Math.sqrt(2 / (dModel + dModel));
    
    // Q projections: full size (numHeads * headDim = dModel)
    this.Wq = Matrix.random(dModel, dModel).mul(s);
    this.bq = Matrix.zeros(1, dModel);
    
    // K, V projections: reduced size (numKVHeads * headDim)
    this.Wk = Matrix.random(dModel, this.kvDim).mul(s);
    this.Wv = Matrix.random(dModel, this.kvDim).mul(s);
    this.bk = Matrix.zeros(1, this.kvDim);
    this.bv = Matrix.zeros(1, this.kvDim);
    
    // Output projection: full size
    this.Wo = Matrix.random(dModel, dModel).mul(s);
    this.bo = Matrix.zeros(1, dModel);
    
    this.outputSize = dModel;
    this._cache = null;
  }
  
  forward(input) {
    const batchSize = input.rows;
    const seqLen = Math.floor(input.cols / this.dModel);
    const d = this.dModel;
    const hd = this.headDim;
    const result = new Matrix(batchSize, seqLen * d);
    
    this._seqs = [];
    this._allQ = [];
    this._allK = [];
    this._allV = [];
    this._allHeadOutputs = [];
    this._batchSize = batchSize;
    this._seqLen = seqLen;
    
    for (let b = 0; b < batchSize; b++) {
      const seq = new Matrix(seqLen, d);
      for (let t = 0; t < seqLen; t++)
        for (let k = 0; k < d; k++)
          seq.set(t, k, input.get(b, t * d + k));
      this._seqs.push(seq);
      
      // Q: [seqLen, dModel] → [seqLen, numHeads * headDim]
      const Q = seq.dot(this.Wq).add(this.bq);
      // K, V: [seqLen, dModel] → [seqLen, numKVHeads * headDim]
      const K = seq.dot(this.Wk).add(this.bk);
      const V = seq.dot(this.Wv).add(this.bv);
      this._allQ.push(Q);
      this._allK.push(K);
      this._allV.push(V);
      
      // Per-head attention with KV head sharing
      const headOutputs = [];
      for (let h = 0; h < this.numHeads; h++) {
        const qOffset = h * hd;
        const kvHead = Math.floor(h / this.groupSize);
        const kvOffset = kvHead * hd;
        
        const Qh = extractCols(Q, qOffset, hd);
        const Kh = extractCols(K, kvOffset, hd); // Shared KV head
        const Vh = extractCols(V, kvOffset, hd); // Shared KV head
        
        const context = flashAttention(Qh, Kh, Vh, this.blockSize, this.causal);
        headOutputs.push(context);
      }
      this._allHeadOutputs.push(headOutputs);
      
      // Concatenate heads
      const concat = new Matrix(seqLen, d);
      for (let h = 0; h < this.numHeads; h++) {
        const offset = h * hd;
        for (let t = 0; t < seqLen; t++)
          for (let k = 0; k < hd; k++)
            concat.set(t, offset + k, headOutputs[h].get(t, k));
      }
      
      // Output projection
      const projected = concat.dot(this.Wo).add(this.bo);
      for (let t = 0; t < seqLen; t++)
        for (let k = 0; k < d; k++)
          result.set(b, t * d + k, projected.get(t, k));
    }
    
    this._cache = { batchSize, seqLen };
    return result;
  }
  
  /**
   * KV cache size in elements (for inference comparison).
   * MHA: 2 * seqLen * dModel
   * GQA: 2 * seqLen * numKVHeads * headDim
   */
  kvCacheSize(seqLen) {
    return 2 * seqLen * this.numKVHeads * this.headDim;
  }
  
  /**
   * MHA-equivalent KV cache size for comparison.
   */
  mhaKVCacheSize(seqLen) {
    return 2 * seqLen * this.dModel;
  }
  
  paramCount() {
    // Q: dModel × dModel + dModel
    // K: dModel × kvDim + kvDim
    // V: dModel × kvDim + kvDim
    // O: dModel × dModel + dModel
    return this.dModel * this.dModel + this.dModel          // Q
         + this.dModel * this.kvDim + this.kvDim             // K
         + this.dModel * this.kvDim + this.kvDim             // V
         + this.dModel * this.dModel + this.dModel;          // O
  }
}
