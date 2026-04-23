// cross-attention.js — Cross-Attention + Attention Pooling
// Cross-attention: Q from one sequence, K/V from another
// Used in encoder-decoder, DALL-E (text→image), image captioning

import { Matrix } from './matrix.js';

/**
 * Cross-attention: Q from decoder, K/V from encoder.
 * @param {Matrix} query - Decoder queries (seqQ × dModel)
 * @param {Matrix} key - Encoder keys (seqK × dModel)
 * @param {Matrix} value - Encoder values (seqK × dModel)
 * @param {number} nHeads - Number of attention heads
 * @param {boolean} causalMask - Whether to apply causal masking (usually false for cross-attention)
 * @returns {Matrix} Output (seqQ × dModel)
 */
export function crossAttention(query, key, value, nHeads = 1, causalMask = false) {
  const seqQ = query.rows;
  const seqK = key.rows;
  const dModel = query.cols;
  const headDim = dModel / nHeads;
  const scale = 1 / Math.sqrt(headDim);
  
  const output = new Matrix(seqQ, dModel);
  
  for (let h = 0; h < nHeads; h++) {
    const offset = h * headDim;
    
    // Compute attention scores: Q @ K^T / sqrt(d)
    const scores = new Matrix(seqQ, seqK);
    for (let i = 0; i < seqQ; i++) {
      for (let j = 0; j < seqK; j++) {
        let dot = 0;
        for (let d = 0; d < headDim; d++) {
          dot += query.get(i, offset + d) * key.get(j, offset + d);
        }
        scores.set(i, j, dot * scale);
        
        if (causalMask && j > i) scores.set(i, j, -1e9);
      }
    }
    
    // Softmax per row
    for (let i = 0; i < seqQ; i++) {
      let max = -Infinity;
      for (let j = 0; j < seqK; j++) max = Math.max(max, scores.get(i, j));
      
      let sum = 0;
      for (let j = 0; j < seqK; j++) {
        const exp = Math.exp(scores.get(i, j) - max);
        scores.set(i, j, exp);
        sum += exp;
      }
      for (let j = 0; j < seqK; j++) scores.set(i, j, scores.get(i, j) / sum);
    }
    
    // Weighted sum of values
    for (let i = 0; i < seqQ; i++) {
      for (let d = 0; d < headDim; d++) {
        let sum = 0;
        for (let j = 0; j < seqK; j++) {
          sum += scores.get(i, j) * value.get(j, offset + d);
        }
        output.set(i, offset + d, sum);
      }
    }
  }
  
  return output;
}

/**
 * Attention pooling: single learnable query attends to all tokens.
 * Used to pool variable-length sequences into fixed-size vectors.
 * @param {Matrix} tokens - Token embeddings (seqLen × dim)
 * @param {Float64Array} queryVector - Learnable query (dim)
 * @returns {Float64Array} Pooled representation (dim)
 */
export function attentionPooling(tokens, queryVector) {
  const seqLen = tokens.rows;
  const dim = tokens.cols;
  const scale = 1 / Math.sqrt(dim);
  
  // Attention scores
  const scores = new Float64Array(seqLen);
  for (let i = 0; i < seqLen; i++) {
    let dot = 0;
    for (let d = 0; d < dim; d++) dot += queryVector[d] * tokens.get(i, d);
    scores[i] = dot * scale;
  }
  
  // Softmax
  const max = Math.max(...scores);
  let sum = 0;
  for (let i = 0; i < seqLen; i++) {
    scores[i] = Math.exp(scores[i] - max);
    sum += scores[i];
  }
  for (let i = 0; i < seqLen; i++) scores[i] /= sum;
  
  // Weighted sum
  const result = new Float64Array(dim);
  for (let i = 0; i < seqLen; i++) {
    for (let d = 0; d < dim; d++) {
      result[d] += scores[i] * tokens.get(i, d);
    }
  }
  
  return result;
}
