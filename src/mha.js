// mha.js — Multi-Head Self-Attention (Vaswani et al., 2017)
// The fundamental attention module used in all transformers.
// Complete implementation with Q/K/V projections, output projection.

import { Matrix } from './matrix.js';

export class MultiHeadAttention {
  /**
   * @param {number} dModel - Model dimension
   * @param {number} nHeads - Number of attention heads
   */
  constructor(dModel, nHeads) {
    this.dModel = dModel;
    this.nHeads = nHeads;
    this.headDim = dModel / nHeads;
    
    const scale = Math.sqrt(2.0 / dModel);
    this.Wq = Matrix.random(dModel, dModel).map(v => v * scale);
    this.Wk = Matrix.random(dModel, dModel).map(v => v * scale);
    this.Wv = Matrix.random(dModel, dModel).map(v => v * scale);
    this.Wo = Matrix.random(dModel, dModel).map(v => v * scale);
    
    this._attnWeights = null; // Saved for visualization
  }

  /**
   * Forward pass.
   * @param {Matrix} x - Input (seqLen × dModel)
   * @param {Matrix} mask - Optional attention mask (seqLen × seqLen)
   * @returns {Matrix} Output (seqLen × dModel)
   */
  forward(x, mask = null) {
    const seqLen = x.rows;
    const { dModel, nHeads, headDim } = this;
    
    // Project Q, K, V
    const Q = matmul(x, this.Wq);
    const K = matmul(x, this.Wk);
    const V = matmul(x, this.Wv);
    
    const scale = 1 / Math.sqrt(headDim);
    const output = new Matrix(seqLen, dModel);
    this._attnWeights = [];
    
    for (let h = 0; h < nHeads; h++) {
      const offset = h * headDim;
      
      // Attention scores: Q_h @ K_h^T / sqrt(d)
      const scores = new Matrix(seqLen, seqLen);
      for (let i = 0; i < seqLen; i++) {
        for (let j = 0; j < seqLen; j++) {
          let dot = 0;
          for (let d = 0; d < headDim; d++) {
            dot += Q.get(i, offset + d) * K.get(j, offset + d);
          }
          let score = dot * scale;
          if (mask) score += mask.get(i, j);
          scores.set(i, j, score);
        }
      }
      
      // Softmax per row
      for (let i = 0; i < seqLen; i++) {
        let max = -Infinity;
        for (let j = 0; j < seqLen; j++) max = Math.max(max, scores.get(i, j));
        let sum = 0;
        for (let j = 0; j < seqLen; j++) {
          const e = Math.exp(scores.get(i, j) - max);
          scores.set(i, j, e);
          sum += e;
        }
        for (let j = 0; j < seqLen; j++) scores.set(i, j, scores.get(i, j) / sum);
      }
      
      this._attnWeights.push(scores);
      
      // Weighted sum of V
      for (let i = 0; i < seqLen; i++) {
        for (let d = 0; d < headDim; d++) {
          let sum = 0;
          for (let j = 0; j < seqLen; j++) {
            sum += scores.get(i, j) * V.get(j, offset + d);
          }
          output.set(i, offset + d, sum);
        }
      }
    }
    
    // Output projection
    return matmul(output, this.Wo);
  }

  /**
   * Get attention weights from last forward pass (for visualization).
   */
  getAttentionWeights() {
    return this._attnWeights;
  }
}

function matmul(A, B) {
  const result = new Matrix(A.rows, B.cols);
  for (let i = 0; i < A.rows; i++) {
    for (let j = 0; j < B.cols; j++) {
      let sum = 0;
      for (let k = 0; k < A.cols; k++) sum += A.get(i, k) * B.get(k, j);
      result.set(i, j, sum);
    }
  }
  return result;
}
