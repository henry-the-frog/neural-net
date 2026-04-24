// multi-head-flash-attention.js — Multi-Head Flash Attention
// Drop-in replacement for MultiHeadAttention that uses flash attention per head
// O(N·d) memory per head instead of O(N²)

import { Matrix } from './matrix.js';
import { flashAttention } from './flash-attention.js';

function extractCols(m, offset, count) {
  const result = new Matrix(m.rows, count);
  for (let i = 0; i < m.rows; i++)
    for (let j = 0; j < count; j++)
      result.set(i, j, m.get(i, offset + j));
  return result;
}

function addInPlace(target, source) {
  for (let i = 0; i < target.data.length; i++) target.data[i] += source.data[i];
}

/**
 * Multi-Head Flash Attention
 * Same API as MultiHeadAttention but uses flash attention per head.
 * Memory: O(N·d) per head instead of O(N²)
 */
export class MultiHeadFlashAttention {
  constructor(dModel, numHeads, { blockSize = 32, causal = false, dropout = 0 } = {}) {
    if (dModel % numHeads !== 0) throw new Error(`dModel (${dModel}) must be divisible by numHeads (${numHeads})`);
    this.dModel = dModel;
    this.numHeads = numHeads;
    this.headDim = dModel / numHeads;
    this.blockSize = blockSize;
    this.causal = causal;
    
    const scale = Math.sqrt(2 / (dModel + dModel));
    this.Wq = Matrix.random(dModel, dModel).mul(scale);
    this.Wk = Matrix.random(dModel, dModel).mul(scale);
    this.Wv = Matrix.random(dModel, dModel).mul(scale);
    this.Wo = Matrix.random(dModel, dModel).mul(scale);
    
    this.bq = Matrix.zeros(1, dModel);
    this.bk = Matrix.zeros(1, dModel);
    this.bv = Matrix.zeros(1, dModel);
    this.bo = Matrix.zeros(1, dModel);
    
    this.outputSize = dModel;
    this.training = true;
    this._cache = null;
    this.dWeights = null;
    this.dBiases = null;
  }
  
  forward(input) {
    const batchSize = input.rows;
    const seqLen = Math.floor(input.cols / this.dModel);
    const d = this.dModel;
    const result = new Matrix(batchSize, seqLen * d);
    
    this._seqs = [];
    this._allQ = [];
    this._allK = [];
    this._allV = [];
    this._allHeadOutputs = []; // Per-head outputs for backward
    this._batchSize = batchSize;
    this._seqLen = seqLen;
    
    for (let b = 0; b < batchSize; b++) {
      const seq = new Matrix(seqLen, d);
      for (let t = 0; t < seqLen; t++)
        for (let k = 0; k < d; k++)
          seq.set(t, k, input.get(b, t * d + k));
      this._seqs.push(seq);
      
      // Linear projections
      const Q = seq.dot(this.Wq).add(this.bq);
      const K = seq.dot(this.Wk).add(this.bk);
      const V = seq.dot(this.Wv).add(this.bv);
      this._allQ.push(Q);
      this._allK.push(K);
      this._allV.push(V);
      
      // Per-head flash attention
      const headOutputs = [];
      for (let h = 0; h < this.numHeads; h++) {
        const offset = h * this.headDim;
        const Qh = extractCols(Q, offset, this.headDim);
        const Kh = extractCols(K, offset, this.headDim);
        const Vh = extractCols(V, offset, this.headDim);
        
        // Flash attention: O(N·headDim) memory
        const context = flashAttention(Qh, Kh, Vh, this.blockSize, this.causal);
        headOutputs.push(context);
      }
      this._allHeadOutputs.push(headOutputs);
      
      // Concatenate heads: [seqLen, dModel]
      const concat = new Matrix(seqLen, d);
      for (let h = 0; h < this.numHeads; h++) {
        const offset = h * this.headDim;
        for (let t = 0; t < seqLen; t++)
          for (let k = 0; k < this.headDim; k++)
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
  
  backward(dOutput) {
    const { batchSize, seqLen } = this._cache;
    const d = this.dModel;
    const hd = this.headDim;
    
    const dInput = new Matrix(batchSize, seqLen * d);
    const dWq = Matrix.zeros(d, d);
    const dWk = Matrix.zeros(d, d);
    const dWv = Matrix.zeros(d, d);
    const dWo = Matrix.zeros(d, d);
    const dbq = Matrix.zeros(1, d);
    const dbk = Matrix.zeros(1, d);
    const dbv = Matrix.zeros(1, d);
    const dbo = Matrix.zeros(1, d);
    
    for (let b = 0; b < batchSize; b++) {
      const seq = this._seqs[b];
      const Q = this._allQ[b], K = this._allK[b], V = this._allV[b];
      
      // Extract dOutput for batch item
      const dOut = new Matrix(seqLen, d);
      for (let t = 0; t < seqLen; t++)
        for (let k = 0; k < d; k++)
          dOut.set(t, k, dOutput.get(b, t * d + k));
      
      // Backward through output projection
      // concat → Wo → out, so dConcat = dOut · Wo^T
      // Reconstruct concat from head outputs
      const concat = new Matrix(seqLen, d);
      for (let h = 0; h < this.numHeads; h++) {
        const offset = h * hd;
        for (let t = 0; t < seqLen; t++)
          for (let k = 0; k < hd; k++)
            concat.set(t, offset + k, this._allHeadOutputs[b][h].get(t, k));
      }
      
      const dConcat = dOut.dot(this.Wo.T());
      addInPlace(dWo, concat.T().dot(dOut));
      for (let j = 0; j < d; j++) {
        let sum = 0;
        for (let i = 0; i < seqLen; i++) sum += dOut.get(i, j);
        dbo.set(0, j, dbo.get(0, j) + sum);
      }
      
      // Backward through each head's flash attention
      // dConcat has gradient for concatenated heads
      const dQ = Matrix.zeros(seqLen, d);
      const dK = Matrix.zeros(seqLen, d);
      const dV = Matrix.zeros(seqLen, d);
      
      for (let h = 0; h < this.numHeads; h++) {
        const offset = h * hd;
        const Qh = extractCols(Q, offset, hd);
        const Kh = extractCols(K, offset, hd);
        const Vh = extractCols(V, offset, hd);
        const Oh = this._allHeadOutputs[b][h];
        
        // Extract head gradient
        const dOh = extractCols(dConcat, offset, hd);
        
        // Flash attention backward: recompute and differentiate
        const scale = 1 / Math.sqrt(hd);
        
        // Compute attention stats (recompute from forward)
        const n = seqLen;
        const m = new Float64Array(n).fill(-Infinity);
        const l = new Float64Array(n).fill(0);
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++) {
            if (this.causal && j > i) continue;
            let dot = 0;
            for (let k = 0; k < hd; k++) dot += Qh.get(i, k) * Kh.get(j, k);
            dot *= scale;
            if (dot > m[i]) m[i] = dot;
          }
          for (let j = 0; j < n; j++) {
            if (this.causal && j > i) continue;
            let dot = 0;
            for (let k = 0; k < hd; k++) dot += Qh.get(i, k) * Kh.get(j, k);
            l[i] += Math.exp(dot * scale - m[i]);
          }
        }
        
        // D[i] = sum_k dO[i,k] * O[i,k]
        const D = new Float64Array(n);
        for (let i = 0; i < n; i++) {
          let dot = 0;
          for (let k = 0; k < hd; k++) dot += dOh.get(i, k) * Oh.get(i, k);
          D[i] = dot;
        }
        
        // Compute gradients
        const dQh = Matrix.zeros(n, hd);
        const dKh = Matrix.zeros(n, hd);
        const dVh = Matrix.zeros(n, hd);
        
        for (let i = 0; i < n; i++) {
          if (l[i] === 0) continue;
          for (let j = 0; j < n; j++) {
            if (this.causal && j > i) continue;
            let dot = 0;
            for (let k = 0; k < hd; k++) dot += Qh.get(i, k) * Kh.get(j, k);
            const p = Math.exp(dot * scale - m[i]) / l[i];
            
            let dP = 0;
            for (let k = 0; k < hd; k++) dP += dOh.get(i, k) * Vh.get(j, k);
            const dS = p * (dP - D[i]) * scale;
            
            for (let k = 0; k < hd; k++) {
              dQh.set(i, k, dQh.get(i, k) + dS * Kh.get(j, k));
              dKh.set(j, k, dKh.get(j, k) + dS * Qh.get(i, k));
              dVh.set(j, k, dVh.get(j, k) + p * dOh.get(i, k));
            }
          }
        }
        
        // Scatter head gradients back to full dimension
        for (let t = 0; t < n; t++) {
          for (let k = 0; k < hd; k++) {
            dQ.set(t, offset + k, dQ.get(t, offset + k) + dQh.get(t, k));
            dK.set(t, offset + k, dK.get(t, offset + k) + dKh.get(t, k));
            dV.set(t, offset + k, dV.get(t, offset + k) + dVh.get(t, k));
          }
        }
      }
      
      // Backward through projections
      const dSeq = dQ.dot(this.Wq.T())
        .add(dK.dot(this.Wk.T()))
        .add(dV.dot(this.Wv.T()));
      
      addInPlace(dWq, seq.T().dot(dQ));
      addInPlace(dWk, seq.T().dot(dK));
      addInPlace(dWv, seq.T().dot(dV));
      
      for (let j = 0; j < d; j++) {
        let sq = 0, sk = 0, sv = 0;
        for (let i = 0; i < seqLen; i++) {
          sq += dQ.get(i, j); sk += dK.get(i, j); sv += dV.get(i, j);
        }
        dbq.set(0, j, dbq.get(0, j) + sq);
        dbk.set(0, j, dbk.get(0, j) + sk);
        dbv.set(0, j, dbv.get(0, j) + sv);
      }
      
      for (let t = 0; t < seqLen; t++)
        for (let k = 0; k < d; k++)
          dInput.set(b, t * d + k, dSeq.get(t, k));
    }
    
    this.dWeights = { dWq, dWk, dWv, dWo };
    this.dBiases = { dbq, dbk, dbv, dbo };
    return dInput;
  }
  
  updateWeights(lr) {
    if (!this.dWeights) return;
    const { dWq, dWk, dWv, dWo } = this.dWeights;
    const { dbq, dbk, dbv, dbo } = this.dBiases;
    this.Wq = this.Wq.sub(dWq.mul(lr));
    this.Wk = this.Wk.sub(dWk.mul(lr));
    this.Wv = this.Wv.sub(dWv.mul(lr));
    this.Wo = this.Wo.sub(dWo.mul(lr));
    this.bq = this.bq.sub(dbq.mul(lr));
    this.bk = this.bk.sub(dbk.mul(lr));
    this.bv = this.bv.sub(dbv.mul(lr));
    this.bo = this.bo.sub(dbo.mul(lr));
  }
  
  paramCount() {
    return 4 * this.dModel * this.dModel + 4 * this.dModel; // 4 weight matrices + 4 bias vectors
  }
}
