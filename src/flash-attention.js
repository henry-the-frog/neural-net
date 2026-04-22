// flash-attention.js — Flash Attention: O(N) memory attention via tiled computation
//
// Implements the Flash Attention algorithm (Dao et al., 2022):
// Instead of materializing the full N×N attention matrix, processes Q×K in tiles
// using the online softmax trick to maintain running statistics.
//
// Memory: O(N·d + blockSize²) instead of O(N²)
// Compute: Same O(N²·d) as standard attention
// Result: Numerically identical to standard attention (within floating-point tolerance)

import { Matrix } from './matrix.js';

// In-place matrix addition (Matrix.add creates new matrix)
function addInPlace(target, source) {
  for (let i = 0; i < target.data.length; i++) target.data[i] += source.data[i];
}

// Serialization helpers
function matToJSON(m) {
  return { rows: m.rows, cols: m.cols, data: Array.from(m.data) };
}
function matFromJSON(j) {
  return new Matrix(j.rows, j.cols, new Float64Array(j.data));
}

/**
 * Flash Attention — memory-efficient scaled dot-product attention.
 * 
 * Algorithm:
 * For each block of query rows (Qi):
 *   For each block of key rows (Kj):
 *     1. Compute Sij = Qi × Kj^T / sqrt(d)     (blockSize × blockSize tile)
 *     2. Update running max, sum, and output using online softmax
 *   Final output rows are correctly normalized
 */
export class FlashAttention {
  /**
   * @param {number} dModel — dimension of Q, K, V vectors
   * @param {Object} opts
   * @param {number} [opts.blockSize=32] — tile size for blocked computation
   * @param {boolean} [opts.causal=false] — apply causal mask
   */
  constructor(dModel, { blockSize = 32, causal = false, dropout = 0 } = {}) {
    this.dModel = dModel;
    this.scale = 1 / Math.sqrt(dModel);
    this.blockSize = blockSize;
    this.causal = causal;
    this.dropoutRate = dropout;
    
    // Weight matrices (same as SelfAttention)
    const s = Math.sqrt(2 / (dModel + dModel));
    this.Wq = Matrix.random(dModel, dModel).mul(s);
    this.Wk = Matrix.random(dModel, dModel).mul(s);
    this.Wv = Matrix.random(dModel, dModel).mul(s);
    this.Wo = Matrix.random(dModel, dModel).mul(s);
    
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
  
  /**
   * Forward pass using flash attention algorithm.
   * @param {Matrix} input — [batch, seqLen * dModel]
   * @returns {Matrix} — [batch, seqLen * dModel]
   */
  forward(input) {
    const batchSize = input.rows;
    const seqLen = Math.floor(input.cols / this.dModel);
    const d = this.dModel;
    const Bs = Math.min(this.blockSize, seqLen);
    
    const allSeqs = [], allQ = [], allK = [], allV = [];
    const allOutput = []; // Store per-batch output for backward
    const allLse = []; // Log-sum-exp for backward
    
    const result = new Matrix(batchSize, seqLen * d);
    
    for (let b = 0; b < batchSize; b++) {
      // Extract sequence: [seqLen, dModel]
      const seq = new Matrix(seqLen, d);
      for (let t = 0; t < seqLen; t++)
        for (let k = 0; k < d; k++)
          seq.set(t, k, input.get(b, t * d + k));
      allSeqs.push(seq);
      
      // Linear projections
      const Q = seq.dot(this.Wq).add(this.bq); // [seqLen, d]
      const K = seq.dot(this.Wk).add(this.bk);
      const V = seq.dot(this.Wv).add(this.bv);
      allQ.push(Q); allK.push(K); allV.push(V);
      
      // Flash attention: tiled computation with online softmax
      // O[i] = sum_j (exp(S[i,j] - m[i]) * V[j]) / l[i]
      // where m[i] = max_j S[i,j] and l[i] = sum_j exp(S[i,j] - m[i])
      
      const O = new Matrix(seqLen, d); // Output accumulator
      const m = new Float64Array(seqLen).fill(-Infinity); // Running max per query
      const l = new Float64Array(seqLen).fill(0); // Running sum per query
      
      // Number of key blocks
      const numKBlocks = Math.ceil(seqLen / Bs);
      
      for (let jb = 0; jb < numKBlocks; jb++) {
        const jStart = jb * Bs;
        const jEnd = Math.min(jStart + Bs, seqLen);
        const jLen = jEnd - jStart;
        
        // Extract Kj block: [jLen, d]
        // Extract Vj block: [jLen, d]
        
        // For each query row, compute tile scores and update running stats
        for (let i = 0; i < seqLen; i++) {
          // Compute scores: S[i, jStart..jEnd] = Q[i] · Kj^T / sqrt(d)
          let newMax = m[i];
          const tileScores = new Float64Array(jLen);
          
          for (let j = 0; j < jLen; j++) {
            // Dot product Q[i] · K[jStart+j]
            let dot = 0;
            for (let k = 0; k < d; k++) {
              dot += Q.get(i, k) * K.get(jStart + j, k);
            }
            dot *= this.scale;
            
            // Causal mask: future positions get -Infinity
            if (this.causal && (jStart + j) > i) {
              dot = -Infinity;
            }
            
            tileScores[j] = dot;
            if (dot > newMax) newMax = dot;
          }
          
          // Skip if all scores are -Infinity (masked out)
          if (newMax === -Infinity && m[i] === -Infinity) continue;
          
          // Online softmax update
          // Rescale old accumulator: O[i] *= exp(m_old - m_new) * l_old
          // Then add new contributions
          
          const mOld = m[i];
          const mNew = newMax > mOld ? newMax : mOld;
          
          // Rescale factor for old accumulator
          const rescale = mOld === -Infinity ? 0 : Math.exp(mOld - mNew);
          const lOld = l[i] * rescale;
          
          // Compute new contributions
          let lNew = 0;
          // Rescale existing accumulator
          for (let k = 0; k < d; k++) O.set(i, k, O.get(i, k) * rescale);
          
          for (let j = 0; j < jLen; j++) {
            if (tileScores[j] === -Infinity) continue;
            const w = Math.exp(tileScores[j] - mNew);
            lNew += w;
            
            // Accumulate weighted V
            for (let k = 0; k < d; k++) {
              O.set(i, k, O.get(i, k) + w * V.get(jStart + j, k));
            }
          }
          
          m[i] = mNew;
          l[i] = lOld + lNew;
        }
      }
      
      // Final normalization: O[i] /= l[i]
      for (let i = 0; i < seqLen; i++) {
        if (l[i] > 0) {
          for (let k = 0; k < d; k++) {
            O.set(i, k, O.get(i, k) / l[i]);
          }
        }
      }
      
      allOutput.push(O);
      allLse.push({ m, l }); // Store for backward
      
      // Output projection: O · Wo + bo
      const projected = O.dot(this.Wo).add(this.bo);
      for (let t = 0; t < seqLen; t++)
        for (let k = 0; k < d; k++)
          result.set(b, t * d + k, projected.get(t, k));
    }
    
    this._cache = { allSeqs, batchSize, seqLen, allQ, allK, allV, allOutput, allLse };
    return result;
  }
  
  /**
   * Backward pass.
   * Uses recomputation strategy (flash attention v2): recompute S tiles during backward
   * rather than storing the full attention matrix.
   */
  backward(dOutput) {
    const { allSeqs, batchSize, seqLen, allQ, allK, allV, allOutput, allLse } = this._cache;
    const d = this.dModel;
    const Bs = Math.min(this.blockSize, seqLen);
    
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
      const Q = allQ[b], K = allK[b], V = allV[b];
      const O = allOutput[b];
      const { m, l } = allLse[b];
      const seq = allSeqs[b];
      
      // Extract dOutput for this batch
      const dOut = new Matrix(seqLen, d);
      for (let t = 0; t < seqLen; t++)
        for (let k = 0; k < d; k++)
          dOut.set(t, k, dOutput.get(b, t * d + k));
      
      // Backward through output projection
      const dO = dOut.dot(this.Wo.T());
      addInPlace(dWo, O.T().dot(dOut));
      for (let j = 0; j < d; j++) {
        let sum = 0;
        for (let i = 0; i < seqLen; i++) sum += dOut.get(i, j);
        dbo.set(0, j, dbo.get(0, j) + sum);
      }
      
      // Precompute D[i] = sum_k dO[i,k] * O[i,k] (for softmax backward)
      const D = new Float64Array(seqLen);
      for (let i = 0; i < seqLen; i++) {
        let dot = 0;
        for (let k = 0; k < d; k++) dot += dO.get(i, k) * O.get(i, k);
        D[i] = dot;
      }
      
      // Backward through attention: recompute tiles
      const dQ = Matrix.zeros(seqLen, d);
      const dK = Matrix.zeros(seqLen, d);
      const dV = Matrix.zeros(seqLen, d);
      
      const numKBlocks = Math.ceil(seqLen / Bs);
      
      for (let jb = 0; jb < numKBlocks; jb++) {
        const jStart = jb * Bs;
        const jEnd = Math.min(jStart + Bs, seqLen);
        const jLen = jEnd - jStart;
        
        for (let i = 0; i < seqLen; i++) {
          if (l[i] === 0) continue;
          
          for (let j = 0; j < jLen; j++) {
            const jIdx = jStart + j;
            
            if (this.causal && jIdx > i) continue;
            
            // Recompute score
            let dot = 0;
            for (let k = 0; k < d; k++) dot += Q.get(i, k) * K.get(jIdx, k);
            dot *= this.scale;
            
            // Recompute attention weight: P[i,j] = exp(S[i,j] - m[i]) / l[i]
            const p = Math.exp(dot - m[i]) / l[i];
            
            // dP[i,j] = sum_k dO[i,k] * V[j,k]
            let dP = 0;
            for (let k = 0; k < d; k++) dP += dO.get(i, k) * V.get(jIdx, k);
            
            // dS[i,j] = P[i,j] * (dP - D[i])  (softmax backward)
            const dS = p * (dP - D[i]) * this.scale;
            
            // Accumulate gradients
            for (let k = 0; k < d; k++) {
              dQ.set(i, k, dQ.get(i, k) + dS * K.get(jIdx, k));
              dK.set(jIdx, k, dK.get(jIdx, k) + dS * Q.get(i, k));
              dV.set(jIdx, k, dV.get(jIdx, k) + p * dO.get(i, k));
            }
          }
        }
      }
      
      // Backward through projections: Q = seq · Wq + bq
      const dSeq = dQ.dot(this.Wq.T())
        .add(dK.dot(this.Wk.T()))
        .add(dV.dot(this.Wv.T()));
      
      addInPlace(dWq, seq.T().dot(dQ));
      addInPlace(dWk, seq.T().dot(dK));
      addInPlace(dWv, seq.T().dot(dV));
      
      for (let j = 0; j < d; j++) {
        let sq = 0, sk = 0, sv = 0;
        for (let i = 0; i < seqLen; i++) {
          sq += dQ.get(i, j);
          sk += dK.get(i, j);
          sv += dV.get(i, j);
        }
        dbq.set(0, j, dbq.get(0, j) + sq);
        dbk.set(0, j, dbk.get(0, j) + sk);
        dbv.set(0, j, dbv.get(0, j) + sv);
      }
      
      // Store dSeq back
      for (let t = 0; t < seqLen; t++)
        for (let k = 0; k < d; k++)
          dInput.set(b, t * d + k, dSeq.get(t, k));
    }
    
    this.dWeights = { dWq, dWk, dWv, dWo };
    this.dBiases = { dbq, dbk, dbv, dbo };
    return dInput;
  }
  
  /**
   * Update weights using gradients.
   */
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
  
  /**
   * Get parameters for serialization.
   */
  toJSON() {
    return {
      type: 'FlashAttention',
      dModel: this.dModel,
      blockSize: this.blockSize,
      causal: this.causal,
      Wq: matToJSON(this.Wq), Wk: matToJSON(this.Wk), Wv: matToJSON(this.Wv), Wo: matToJSON(this.Wo),
      bq: matToJSON(this.bq), bk: matToJSON(this.bk), bv: matToJSON(this.bv), bo: matToJSON(this.bo),
    };
  }
  
  static fromJSON(json) {
    const attn = new FlashAttention(json.dModel, { blockSize: json.blockSize, causal: json.causal });
    attn.Wq = matFromJSON(json.Wq); attn.Wk = matFromJSON(json.Wk);
    attn.Wv = matFromJSON(json.Wv); attn.Wo = matFromJSON(json.Wo);
    attn.bq = matFromJSON(json.bq); attn.bk = matFromJSON(json.bk);
    attn.bv = matFromJSON(json.bv); attn.bo = matFromJSON(json.bo);
    return attn;
  }
}

/**
 * Standalone flash attention function (no learned weights).
 * Computes: softmax(Q·K^T / sqrt(d)) · V using tiled online softmax.
 * 
 * @param {Matrix} Q — [seqLen, d]
 * @param {Matrix} K — [seqLen, d]  
 * @param {Matrix} V — [seqLen, d]
 * @param {Object} opts — { blockSize, causal }
 * @returns {Matrix} — [seqLen, d]
 */
export function flashAttention(Q, K, V, { blockSize = 32, causal = false } = {}) {
  const seqLen = Q.rows;
  const d = Q.cols;
  const scale = 1 / Math.sqrt(d);
  const Bs = Math.min(blockSize, seqLen);
  
  const O = new Matrix(seqLen, d);
  const m = new Float64Array(seqLen).fill(-Infinity);
  const l = new Float64Array(seqLen).fill(0);
  
  const numKBlocks = Math.ceil(seqLen / Bs);
  
  for (let jb = 0; jb < numKBlocks; jb++) {
    const jStart = jb * Bs;
    const jEnd = Math.min(jStart + Bs, seqLen);
    const jLen = jEnd - jStart;
    
    for (let i = 0; i < seqLen; i++) {
      let newMax = m[i];
      const scores = new Float64Array(jLen);
      
      for (let j = 0; j < jLen; j++) {
        if (causal && (jStart + j) > i) {
          scores[j] = -Infinity;
          continue;
        }
        let dot = 0;
        for (let k = 0; k < d; k++) dot += Q.get(i, k) * K.get(jStart + j, k);
        scores[j] = dot * scale;
        if (scores[j] > newMax) newMax = scores[j];
      }
      
      if (newMax === -Infinity && m[i] === -Infinity) continue;
      
      const mNew = newMax > m[i] ? newMax : m[i];
      const rescale = m[i] === -Infinity ? 0 : Math.exp(m[i] - mNew);
      const lOld = l[i] * rescale;
      
      let lNew = 0;
      // Rescale existing accumulator once
      for (let k = 0; k < d; k++) O.set(i, k, O.get(i, k) * rescale);
      
      for (let j = 0; j < jLen; j++) {
        if (scores[j] === -Infinity) continue;
        const w = Math.exp(scores[j] - mNew);
        lNew += w;
        for (let k = 0; k < d; k++) {
          O.set(i, k, O.get(i, k) + w * V.get(jStart + j, k));
        }
      }
      
      m[i] = mNew;
      l[i] = lOld + lNew;
    }
  }
  
  // Normalize
  for (let i = 0; i < seqLen; i++) {
    if (l[i] > 0) {
      for (let k = 0; k < d; k++) O.set(i, k, O.get(i, k) / l[i]);
    }
  }
  
  return O;
}
