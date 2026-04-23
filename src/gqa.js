// gqa.js — Grouped Query Attention (Ainslie et al., 2023)
// Shares KV heads across groups of query heads to reduce memory/compute.
//
// MHA: nKVHeads = nHeads (standard multi-head attention)
// GQA: nKVHeads < nHeads (grouped query attention)
// MQA: nKVHeads = 1 (multi-query attention)

import { Matrix } from './matrix.js';

/**
 * Grouped Query Attention.
 * @param {Matrix} Q - Queries (seqLen × dModel)
 * @param {Matrix} K - Keys (seqLen × dKV) where dKV = nKVHeads * headDim
 * @param {Matrix} V - Values (seqLen × dKV)
 * @param {number} nHeads - Number of query heads
 * @param {number} nKVHeads - Number of KV heads (must divide nHeads evenly)
 * @param {boolean} causal - Apply causal mask
 * @returns {Matrix} Output (seqLen × dModel)
 */
export function groupedQueryAttention(Q, K, V, nHeads, nKVHeads, causal = false) {
  const seqLen = Q.rows;
  const dModel = Q.cols;
  const headDim = dModel / nHeads;
  const kvHeadDim = K.cols / nKVHeads;
  
  if (dModel % nHeads !== 0) throw new Error(`dModel (${dModel}) must be divisible by nHeads (${nHeads})`);
  if (K.cols % nKVHeads !== 0) throw new Error(`K dim (${K.cols}) must be divisible by nKVHeads (${nKVHeads})`);
  if (nHeads % nKVHeads !== 0) throw new Error(`nHeads (${nHeads}) must be divisible by nKVHeads (${nKVHeads})`);
  if (headDim !== kvHeadDim) throw new Error(`headDim (${headDim}) must equal kvHeadDim (${kvHeadDim})`);
  
  const groupSize = nHeads / nKVHeads; // Q heads per KV head
  const scale = 1.0 / Math.sqrt(headDim);
  const output = new Matrix(seqLen, dModel);
  
  for (let h = 0; h < nHeads; h++) {
    const kvHead = Math.floor(h / groupSize); // Which KV head this Q head shares
    
    // For each position, compute attention
    for (let i = 0; i < seqLen; i++) {
      // Compute attention scores: Q[i] @ K[j]^T for all j
      let maxScore = -Infinity;
      const scores = new Float64Array(seqLen);
      
      for (let j = 0; j < seqLen; j++) {
        if (causal && j > i) { scores[j] = -Infinity; continue; }
        let dot = 0;
        for (let d = 0; d < headDim; d++) {
          dot += Q.get(i, h * headDim + d) * K.get(j, kvHead * headDim + d);
        }
        scores[j] = dot * scale;
        maxScore = Math.max(maxScore, scores[j]);
      }
      
      // Softmax
      let sumExp = 0;
      for (let j = 0; j < seqLen; j++) {
        scores[j] = Math.exp(scores[j] - maxScore);
        sumExp += scores[j];
      }
      
      // Weighted sum of V
      for (let d = 0; d < headDim; d++) {
        let val = 0;
        for (let j = 0; j < seqLen; j++) {
          val += (scores[j] / sumExp) * V.get(j, kvHead * headDim + d);
        }
        output.set(i, h * headDim + d, val);
      }
    }
  }
  
  return output;
}

/**
 * Convenience: compute KV dimensions for GQA.
 * @param {number} dModel - Model dimension
 * @param {number} nHeads - Number of query heads
 * @param {number} nKVHeads - Number of KV heads
 * @returns {{ dQ: number, dKV: number, headDim: number, groupSize: number, kvSaving: string }}
 */
export function gqaDimensions(dModel, nHeads, nKVHeads) {
  const headDim = dModel / nHeads;
  const dKV = nKVHeads * headDim;
  const groupSize = nHeads / nKVHeads;
  const kvSaving = ((1 - nKVHeads / nHeads) * 100).toFixed(1);
  
  return {
    dQ: dModel,
    dKV,
    headDim,
    groupSize,
    kvSaving: `${kvSaving}% KV memory saved`,
  };
}
