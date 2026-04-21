// flash-attention.js — Simplified Flash Attention Implementation
// Paper: "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)
//
// Standard attention: O(N²) memory (stores full attention matrix)
// Flash attention: O(N) memory (computes attention in tiles using online softmax)
//
// Key insight: We never need to materialize the full N×N attention matrix.
// Instead, compute softmax incrementally across tiles using the "online softmax" trick:
// - Track running maximum and running sum
// - Rescale previous partial results when new maximum is found
//
// This is a simplified JS implementation for educational purposes.
// Real Flash Attention also optimizes GPU SRAM/HBM memory hierarchy.

import { Matrix } from './matrix.js';

/**
 * Standard attention (for correctness comparison).
 * O(N²) memory — materializes full attention matrix.
 *
 * @param {Matrix} Q - [seqLen, headDim]
 * @param {Matrix} K - [seqLen, headDim]
 * @param {Matrix} V - [seqLen, headDim]
 * @param {boolean} causal - apply causal mask
 * @returns {{ output: Matrix, stats: object }}
 */
export function standardAttention(Q, K, V, causal = false) {
  const N = Q.rows;
  const d = Q.cols;
  const scale = 1 / Math.sqrt(d);

  // Compute full attention matrix: O(N²) memory
  const scores = Q.dot(K.T()).mul(scale); // [N, N]

  if (causal) {
    for (let i = 0; i < N; i++)
      for (let j = i + 1; j < N; j++)
        scores.set(i, j, -1e9);
  }

  // Softmax each row
  const attn = new Matrix(N, N);
  for (let i = 0; i < N; i++) {
    let max = -Infinity;
    for (let j = 0; j < N; j++) max = Math.max(max, scores.get(i, j));
    let sum = 0;
    for (let j = 0; j < N; j++) {
      attn.set(i, j, Math.exp(scores.get(i, j) - max));
      sum += attn.get(i, j);
    }
    for (let j = 0; j < N; j++) attn.set(i, j, attn.get(i, j) / sum);
  }

  const output = attn.dot(V); // [N, d]

  return {
    output,
    stats: {
      peakMemory: N * N, // the attention matrix
      method: 'standard',
    }
  };
}

/**
 * Flash Attention — tiled computation with online softmax.
 * O(N * tileSize) memory instead of O(N²).
 *
 * Algorithm (per query tile):
 * 1. Initialize output O = 0, running max m = -inf, running sum l = 0
 * 2. For each KV tile:
 *    a. Compute tile scores: S_tile = Q_tile · K_tile^T / sqrt(d)
 *    b. Apply causal mask if needed
 *    c. Find new max: m_new = max(m_old, max(S_tile))
 *    d. Rescale old partial: l_old = l_old * exp(m_old - m_new)
 *    e. Compute new exponentials: exp_tile = exp(S_tile - m_new)
 *    f. Update: O = O * exp(m_old - m_new) + exp_tile · V_tile
 *    g. Update: l = l_old + sum(exp_tile)
 * 3. Final: O = O / l
 *
 * @param {Matrix} Q - [seqLen, headDim]
 * @param {Matrix} K - [seqLen, headDim]
 * @param {Matrix} V - [seqLen, headDim]
 * @param {boolean} causal - apply causal mask
 * @param {number} tileSize - tile/block size (default: sqrt(N) or 16)
 * @returns {{ output: Matrix, stats: object }}
 */
export function flashAttention(Q, K, V, causal = false, tileSize = 0) {
  const N = Q.rows;
  const d = Q.cols;
  const scale = 1 / Math.sqrt(d);

  if (tileSize <= 0) tileSize = Math.max(1, Math.min(16, Math.ceil(Math.sqrt(N))));

  const output = new Matrix(N, d);

  // Per-row: running max, running sum
  const m = new Float64Array(N).fill(-Infinity); // running max of scores
  const l = new Float64Array(N).fill(0);         // running sum of exp(scores - m)

  let peakTileMemory = 0;

  // Iterate over KV tiles
  const numKVTiles = Math.ceil(N / tileSize);

  for (let kj = 0; kj < numKVTiles; kj++) {
    const kvStart = kj * tileSize;
    const kvEnd = Math.min(kvStart + tileSize, N);
    const kvLen = kvEnd - kvStart;

    // Extract K_tile, V_tile: [kvLen, d]
    const Kt = subMatrix(K, kvStart, kvEnd);
    const Vt = subMatrix(V, kvStart, kvEnd);

    // For each query position (could be tiled too, but row-wise is simpler)
    for (let qi = 0; qi < N; qi++) {
      // Compute scores: Q[qi] · K_tile^T → [kvLen]
      const scores = new Float64Array(kvLen);
      for (let j = 0; j < kvLen; j++) {
        let dot = 0;
        for (let dd = 0; dd < d; dd++) dot += Q.get(qi, dd) * Kt.get(j, dd);
        scores[j] = dot * scale;
      }

      // Causal mask: mask out positions where kv_pos > query_pos
      if (causal) {
        for (let j = 0; j < kvLen; j++) {
          if (kvStart + j > qi) scores[j] = -1e9;
        }
      }

      // Online softmax update
      const m_old = m[qi];
      let m_new = m_old;
      for (let j = 0; j < kvLen; j++) m_new = Math.max(m_new, scores[j]);

      // Rescale old partial
      const rescale = Math.exp(m_old - m_new);
      l[qi] *= rescale;

      // Rescale old output
      for (let dd = 0; dd < d; dd++) {
        output.set(qi, dd, output.get(qi, dd) * rescale);
      }

      // Accumulate new tile
      for (let j = 0; j < kvLen; j++) {
        const expScore = Math.exp(scores[j] - m_new);
        l[qi] += expScore;
        for (let dd = 0; dd < d; dd++) {
          output.set(qi, dd, output.get(qi, dd) + expScore * Vt.get(j, dd));
        }
      }

      m[qi] = m_new;
    }

    peakTileMemory = Math.max(peakTileMemory, kvLen); // scores per query
  }

  // Final normalization: O = O / l
  for (let i = 0; i < N; i++) {
    if (l[i] > 0) {
      for (let dd = 0; dd < d; dd++) {
        output.set(i, dd, output.get(i, dd) / l[i]);
      }
    }
  }

  return {
    output,
    stats: {
      peakMemory: N * tileSize, // much less than N²
      tileSize,
      numTiles: numKVTiles,
      method: 'flash',
    }
  };
}

/**
 * Extract a sub-matrix (rows from startRow to endRow).
 */
function subMatrix(mat, startRow, endRow) {
  const rows = endRow - startRow;
  const result = new Matrix(rows, mat.cols);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < mat.cols; c++)
      result.set(r, c, mat.get(startRow + r, c));
  return result;
}
