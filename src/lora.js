// lora.js — LoRA (Low-Rank Adaptation) for Efficient Fine-Tuning
// Paper: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
//
// Key idea: Instead of fine-tuning all W (dModel × dModel) parameters,
// decompose the update as: W' = W + BA
// where B is (dModel × r) and A is (r × dModel), with r << dModel.
//
// This reduces trainable parameters from d² to 2dr.
// For r=8, dModel=4096: 65M → 65K params (1000x reduction!)
//
// LoRA adapters are small, composable, and can be swapped at inference time.

import { Matrix } from './matrix.js';

/**
 * LoRA adapter for a single weight matrix.
 *
 * Forward: y = x(W + α/r · BA)
 * where W is frozen, B and A are trainable, α is a scaling factor.
 *
 * @param {number} dIn - input dimension
 * @param {number} dOut - output dimension
 * @param {number} rank - LoRA rank (r)
 * @param {number} alpha - scaling factor (default: rank)
 */
export class LoRAAdapter {
  constructor(dIn, dOut, rank, alpha = null) {
    this.dIn = dIn;
    this.dOut = dOut;
    this.rank = rank;
    this.alpha = alpha ?? rank;
    this.scaling = this.alpha / this.rank;

    // A: (rank, dOut) — initialized with Kaiming
    this.A = Matrix.random(rank, dOut).mul(Math.sqrt(2 / (rank + dOut)));
    // B: (dIn, rank) — initialized to zeros (so initial adapter is identity)
    this.B = Matrix.zeros(dIn, rank);

    this.enabled = true;
  }

  /**
   * Compute LoRA delta: scaling * B · A
   * @returns {Matrix} [dIn, dOut] matrix to add to base weight
   */
  delta() {
    if (!this.enabled) return Matrix.zeros(this.dIn, this.dOut);
    return this.B.dot(this.A).mul(this.scaling);
  }

  /**
   * Apply LoRA to input: x · (W + delta)
   * @param {Matrix} input - [batch, dIn]
   * @param {Matrix} baseWeight - [dIn, dOut] frozen base weight
   * @returns {Matrix} [batch, dOut]
   */
  forward(input, baseWeight) {
    // Base computation: x · W
    const base = input.dot(baseWeight);
    
    if (!this.enabled) return base;

    // LoRA computation: x · B · A · scaling
    const loraOut = input.dot(this.B).dot(this.A).mul(this.scaling);

    // Add
    const result = new Matrix(base.rows, base.cols);
    for (let r = 0; r < base.rows; r++)
      for (let c = 0; c < base.cols; c++)
        result.set(r, c, base.get(r, c) + loraOut.get(r, c));

    return result;
  }

  /**
   * Merge LoRA into base weight (for inference).
   * After merge, no runtime cost — the adapter is "baked in".
   * @param {Matrix} baseWeight
   * @returns {Matrix} merged weight
   */
  merge(baseWeight) {
    const d = this.delta();
    const merged = new Matrix(baseWeight.rows, baseWeight.cols);
    for (let r = 0; r < baseWeight.rows; r++)
      for (let c = 0; c < baseWeight.cols; c++)
        merged.set(r, c, baseWeight.get(r, c) + d.get(r, c));
    return merged;
  }

  /**
   * Trainable parameter count (much smaller than full weight).
   */
  paramCount() {
    return this.B.rows * this.B.cols + this.A.rows * this.A.cols;
  }

  /**
   * Full weight parameter count (what we'd need without LoRA).
   */
  fullParamCount() {
    return this.dIn * this.dOut;
  }

  /**
   * Compression ratio: full params / LoRA params.
   */
  compressionRatio() {
    return this.fullParamCount() / this.paramCount();
  }

  /**
   * Export adapter weights for saving.
   */
  export() {
    return {
      rank: this.rank,
      alpha: this.alpha,
      A: matrixToArray(this.A),
      B: matrixToArray(this.B),
    };
  }

  /**
   * Import adapter weights.
   */
  static import(data, dIn, dOut) {
    const adapter = new LoRAAdapter(dIn, dOut, data.rank, data.alpha);
    adapter.A = arrayToMatrix(data.A, data.rank, dOut);
    adapter.B = arrayToMatrix(data.B, dIn, data.rank);
    return adapter;
  }
}

/**
 * LoRA configuration for a model: which layers to adapt.
 * Common targets: Q, K, V, O projections in attention.
 */
export class LoRAConfig {
  constructor(rank = 8, alpha = null, targets = ['Wq', 'Wv']) {
    this.rank = rank;
    this.alpha = alpha ?? rank;
    this.targets = targets;
  }

  /**
   * Total trainable LoRA params for a given model config.
   */
  estimateParams(dModel, numLayers) {
    const paramsPerAdapter = 2 * dModel * this.rank; // B: d×r + A: r×d
    return paramsPerAdapter * this.targets.length * numLayers;
  }

  /**
   * Full model params (for comparison).
   */
  estimateFullParams(dModel, numLayers) {
    const paramsPerWeight = dModel * dModel;
    return paramsPerWeight * this.targets.length * numLayers;
  }
}

// --- Helpers ---

function matrixToArray(mat) {
  const data = [];
  for (let r = 0; r < mat.rows; r++)
    for (let c = 0; c < mat.cols; c++)
      data.push(mat.get(r, c));
  return data;
}

function arrayToMatrix(arr, rows, cols) {
  const mat = new Matrix(rows, cols);
  for (let i = 0; i < arr.length; i++) {
    mat.set(Math.floor(i / cols), i % cols, arr[i]);
  }
  return mat;
}
