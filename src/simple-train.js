// simple-train.js — Simple training for ModernDecoder via numerical gradients
// This is EXTREMELY slow but correct. For educational purposes only.
// Real training would need analytical gradients through all layers.

import { ModernDecoder } from './modern-decoder.js';
import { softmax } from './sampling.js';
import { Matrix } from './matrix.js';

/**
 * Compute cross-entropy loss for next-token prediction.
 *
 * @param {ModernDecoder} model
 * @param {number[][]} sequences - array of token ID sequences
 * @returns {number} average loss
 */
export function computeLoss(model, sequences) {
  let totalLoss = 0;
  let totalTokens = 0;

  for (const seq of sequences) {
    if (seq.length < 2) continue;

    // Input: all tokens except last
    const input = [seq.slice(0, -1)];
    const logits = model.forward(input, false);
    const seqLen = seq.length - 1;

    // For each position, compute cross-entropy of predicted next token
    for (let t = 0; t < seqLen; t++) {
      const targetToken = seq[t + 1];
      const posLogits = new Float64Array(model.vocabSize);
      for (let v = 0; v < model.vocabSize; v++) {
        posLogits[v] = logits.get(0, t * model.vocabSize + v);
      }

      const probs = softmax(posLogits);
      const loss = -Math.log(Math.max(probs[targetToken], 1e-15));
      totalLoss += loss;
      totalTokens++;
    }
  }

  return totalTokens > 0 ? totalLoss / totalTokens : 0;
}

/**
 * Train model for one step using parameter perturbation (SPSA-like).
 * This is a gradient-free optimization method — extremely simple.
 *
 * @param {ModernDecoder} model
 * @param {number[][]} sequences - training data
 * @param {number} lr - learning rate
 * @param {number} perturbation - perturbation size
 * @returns {number} loss after update
 */
export function trainStepSPSA(model, sequences, lr = 0.01, perturbation = 0.001) {
  // Collect all trainable weight matrices
  const params = collectParams(model);

  // For each parameter matrix, perturb and measure loss gradient
  const baseLoss = computeLoss(model, sequences);

  for (const { mat, row, col } of randomSubsetOfParams(params, 50)) {
    const original = mat.get(row, col);

    // Forward perturbation
    mat.set(row, col, original + perturbation);
    model.clearCache();
    const lossPlus = computeLoss(model, sequences);

    // Backward perturbation
    mat.set(row, col, original - perturbation);
    model.clearCache();
    const lossMinus = computeLoss(model, sequences);

    // Estimate gradient and update
    const grad = (lossPlus - lossMinus) / (2 * perturbation);
    mat.set(row, col, original - lr * grad);
  }

  model.clearCache();
  return computeLoss(model, sequences);
}

/**
 * Collect all weight matrices from the model.
 */
function collectParams(model) {
  const params = [];

  // Embedding
  params.push(model.embedding);

  // Each decoder block
  for (const block of model.blocks) {
    const attn = block.attention;
    params.push(attn.Wq, attn.Wk, attn.Wv, attn.Wo);
    params.push(block.ffn.W1, block.ffn.W2, block.ffn.W3);
  }

  // Output projection
  params.push(model.outputProj);

  return params;
}

/**
 * Get a random subset of individual parameters to update (for efficiency).
 */
function* randomSubsetOfParams(matrices, count) {
  const allParams = [];
  for (const mat of matrices) {
    for (let r = 0; r < mat.rows; r++) {
      for (let c = 0; c < mat.cols; c++) {
        allParams.push({ mat, row: r, col: c });
      }
    }
  }

  // Shuffle and take first `count`
  for (let i = allParams.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [allParams[i], allParams[j]] = [allParams[j], allParams[i]];
  }

  const n = Math.min(count, allParams.length);
  for (let i = 0; i < n; i++) yield allParams[i];
}
