// label-smoothing.js — Label smoothing for cross-entropy loss
// Paper: "Rethinking the Inception Architecture" (Szegedy et al., 2016)

import { Matrix } from './matrix.js';

/**
 * Label Smoothing Cross-Entropy Loss.
 * 
 * Instead of hard one-hot targets [0, 0, 1, 0]:
 *   soft_target[i] = (1 - ε) * one_hot[i] + ε / K
 * 
 * Where ε is the smoothing factor and K is the number of classes.
 * 
 * Benefits:
 * - Prevents model from becoming overconfident
 * - Acts as a regularizer
 * - Improves calibration
 * 
 * @param {number} smoothing - Smoothing factor (default 0.1)
 * @param {number} numClasses - Number of classes
 */
export class LabelSmoothingCrossEntropy {
  constructor(smoothing = 0.1, numClasses = null) {
    this.smoothing = smoothing;
    this.numClasses = numClasses;
  }

  /**
   * Compute label-smoothed cross-entropy loss.
   * @param {Float64Array|Array} logits - Raw logits (pre-softmax)
   * @param {number} target - True class index
   * @returns {{ loss: number, dLogits: Float64Array }}
   */
  forward(logits, target) {
    const K = this.numClasses || logits.length;
    const eps = this.smoothing;
    
    // Compute softmax
    const maxLogit = Math.max(...logits);
    const expLogits = new Float64Array(K);
    let sumExp = 0;
    for (let i = 0; i < K; i++) {
      expLogits[i] = Math.exp(logits[i] - maxLogit);
      sumExp += expLogits[i];
    }
    const probs = new Float64Array(K);
    for (let i = 0; i < K; i++) {
      probs[i] = expLogits[i] / sumExp;
    }
    
    // Smooth targets: (1 - eps) * one_hot + eps / K
    const smoothTargets = new Float64Array(K);
    for (let i = 0; i < K; i++) {
      smoothTargets[i] = eps / K;
    }
    smoothTargets[target] += (1 - eps);
    
    // Cross-entropy loss: -sum(target * log(prob))
    let loss = 0;
    for (let i = 0; i < K; i++) {
      if (smoothTargets[i] > 0) {
        loss -= smoothTargets[i] * Math.log(Math.max(probs[i], 1e-10));
      }
    }
    
    // Gradient: prob - smooth_target
    const dLogits = new Float64Array(K);
    for (let i = 0; i < K; i++) {
      dLogits[i] = probs[i] - smoothTargets[i];
    }
    
    return { loss, dLogits, probs };
  }

  /**
   * Batch version: compute loss over a sequence of logits and targets.
   * @param {Matrix} logitsMatrix - (seqLen x vocabSize) logits
   * @param {Array<number>} targets - Target indices
   * @returns {{ loss: number, dLogits: Matrix }}
   */
  batchForward(logitsMatrix, targets) {
    const seqLen = targets.length;
    const K = logitsMatrix.cols;
    if (!this.numClasses) this.numClasses = K;
    
    let totalLoss = 0;
    const dLogits = new Matrix(seqLen, K);
    
    for (let t = 0; t < seqLen; t++) {
      const logits = new Float64Array(K);
      for (let i = 0; i < K; i++) logits[i] = logitsMatrix.get(t, i);
      
      const { loss, dLogits: dL } = this.forward(logits, targets[t]);
      totalLoss += loss;
      
      for (let i = 0; i < K; i++) {
        dLogits.set(t, i, dL[i]);
      }
    }
    
    return { loss: totalLoss / seqLen, dLogits };
  }
}

/**
 * Convenience function: create a label smoothing loss.
 * @param {number} smoothing - Smoothing factor
 * @returns {function} Loss function compatible with CharLM
 */
export function createLabelSmoothingLoss(smoothing = 0.1) {
  return function labelSmoothingLoss(logitsMatrix, targets) {
    const ls = new LabelSmoothingCrossEntropy(smoothing, logitsMatrix.cols);
    return ls.batchForward(logitsMatrix, targets);
  };
}
