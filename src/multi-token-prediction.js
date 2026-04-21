// multi-token-prediction.js — Multi-Token Prediction
// Paper: "Better & Faster Large Language Models via Multi-token Prediction" (Gloeckle et al., 2024)
//
// Standard LM: predict next token at each position
// Multi-token: predict next K tokens using K separate output heads
// Benefits: better representations, faster inference (can accept multiple tokens per step)

import { Matrix } from './matrix.js';
import { softmax } from './sampling.js';

/**
 * Multi-Token Prediction Heads
 *
 * Each head i predicts token at position t+i (for i=1..K).
 * All heads share the same hidden representation from the transformer.
 *
 * @param {number} dModel - model dimension
 * @param {number} vocabSize - vocabulary size
 * @param {number} numHeads - number of prediction heads (K)
 */
export class MultiTokenPredictionHeads {
  constructor(dModel, vocabSize, numHeads = 4) {
    this.dModel = dModel;
    this.vocabSize = vocabSize;
    this.numHeads = numHeads;

    // K output projection matrices (one per future position)
    this.heads = [];
    const scale = Math.sqrt(2 / (dModel + vocabSize));
    for (let i = 0; i < numHeads; i++) {
      this.heads.push(Matrix.random(dModel, vocabSize).mul(scale));
    }
  }

  /**
   * Forward: predict K future tokens for each position.
   *
   * @param {Matrix} hidden - [seqLen, dModel] hidden states from transformer
   * @returns {Matrix[]} K matrices, each [seqLen, vocabSize]
   */
  forward(hidden) {
    return this.heads.map(W => hidden.dot(W));
  }

  /**
   * Compute multi-token prediction loss.
   * For each head i, the target at position t is token[t+i].
   *
   * @param {Matrix} hidden - [seqLen, dModel]
   * @param {number[]} tokens - full sequence (seqLen + numHeads tokens needed)
   * @returns {{ loss: number, perHeadLoss: number[] }}
   */
  computeLoss(hidden, tokens) {
    const allLogits = this.forward(hidden);
    const seqLen = hidden.rows;
    let totalLoss = 0;
    const perHeadLoss = [];

    for (let h = 0; h < this.numHeads; h++) {
      const logits = allLogits[h];
      let headLoss = 0;
      let count = 0;

      for (let t = 0; t < seqLen; t++) {
        const targetIdx = t + h + 1; // predict token at t+h+1
        if (targetIdx >= tokens.length) break;

        const posLogits = new Float64Array(this.vocabSize);
        for (let v = 0; v < this.vocabSize; v++) {
          posLogits[v] = logits.get(t, v);
        }
        const probs = softmax(posLogits);
        const target = tokens[targetIdx];
        headLoss += -Math.log(Math.max(probs[target], 1e-15));
        count++;
      }

      const avgLoss = count > 0 ? headLoss / count : 0;
      perHeadLoss.push(avgLoss);
      totalLoss += avgLoss;
    }

    return {
      loss: totalLoss / this.numHeads, // average across heads
      perHeadLoss,
    };
  }

  /**
   * Use multi-token prediction for faster inference.
   * Returns K candidate tokens at once.
   *
   * @param {Matrix} hidden - [1, dModel] last position hidden state
   * @returns {number[]} K predicted tokens
   */
  predictMultiple(hidden) {
    const candidates = [];
    for (const W of this.heads) {
      const logits = hidden.dot(W);
      let maxIdx = 0, maxVal = -Infinity;
      for (let v = 0; v < this.vocabSize; v++) {
        if (logits.get(0, v) > maxVal) { maxVal = logits.get(0, v); maxIdx = v; }
      }
      candidates.push(maxIdx);
    }
    return candidates;
  }

  paramCount() {
    return this.numHeads * this.dModel * this.vocabSize;
  }
}
