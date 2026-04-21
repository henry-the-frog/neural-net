// moe.js — Mixture of Experts (MoE) Feed-Forward Layer
// Used in: Mixtral 8x7B, GPT-4 (rumored), Switch Transformer, etc.
//
// Instead of a single large FFN, use N expert FFNs and a router that
// selects the top-K experts for each token. This allows massive model
// capacity (total params) while keeping compute cost per token low
// (only K experts activated per token).
//
// Mixtral: 8 experts, top-2 routing → 2/8 = 25% compute of dense model
// Switch Transformer: top-1 routing → even more efficient

import { Matrix } from './matrix.js';
import { SwiGLUFFN } from './modern-decoder.js';

/**
 * Softmax over an array.
 */
function softmaxArr(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

/**
 * MoE Layer: routes each token to top-K experts.
 *
 * Architecture:
 *   1. Router: linear projection from dModel → numExperts (learned)
 *   2. Top-K selection: pick K experts with highest router scores
 *   3. Expert computation: run each selected expert's FFN
 *   4. Weighted sum: combine expert outputs using softmax(router scores)
 *
 * @param {number} dModel - model dimension
 * @param {number} numExperts - total number of experts
 * @param {number} topK - number of experts to activate per token
 * @param {number} dHidden - hidden dimension per expert FFN
 */
export class MixtureOfExperts {
  constructor(dModel, numExperts, topK = 2, dHidden = null) {
    if (topK > numExperts) throw new Error('topK must be <= numExperts');
    if (topK < 1) throw new Error('topK must be >= 1');

    this.dModel = dModel;
    this.numExperts = numExperts;
    this.topK = topK;
    this.outputSize = dModel;

    // Router: dModel → numExperts
    const scale = Math.sqrt(2 / (dModel + numExperts));
    this.routerW = Matrix.random(dModel, numExperts).mul(scale);
    this.routerB = Matrix.zeros(1, numExperts);

    // Expert FFNs
    this.experts = [];
    for (let i = 0; i < numExperts; i++) {
      this.experts.push(new SwiGLUFFN(dModel, dHidden));
    }

    // Stats tracking
    this._routingStats = new Array(numExperts).fill(0);
  }

  /**
   * Forward pass: route each token to top-K experts.
   * @param {Matrix} input - [numTokens, dModel]
   * @returns {Matrix} output - [numTokens, dModel]
   */
  forward(input) {
    const N = input.rows;
    const output = new Matrix(N, this.dModel);

    for (let t = 0; t < N; t++) {
      // Extract token vector
      const token = new Matrix(1, this.dModel);
      for (let d = 0; d < this.dModel; d++) token.set(0, d, input.get(t, d));

      // Router scores
      const logits = token.dot(this.routerW);
      for (let e = 0; e < this.numExperts; e++) {
        logits.set(0, e, logits.get(0, e) + this.routerB.get(0, e));
      }

      // Top-K selection
      const scores = [];
      for (let e = 0; e < this.numExperts; e++) {
        scores.push({ expertIdx: e, score: logits.get(0, e) });
      }
      scores.sort((a, b) => b.score - a.score);
      const topExperts = scores.slice(0, this.topK);

      // Softmax over selected experts' scores
      const topScores = topExperts.map(e => e.score);
      const weights = softmaxArr(topScores);

      // Compute weighted sum of expert outputs
      for (let k = 0; k < this.topK; k++) {
        const expertIdx = topExperts[k].expertIdx;
        const weight = weights[k];
        this._routingStats[expertIdx]++;

        const expertOut = this.experts[expertIdx].forward(token);
        for (let d = 0; d < this.dModel; d++) {
          output.set(t, d, output.get(t, d) + weight * expertOut.get(0, d));
        }
      }
    }

    return output;
  }

  /**
   * Get routing statistics: how often each expert was selected.
   */
  routingStats() {
    const total = this._routingStats.reduce((a, b) => a + b, 0);
    return this._routingStats.map((count, idx) => ({
      expert: idx,
      count,
      pct: total > 0 ? (count / total * 100).toFixed(1) + '%' : '0%',
    }));
  }

  /**
   * Reset routing stats.
   */
  resetStats() {
    this._routingStats.fill(0);
  }

  /**
   * Load balancing loss (auxiliary loss for training).
   * Penalizes uneven expert utilization.
   * L_balance = numExperts * Σ(f_i * P_i) where f_i is fraction of tokens routed
   * to expert i and P_i is mean router probability for expert i.
   */
  loadBalanceLoss() {
    const total = this._routingStats.reduce((a, b) => a + b, 0);
    if (total === 0) return 0;
    const fractions = this._routingStats.map(c => c / total);
    // Ideal: each expert gets 1/numExperts fraction
    // Penalize deviation from uniform
    let loss = 0;
    for (const f of fractions) {
      loss += f * f;
    }
    return this.numExperts * loss;
  }

  /**
   * Parameter count: router + all experts.
   */
  paramCount() {
    let count = this.routerW.rows * this.routerW.cols; // router
    for (const expert of this.experts) {
      count += expert.W1.rows * expert.W1.cols;
      count += expert.W2.rows * expert.W2.cols;
      count += expert.W3.rows * expert.W3.cols;
    }
    return count;
  }

  /**
   * Active parameters per token (only topK experts).
   */
  activeParamsPerToken() {
    const routerParams = this.routerW.rows * this.routerW.cols;
    const expertParams = this.experts[0].W1.rows * this.experts[0].W1.cols * 3; // approx
    return routerParams + this.topK * expertParams;
  }
}
