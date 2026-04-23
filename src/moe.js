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
 * Simple expert FFN: Dense(inputSize → dHidden, ReLU) → Dense(dHidden → outputSize)
 */
class ExpertFFN {
  constructor(inputSize, dHidden, outputSize) {
    this.inputSize = inputSize;
    this.dHidden = dHidden;
    this.outputSize = outputSize;

    const scale1 = Math.sqrt(2 / (inputSize + dHidden));
    const scale2 = Math.sqrt(2 / (dHidden + outputSize));
    this.W1 = Matrix.random(inputSize, dHidden).mul(scale1);
    this.b1 = Matrix.zeros(1, dHidden);
    this.W2 = Matrix.random(dHidden, outputSize).mul(scale2);
    this.b2 = Matrix.zeros(1, outputSize);

    this._cache = null;
    this.dW1 = null;
    this.db1 = null;
    this.dW2 = null;
    this.db2 = null;
  }

  forward(input) {
    // input: [batch, inputSize]
    const hidden = input.dot(this.W1); // [batch, dHidden]
    // Add bias
    for (let r = 0; r < hidden.rows; r++)
      for (let c = 0; c < hidden.cols; c++)
        hidden.set(r, c, hidden.get(r, c) + this.b1.get(0, c));
    // ReLU
    const activated = new Matrix(hidden.rows, hidden.cols);
    for (let r = 0; r < hidden.rows; r++)
      for (let c = 0; c < hidden.cols; c++)
        activated.set(r, c, Math.max(0, hidden.get(r, c)));
    // Output
    const out = activated.dot(this.W2); // [batch, outputSize]
    for (let r = 0; r < out.rows; r++)
      for (let c = 0; c < out.cols; c++)
        out.set(r, c, out.get(r, c) + this.b2.get(0, c));

    this._cache = { input, hidden, activated };
    return out;
  }

  backward(dOutput) {
    // dOutput: [batch, outputSize]
    const { input, hidden, activated } = this._cache;
    const batch = dOutput.rows;

    // dW2, db2
    this.dW2 = activated.transpose().dot(dOutput); // [dHidden, outputSize]
    this.db2 = Matrix.zeros(1, this.outputSize);
    for (let r = 0; r < batch; r++)
      for (let c = 0; c < this.outputSize; c++)
        this.db2.set(0, c, this.db2.get(0, c) + dOutput.get(r, c));

    // dActivated
    const dActivated = dOutput.dot(this.W2.transpose()); // [batch, dHidden]

    // ReLU backward
    const dHidden = new Matrix(batch, this.dHidden);
    for (let r = 0; r < batch; r++)
      for (let c = 0; c < this.dHidden; c++)
        dHidden.set(r, c, hidden.get(r, c) > 0 ? dActivated.get(r, c) : 0);

    // dW1, db1
    this.dW1 = input.transpose().dot(dHidden); // [inputSize, dHidden]
    this.db1 = Matrix.zeros(1, this.dHidden);
    for (let r = 0; r < batch; r++)
      for (let c = 0; c < this.dHidden; c++)
        this.db1.set(0, c, this.db1.get(0, c) + dHidden.get(r, c));

    // dInput
    return dHidden.dot(this.W1.transpose()); // [batch, inputSize]
  }

  update(lr) {
    for (let r = 0; r < this.W1.rows; r++)
      for (let c = 0; c < this.W1.cols; c++)
        this.W1.set(r, c, this.W1.get(r, c) - lr * this.dW1.get(r, c));
    for (let c = 0; c < this.b1.cols; c++)
      this.b1.set(0, c, this.b1.get(0, c) - lr * this.db1.get(0, c));
    for (let r = 0; r < this.W2.rows; r++)
      for (let c = 0; c < this.W2.cols; c++)
        this.W2.set(r, c, this.W2.get(r, c) - lr * this.dW2.get(r, c));
    for (let c = 0; c < this.b2.cols; c++)
      this.b2.set(0, c, this.b2.get(0, c) - lr * this.db2.get(0, c));
  }

  paramCount() {
    return this.W1.rows * this.W1.cols + this.b1.cols
         + this.W2.rows * this.W2.cols + this.b2.cols;
  }
}

/**
 * MoE Layer: routes each token to top-K experts.
 *
 * Architecture:
 *   1. Router: linear projection from inputSize → numExperts (learned)
 *   2. Top-K selection: pick K experts with highest router scores
 *   3. Expert computation: run each selected expert's FFN
 *   4. Weighted sum: combine expert outputs using softmax(router scores)
 *
 * @param {number} inputSize - input dimension
 * @param {number} numExperts - total number of experts
 * @param {number} dHidden - hidden dimension per expert FFN
 * @param {number} outputSize - output dimension
 * @param {number} topK - number of experts to activate per token (default: 2)
 */
export class MixtureOfExperts {
  constructor(inputSize, numExperts, dHidden, outputSize, topK = 2) {
    if (topK > numExperts) throw new Error('topK must be <= numExperts');
    if (topK < 1) throw new Error('topK must be >= 1');

    this.inputSize = inputSize;
    this.numExperts = numExperts;
    this.dHidden = dHidden;
    this.outputSize = outputSize;
    this.topK = topK;

    // Router: inputSize → numExperts
    const scale = Math.sqrt(2 / (inputSize + numExperts));
    this.routerW = Matrix.random(inputSize, numExperts).mul(scale);
    this.routerB = Matrix.zeros(1, numExperts);

    // Expert FFNs
    this.experts = [];
    for (let i = 0; i < numExperts; i++) {
      this.experts.push(new ExpertFFN(inputSize, dHidden, outputSize));
    }

    // Stats tracking
    this.routingCounts = new Array(numExperts).fill(0);
    this.totalRouted = 0;

    // Cache for backward
    this._cache = null;
    this.gateProbs = null;
    this.topKIndices = null;

    // Router gradients
    this.dRouterW = null;
    this.dRouterB = null;

    // Compatibility: expose gate object for tests that use moe.gate.probs / moe.gate.weights
    this.gate = {
      weights: this.routerW,
      probs: null,
    };
  }

  /**
   * Forward pass: route each token to top-K experts.
   * @param {Matrix} input - [batch, inputSize]
   * @returns {Matrix} output - [batch, outputSize]
   */
  forward(input) {
    const N = input.rows;
    const output = new Matrix(N, this.outputSize);

    // Compute gate logits for all tokens
    this.gateProbs = new Matrix(N, this.numExperts);
    this.topKIndices = [];

    const expertWeights = []; // [token][k] → weight
    
    // Per-token per-expert cache for correct backward
    // _expertCaches[expertIdx] = [{input, hidden, activated}, ...] ordered by token appearance
    const expertCaches = [];
    for (let e = 0; e < this.numExperts; e++) expertCaches.push([]);
    // _expertTokenMap[expertIdx] = [tokenIdx, ...] tracks which tokens used this expert
    const expertTokenMap = [];
    for (let e = 0; e < this.numExperts; e++) expertTokenMap.push([]);
    
    // Store expert outputs per token for router gradient
    const expertOutputs = [];

    for (let t = 0; t < N; t++) {
      // Extract token vector
      const token = new Matrix(1, this.inputSize);
      for (let d = 0; d < this.inputSize; d++) token.set(0, d, input.get(t, d));

      // Router scores
      const logits = token.dot(this.routerW);
      for (let e = 0; e < this.numExperts; e++) {
        logits.set(0, e, logits.get(0, e) + this.routerB.get(0, e));
      }

      // Full softmax for gateProbs
      const allScores = [];
      for (let e = 0; e < this.numExperts; e++) allScores.push(logits.get(0, e));
      const fullProbs = softmaxArr(allScores);
      for (let e = 0; e < this.numExperts; e++) {
        this.gateProbs.set(t, e, fullProbs[e]);
      }

      // Top-K selection
      const indexed = fullProbs.map((p, i) => ({ idx: i, prob: p }));
      indexed.sort((a, b) => b.prob - a.prob);
      const topKSelected = indexed.slice(0, this.topK);
      this.topKIndices.push(topKSelected.map(e => e.idx));

      // Renormalize top-K weights
      const topSum = topKSelected.reduce((s, e) => s + e.prob, 0);
      const weights = topKSelected.map(e => e.prob / topSum);
      expertWeights.push(weights);

      // Compute weighted sum of expert outputs
      const tokenExpertOuts = {};
      for (let k = 0; k < this.topK; k++) {
        const expertIdx = topKSelected[k].idx;
        const weight = weights[k];
        this.routingCounts[expertIdx]++;
        this.totalRouted++;

        // Forward through expert and cache per-token
        const expert = this.experts[expertIdx];
        const expertOut = expert.forward(token);
        
        // Save the cache before it gets overwritten
        expertCaches[expertIdx].push({
          input: expert._cache.input,
          hidden: expert._cache.hidden,
          activated: expert._cache.activated,
        });
        expertTokenMap[expertIdx].push(t);
        
        tokenExpertOuts[expertIdx] = expertOut;

        for (let d = 0; d < this.outputSize; d++) {
          output.set(t, d, output.get(t, d) + weight * expertOut.get(0, d));
        }
      }
      expertOutputs.push(tokenExpertOuts);
    }

    this._cache = { input, expertOutputs, expertWeights, expertCaches, expertTokenMap };
    this.gate.probs = this.gateProbs;
    return output;
  }

  /**
   * Backward pass: compute gradients for router and experts.
   * @param {Matrix} dOutput - [batch, outputSize]
   * @returns {Matrix} dInput - [batch, inputSize]
   */
  backward(dOutput) {
    const { input, expertOutputs, expertWeights, expertCaches, expertTokenMap } = this._cache;
    const N = dOutput.rows;
    const dInput = new Matrix(N, this.inputSize);

    // Initialize router gradients
    this.dRouterW = Matrix.zeros(this.inputSize, this.numExperts);
    this.dRouterB = Matrix.zeros(1, this.numExperts);

    // Zero out expert gradients
    for (const expert of this.experts) {
      expert.dW1 = Matrix.zeros(expert.W1.rows, expert.W1.cols);
      expert.db1 = Matrix.zeros(1, expert.dHidden);
      expert.dW2 = Matrix.zeros(expert.W2.rows, expert.W2.cols);
      expert.db2 = Matrix.zeros(1, expert.outputSize);
    }

    for (let t = 0; t < N; t++) {
      const token = new Matrix(1, this.inputSize);
      for (let d = 0; d < this.inputSize; d++) token.set(0, d, input.get(t, d));

      const dOut = new Matrix(1, this.outputSize);
      for (let d = 0; d < this.outputSize; d++) dOut.set(0, d, dOutput.get(t, d));

      const topK = this.topKIndices[t];
      const weights = expertWeights[t];

      // Compute per-expert contribution for router gradient
      const dLdWeights = [];
      for (let k = 0; k < this.topK; k++) {
        const expertIdx = topK[k];
        const expertOut = expertOutputs[t][expertIdx];
        let dot = 0;
        for (let d = 0; d < this.outputSize; d++) {
          dot += dOut.get(0, d) * expertOut.get(0, d);
        }
        dLdWeights.push(dot);
      }

      // Softmax Jacobian for router gradient
      const dLdLogits = new Array(this.numExperts).fill(0);
      for (let j = 0; j < this.topK; j++) {
        for (let i = 0; i < this.topK; i++) {
          const dwi = dLdWeights[i];
          const wi = weights[i];
          const wj = weights[j];
          const delta = (i === j) ? 1 : 0;
          dLdLogits[topK[j]] += dwi * wi * (delta - wj);
        }
      }

      // Accumulate router gradients
      for (let d = 0; d < this.inputSize; d++) {
        for (let e = 0; e < this.numExperts; e++) {
          this.dRouterW.set(d, e, this.dRouterW.get(d, e) + token.get(0, d) * dLdLogits[e]);
        }
      }
      for (let e = 0; e < this.numExperts; e++) {
        this.dRouterB.set(0, e, this.dRouterB.get(0, e) + dLdLogits[e]);
      }

      // Router contribution to dInput: dL/dinput += dLdLogits · routerW^T
      for (let d = 0; d < this.inputSize; d++) {
        let routerGrad = 0;
        for (let e = 0; e < this.numExperts; e++) {
          routerGrad += dLdLogits[e] * this.routerW.get(d, e);
        }
        dInput.set(t, d, dInput.get(t, d) + routerGrad);
      }

      // Expert backward passes with correct per-token cache
      for (let k = 0; k < this.topK; k++) {
        const expertIdx = topK[k];
        const weight = weights[k];
        const expert = this.experts[expertIdx];

        // Find this token's cache for this expert
        const tokenList = expertTokenMap[expertIdx];
        const cacheList = expertCaches[expertIdx];
        const cacheIdx = tokenList.indexOf(t);
        
        // Restore expert's cache
        expert._cache = cacheList[cacheIdx];

        const dExpertOut = new Matrix(1, this.outputSize);
        for (let d = 0; d < this.outputSize; d++) {
          dExpertOut.set(0, d, weight * dOut.get(0, d));
        }

        // Compute gradients (but accumulate rather than replace)
        const { input: cachedInput, hidden, activated } = expert._cache;
        const batch = 1;

        // dW2, db2
        const dW2 = activated.transpose().dot(dExpertOut);
        for (let r = 0; r < expert.W2.rows; r++)
          for (let c = 0; c < expert.W2.cols; c++)
            expert.dW2.set(r, c, expert.dW2.get(r, c) + dW2.get(r, c));
        for (let c = 0; c < expert.outputSize; c++)
          expert.db2.set(0, c, expert.db2.get(0, c) + dExpertOut.get(0, c));

        // dActivated
        const dActivated = dExpertOut.dot(expert.W2.transpose());

        // ReLU backward
        const dHidden = new Matrix(1, expert.dHidden);
        for (let c = 0; c < expert.dHidden; c++)
          dHidden.set(0, c, hidden.get(0, c) > 0 ? dActivated.get(0, c) : 0);

        // dW1, db1
        const dW1 = cachedInput.transpose().dot(dHidden);
        for (let r = 0; r < expert.W1.rows; r++)
          for (let c = 0; c < expert.W1.cols; c++)
            expert.dW1.set(r, c, expert.dW1.get(r, c) + dW1.get(r, c));
        for (let c = 0; c < expert.dHidden; c++)
          expert.db1.set(0, c, expert.db1.get(0, c) + dHidden.get(0, c));

        // dInput
        const dExpertInput = dHidden.dot(expert.W1.transpose());
        for (let d = 0; d < this.inputSize; d++) {
          dInput.set(t, d, dInput.get(t, d) + dExpertInput.get(0, d));
        }
      }
    }

    // Update gate reference
    this.gate.weights = this.routerW;
    return dInput;
  }

  /**
   * Update weights (SGD).
   * @param {number} lr - learning rate
   */
  update(lr) {
    // Update experts
    for (const expert of this.experts) {
      if (expert.dW1) expert.update(lr);
    }
    // Update router
    if (this.dRouterW) {
      for (let r = 0; r < this.routerW.rows; r++)
        for (let c = 0; c < this.routerW.cols; c++)
          this.routerW.set(r, c, this.routerW.get(r, c) - lr * this.dRouterW.get(r, c));
      for (let c = 0; c < this.routerB.cols; c++)
        this.routerB.set(0, c, this.routerB.get(0, c) - lr * this.dRouterB.get(0, c));
    }
  }

  /**
   * Reset routing statistics.
   */
  resetRoutingStats() {
    this.routingCounts = new Array(this.numExperts).fill(0);
    this.totalRouted = 0;
  }

  /**
   * Get routing distribution (fraction per expert).
   */
  routingDistribution() {
    if (this.totalRouted === 0) return new Array(this.numExperts).fill(0);
    return this.routingCounts.map(c => c / this.totalRouted);
  }

  /**
   * Get routing statistics: how often each expert was selected.
   */
  routingStats() {
    return this.routingCounts.map((count, idx) => ({
      expert: idx,
      count,
      pct: this.totalRouted > 0 ? (count / this.totalRouted * 100).toFixed(1) + '%' : '0%',
    }));
  }

  /**
   * Load balancing loss (auxiliary loss for training).
   * Penalizes uneven expert utilization.
   * Uses coefficient-of-variation style: 0 for perfect balance, positive for imbalance.
   */
  loadBalanceLoss() {
    if (this.totalRouted === 0) return 0;
    const fractions = this.routingCounts.map(c => c / this.totalRouted);
    const ideal = 1 / this.numExperts;
    let loss = 0;
    for (const f of fractions) {
      loss += (f - ideal) * (f - ideal);
    }
    return loss * this.numExperts;
  }

  /**
   * Parameter count: router + all experts.
   */
  paramCount() {
    let count = this.routerW.rows * this.routerW.cols + this.routerB.cols; // router W + b
    for (const expert of this.experts) {
      count += expert.paramCount();
    }
    return count;
  }

  /**
   * Active parameters per token (only topK experts).
   */
  activeParamsPerToken() {
    const routerParams = this.routerW.rows * this.routerW.cols + this.routerB.cols;
    const expertParams = this.experts[0].paramCount();
    return routerParams + this.topK * expertParams;
  }
}
