// moe.js — Mixture of Experts Layer (Shazeer et al., 2017)
// Routes each token to top-k experts out of N total.
// Scales model capacity without proportionally scaling compute.
//
// Architecture:
//   Router: linear layer mapping input → expert scores
//   Experts: N independent FFN layers
//   Output: weighted sum of top-k expert outputs
//
// Load balancing: auxiliary loss encourages equal expert usage

import { Matrix } from './matrix.js';
import { Dense } from './layer.js';

/**
 * Top-k routing function.
 * @param {Float64Array} scores - Expert scores for one token
 * @param {number} k - Number of experts to select
 * @returns {{ indices: number[], weights: Float64Array }}
 */
function topKRoute(scores, k) {
  const n = scores.length;
  // Find top-k indices
  const indexed = Array.from(scores).map((s, i) => ({ s, i }));
  indexed.sort((a, b) => b.s - a.s);
  const topK = indexed.slice(0, k);
  
  // Softmax over top-k scores
  const maxScore = Math.max(...topK.map(t => t.s));
  const expScores = topK.map(t => Math.exp(t.s - maxScore));
  const sumExp = expScores.reduce((a, b) => a + b, 0);
  const weights = new Float64Array(k);
  for (let i = 0; i < k; i++) {
    weights[i] = expScores[i] / sumExp;
  }
  
  return { indices: topK.map(t => t.i), weights };
}

export class MixtureOfExperts {
  /**
   * @param {number} inputDim - Input/output dimension
   * @param {number} hiddenDim - Expert FFN hidden dimension
   * @param {number} numExperts - Total number of experts
   * @param {number} topK - Number of experts per token
   * @param {number} loadBalanceCoeff - Load balancing loss coefficient
   */
  constructor(inputDim, hiddenDim, numExperts = 8, topK = 2, loadBalanceCoeff = 0.01) {
    this.inputDim = inputDim;
    this.hiddenDim = hiddenDim;
    this.numExperts = numExperts;
    this.topK = topK;
    this.loadBalanceCoeff = loadBalanceCoeff;
    
    // Router: maps input to expert scores
    this.router = new Dense(inputDim, numExperts, 'linear');
    
    // Experts: N independent FFN layers (input → hidden → input)
    this.experts = [];
    for (let i = 0; i < numExperts; i++) {
      this.experts.push({
        up: new Dense(inputDim, hiddenDim, 'relu'),
        down: new Dense(hiddenDim, inputDim, 'linear'),
      });
    }
    
    // Tracking for load balancing
    this._expertCounts = new Float64Array(numExperts);
    this._totalTokens = 0;
    this._routingInfo = null;
  }

  /**
   * Forward pass: route each token through top-k experts.
   * @param {Matrix} x - Input (seqLen × inputDim)
   * @returns {{ output: Matrix, auxLoss: number }}
   */
  forward(x) {
    const seqLen = x.rows;
    const d = x.cols;
    const output = new Matrix(seqLen, d);
    
    // Compute router scores
    const routerScores = this.router.forward(x); // seqLen × numExperts
    
    // Save routing info for backward
    this._routingInfo = {
      input: x,
      routerScores,
      tokenRouting: [], // per-token: { indices, weights }
    };
    
    // Reset expert counts
    this._expertCounts.fill(0);
    this._totalTokens = seqLen;
    
    for (let t = 0; t < seqLen; t++) {
      // Get scores for this token
      const scores = new Float64Array(this.numExperts);
      for (let e = 0; e < this.numExperts; e++) {
        scores[e] = routerScores.get(t, e);
      }
      
      // Route to top-k experts
      const { indices, weights } = topKRoute(scores, this.topK);
      this._routingInfo.tokenRouting.push({ indices, weights });
      
      // Track expert usage
      for (const idx of indices) this._expertCounts[idx]++;
      
      // Get token input
      const tokenInput = new Matrix(1, d);
      for (let j = 0; j < d; j++) tokenInput.set(0, j, x.get(t, j));
      
      // Compute weighted expert outputs
      for (let k = 0; k < this.topK; k++) {
        const expertIdx = indices[k];
        const weight = weights[k];
        const expert = this.experts[expertIdx];
        
        const hidden = expert.up.forward(tokenInput);
        const expertOut = expert.down.forward(hidden);
        
        for (let j = 0; j < d; j++) {
          output.set(t, j, output.get(t, j) + weight * expertOut.get(0, j));
        }
      }
    }
    
    // Compute load balancing auxiliary loss
    const auxLoss = this._computeLoadBalanceLoss();
    
    return { output, auxLoss };
  }

  /**
   * Compute auxiliary load balancing loss.
   * Encourages equal expert usage across tokens.
   * Loss = coeff * N * sum(f_i * P_i) where f_i = fraction of tokens routed to expert i
   * and P_i = average router probability for expert i.
   */
  _computeLoadBalanceLoss() {
    if (this._totalTokens === 0) return 0;
    
    let loss = 0;
    for (let e = 0; e < this.numExperts; e++) {
      const f = this._expertCounts[e] / this._totalTokens;
      // P is the average softmax probability for expert e across all tokens
      let pSum = 0;
      for (let t = 0; t < this._totalTokens; t++) {
        pSum += Math.exp(this._routingInfo.routerScores.get(t, e));
      }
      const p = pSum / this._totalTokens;
      loss += f * p;
    }
    
    return this.loadBalanceCoeff * this.numExperts * loss;
  }

  /**
   * Backward pass (simplified: only updates expert FFNs, not router).
   * Full backward with router gradients requires more complex implementation.
   */
  backward(dOutput) {
    // For each token, propagate gradient through the selected experts
    const seqLen = dOutput.rows;
    const d = dOutput.cols;
    
    for (let t = 0; t < seqLen; t++) {
      const { indices, weights } = this._routingInfo.tokenRouting[t];
      const dToken = new Matrix(1, d);
      for (let j = 0; j < d; j++) dToken.set(0, j, dOutput.get(t, j));
      
      for (let k = 0; k < this.topK; k++) {
        const expertIdx = indices[k];
        const weight = weights[k];
        const expert = this.experts[expertIdx];
        
        // Scale gradient by routing weight
        const scaledDToken = new Matrix(1, d);
        for (let j = 0; j < d; j++) scaledDToken.set(0, j, dToken.get(0, j) * weight);
        
        // Backward through expert
        const dHidden = expert.down.backward(scaledDToken);
        expert.up.backward(dHidden);
      }
    }
    
    // Router backward (gradient of scores w.r.t. routing decisions)
    // Simplified: we don't compute router gradients in this version
    return null;
  }

  update(lr) {
    this.router.update(lr);
    for (const expert of this.experts) {
      expert.up.update(lr);
      expert.down.update(lr);
    }
  }

  paramCount() {
    let count = this.router.paramCount();
    for (const expert of this.experts) {
      count += expert.up.paramCount() + expert.down.paramCount();
    }
    return count;
  }

  /**
   * Get expert utilization statistics.
   */
  getExpertStats() {
    const total = this._totalTokens || 1;
    return Array.from(this._expertCounts).map((count, i) => ({
      expert: i,
      tokens: count,
      fraction: count / total,
      balanced: Math.abs(count / total - 1 / this.numExperts) < 0.1,
    }));
  }
}
