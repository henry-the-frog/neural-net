// moe.js — Mixture of Experts Layer (Shazeer et al., 2017)
// Routes each token to top-k experts out of N total.
// Scales model capacity without proportionally scaling compute.

import { Matrix } from './matrix.js';
import { Dense } from './layer.js';

/**
 * Top-k routing function.
 * @param {Float64Array} scores - Expert scores for one token
 * @param {number} k - Number of experts to select
 * @returns {{ indices: number[], weights: Float64Array }}
 */
function topKRoute(scores, k) {
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

/**
 * Full softmax over all scores.
 */
function softmax(scores) {
  const max = Math.max(...scores);
  const exps = scores.map(s => Math.exp(s - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

export class MixtureOfExperts {
  /**
   * @param {number} inputDim - Input dimension
   * @param {number} numExperts - Number of expert networks
   * @param {number} hiddenDim - Expert FFN hidden dimension
   * @param {number} outputDim - Output dimension
   * @param {number} topK - Number of experts per token (default 2)
   * @param {number} loadBalanceCoeff - Load balancing loss coefficient
   */
  constructor(inputDim, numExperts = 4, hiddenDim = 16, outputDim = null, topK = 2, loadBalanceCoeff = 0.01) {
    this.inputDim = inputDim;
    this.numExperts = numExperts;
    this.hiddenDim = hiddenDim;
    this.outputDim = outputDim || inputDim;
    this.topK = topK;
    this.loadBalanceCoeff = loadBalanceCoeff;
    
    // Gate/Router: maps input to expert scores (softmax over numExperts)
    this.gate = new Dense(inputDim, numExperts, 'linear');
    
    // Experts: N independent FFN layers (input → hidden → output)
    this.experts = [];
    for (let i = 0; i < numExperts; i++) {
      this.experts.push({
        up: new Dense(inputDim, hiddenDim, 'relu'),
        down: new Dense(hiddenDim, this.outputDim, 'linear'),
      });
    }
    
    // Tracking for load balancing
    this.routingCounts = new Array(numExperts).fill(0);
    this.totalRouted = 0;
    this.topKIndices = []; // per-sample top-K indices
    this.gateProbs = null; // (batch, numExperts) gate probabilities
    
    // Saved for backward
    this._input = null;
    this._tokenRouting = null;
  }

  /**
   * Reset routing statistics.
   */
  resetRoutingStats() {
    this.routingCounts = new Array(this.numExperts).fill(0);
    this.totalRouted = 0;
    this.topKIndices = [];
  }

  /**
   * Get routing distribution (fraction of tokens per expert).
   */
  routingDistribution() {
    if (this.totalRouted === 0) return new Array(this.numExperts).fill(0);
    return this.routingCounts.map(c => c / this.totalRouted);
  }

  /**
   * Compute load balance loss from current routing counts.
   * Perfectly balanced → 0. Imbalanced → positive.
   */
  loadBalanceLoss() {
    if (this.totalRouted === 0) return 0;
    const n = this.routingCounts.length;
    const expected = this.totalRouted / n;
    let loss = 0;
    for (let i = 0; i < n; i++) {
      const diff = this.routingCounts[i] - expected;
      loss += diff * diff;
    }
    return loss / (this.totalRouted * this.totalRouted);
  }

  /**
   * Forward pass: route each token through top-k experts.
   * @param {Matrix} x - Input (batchSize × inputDim)
   * @returns {Matrix} output (batchSize × outputDim)
   */
  forward(x) {
    const batchSize = x.rows;
    const output = new Matrix(batchSize, this.outputDim);
    
    // Compute gate scores
    const gateScores = this.gate.forward(x); // batchSize × numExperts
    
    // Compute full softmax gate probabilities
    this.gateProbs = new Matrix(batchSize, this.numExperts);
    for (let b = 0; b < batchSize; b++) {
      const scores = new Array(this.numExperts);
      for (let e = 0; e < this.numExperts; e++) {
        scores[e] = gateScores.get(b, e);
      }
      const probs = softmax(scores);
      for (let e = 0; e < this.numExperts; e++) {
        this.gateProbs.set(b, e, probs[e]);
      }
    }
    // Also store on gate object for test compatibility
    this.gate.probs = this.gateProbs;
    
    // Save for backward
    this._input = x;
    this._tokenRouting = [];
    this._expertHiddens = []; // per-sample per-expert hidden states
    this._expertInputs = []; // per-sample input matrices
    this.topKIndices = [];
    
    for (let b = 0; b < batchSize; b++) {
      // Get scores for this sample
      const scores = new Float64Array(this.numExperts);
      for (let e = 0; e < this.numExperts; e++) {
        scores[e] = gateScores.get(b, e);
      }
      
      // Route to top-k experts
      const { indices, weights } = topKRoute(scores, this.topK);
      this._tokenRouting.push({ indices, weights });
      this.topKIndices.push(indices);
      
      // Track expert usage
      for (const idx of indices) {
        this.routingCounts[idx]++;
      }
      this.totalRouted += this.topK;
      
      // Get sample input
      const sampleInput = new Matrix(1, this.inputDim);
      for (let j = 0; j < this.inputDim; j++) sampleInput.set(0, j, x.get(b, j));
      this._expertInputs.push(sampleInput);
      
      // Compute weighted expert outputs
      const hiddens = {};
      for (let k = 0; k < this.topK; k++) {
        const expertIdx = indices[k];
        const weight = weights[k];
        const expert = this.experts[expertIdx];
        
        const hidden = expert.up.forward(sampleInput);
        const expertOut = expert.down.forward(hidden);
        hiddens[expertIdx] = { hidden: new Matrix(hidden.rows, hidden.cols, new Float64Array(hidden.data)), sampleInput };
        
        for (let j = 0; j < this.outputDim; j++) {
          output.set(b, j, output.get(b, j) + weight * expertOut.get(0, j));
        }
      }
      this._expertHiddens.push(hiddens);
    }
    
    return output;
  }

  /**
   * Backward pass: propagate gradient through selected experts.
   * @param {Matrix} dOutput - (batchSize × outputDim)
   * @returns {Matrix} dInput - (batchSize × inputDim)
   */
  backward(dOutput) {
    const batchSize = dOutput.rows;
    const dInput = new Matrix(batchSize, this.inputDim);
    
    // Zero-initialize gradient accumulators for all experts
    const gradAccum = this.experts.map(expert => ({
      up_dW: Matrix.zeros(expert.up.weights.rows, expert.up.weights.cols),
      up_dB: Matrix.zeros(1, expert.up.weights.cols),
      down_dW: Matrix.zeros(expert.down.weights.rows, expert.down.weights.cols),
      down_dB: Matrix.zeros(1, expert.down.weights.cols),
    }));
    
    // Gate gradient: compute proper softmax Jacobian
    const dGateScores = new Matrix(batchSize, this.numExperts);
    
    for (let b = 0; b < batchSize; b++) {
      const { indices, weights } = this._tokenRouting[b];
      const sampleInput = this._expertInputs[b];
      
      // Compute expert outputs for this sample
      const expertOuts = {};
      for (let k = 0; k < this.topK; k++) {
        const expertIdx = indices[k];
        const expert = this.experts[expertIdx];
        const hidden = expert.up.forward(sampleInput);
        const expertOut = expert.down.forward(hidden);
        expertOuts[expertIdx] = new Float64Array(expertOut.data);
      }
      
      // For each selected expert pair (k, e), compute dL/ds_e contribution
      // dL/ds_e = Σ_j dL/dy_j * Σ_k expertOut_k_j * w_k * (δ_{ke} - w_e)
      for (let ei = 0; ei < this.topK; ei++) {
        const e = indices[ei];
        const w_e = weights[ei];
        
        // Σ_j dL/dy_j * Σ_k expertOut_k_j * w_k * (δ_{ke} - w_e)
        let dScore = 0;
        for (let ki = 0; ki < this.topK; ki++) {
          const k = indices[ki];
          const w_k = weights[ki];
          const jacobian = w_k * ((k === e ? 1 : 0) - w_e);
          for (let j = 0; j < this.outputDim; j++) {
            dScore += dOutput.get(b, j) * expertOuts[k][j] * jacobian;
          }
        }
        dGateScores.set(b, e, dScore);
      }
      
      // Now backward through experts with proper state
      for (let k = 0; k < this.topK; k++) {
        const expertIdx = indices[k];
        const weight = weights[k];
        const expert = this.experts[expertIdx];
        
        // Re-forward to set correct internal state for this sample
        expert.up.forward(sampleInput);
        expert.down.forward(expert.up.a);
        
        // Scale gradient by routing weight
        const dToken = new Matrix(1, this.outputDim);
        for (let j = 0; j < this.outputDim; j++) {
          dToken.set(0, j, dOutput.get(b, j) * weight);
        }
        
        // Backward through expert
        const dHidden = expert.down.backward(dToken);
        const dExpertInput = expert.up.backward(dHidden);
        
        // Accumulate weight gradients
        const acc = gradAccum[expertIdx];
        for (let i = 0; i < acc.up_dW.data.length; i++) acc.up_dW.data[i] += expert.up.dWeights.data[i];
        for (let i = 0; i < acc.up_dB.data.length; i++) acc.up_dB.data[i] += expert.up.dBiases.data[i];
        for (let i = 0; i < acc.down_dW.data.length; i++) acc.down_dW.data[i] += expert.down.dWeights.data[i];
        for (let i = 0; i < acc.down_dB.data.length; i++) acc.down_dB.data[i] += expert.down.dBiases.data[i];
        
        // Accumulate input gradients
        if (dExpertInput) {
          for (let j = 0; j < this.inputDim; j++) {
            dInput.set(b, j, dInput.get(b, j) + dExpertInput.get(0, j));
          }
        }
      }
    }
    
    // Set accumulated gradients back on expert layers
    for (let e = 0; e < this.numExperts; e++) {
      const expert = this.experts[e];
      const acc = gradAccum[e];
      expert.up.dWeights = acc.up_dW;
      expert.up.dBiases = acc.up_dB;
      expert.up.input = new Matrix(batchSize, this.inputDim);
      expert.down.dWeights = acc.down_dW;
      expert.down.dBiases = acc.down_dB;
      expert.down.input = new Matrix(batchSize, this.hiddenDim);
    }
    
    // Backward through gate
    this.gate.forward(this._input); // Set gate internal state
    const dGateInput = this.gate.backward(dGateScores);
    
    // Add gate input gradient to total
    if (dGateInput) {
      for (let i = 0; i < dInput.data.length; i++) {
        dInput.data[i] += dGateInput.data[i];
      }
    }
    
    return dInput;
  }

  update(lr) {
    this.gate.update(lr);
    for (const expert of this.experts) {
      expert.up.update(lr);
      expert.down.update(lr);
    }
  }

  paramCount() {
    let count = this.gate.paramCount();
    for (const expert of this.experts) {
      count += expert.up.paramCount() + expert.down.paramCount();
    }
    return count;
  }

  toJSON() {
    const serializeDense = (layer) => ({
      weights: { data: Array.from(layer.weights.data), shape: [layer.weights.rows, layer.weights.cols] },
      biases: { data: Array.from(layer.biases.data), shape: [layer.biases.rows, layer.biases.cols] },
      activation: layer.activation?.name || 'linear',
    });
    return {
      type: 'MixtureOfExperts',
      inputDim: this.inputDim,
      numExperts: this.numExperts,
      hiddenDim: this.hiddenDim,
      outputDim: this.outputDim,
      topK: this.topK,
      loadBalanceCoeff: this.loadBalanceCoeff,
      gate: serializeDense(this.gate),
      experts: this.experts.map(e => ({ up: serializeDense(e.up), down: serializeDense(e.down) })),
    };
  }

  static fromJSON(json) {
    const moe = new MixtureOfExperts(
      json.inputDim, json.numExperts, json.hiddenDim, json.outputDim, json.topK, json.loadBalanceCoeff
    );
    const deserializeDense = (layer, data) => {
      if (data.weights) {
        layer.weights = new Matrix(data.weights.shape[0], data.weights.shape[1], new Float64Array(data.weights.data));
      }
      if (data.biases) {
        layer.biases = new Matrix(data.biases.shape[0], data.biases.shape[1], new Float64Array(data.biases.data));
      }
    };
    if (json.gate) deserializeDense(moe.gate, json.gate);
    if (json.experts) {
      for (let i = 0; i < Math.min(json.experts.length, moe.experts.length); i++) {
        if (json.experts[i].up) deserializeDense(moe.experts[i].up, json.experts[i].up);
        if (json.experts[i].down) deserializeDense(moe.experts[i].down, json.experts[i].down);
      }
    }
    return moe;
  }

  /**
   * Get expert utilization statistics.
   */
  getExpertStats() {
    const total = this.totalRouted || 1;
    return this.routingCounts.map((count, i) => ({
      expert: i,
      tokens: count,
      fraction: count / total,
    }));
  }
}
