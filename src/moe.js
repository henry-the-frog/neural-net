// moe.js — Mixture of Experts layer
// Sparse gating with top-k expert routing, load balancing, and capacity factors

import { Matrix } from './matrix.js';
import { Dense } from './layer.js';

// ===== Gating Network =====
// Produces routing probabilities for each expert
class GatingNetwork {
  constructor(inputSize, numExperts) {
    this.inputSize = inputSize;
    this.numExperts = numExperts;
    // Simple linear gate: input → expert logits
    this.weights = Matrix.random(inputSize, numExperts);
    this.biases = Matrix.zeros(1, numExperts);
    this.input = null;
    this.logits = null;
    this.probs = null;
    this.dWeights = null;
    this.dBiases = null;
  }

  forward(input) {
    this.input = input;
    // Compute logits
    this.logits = input.dot(this.weights);
    for (let i = 0; i < this.logits.rows; i++) {
      for (let j = 0; j < this.logits.cols; j++) {
        this.logits.set(i, j, this.logits.get(i, j) + this.biases.get(0, j));
      }
    }
    // Softmax over experts for each sample
    this.probs = softmax(this.logits);
    return this.probs;
  }

  backward(dProbs) {
    // Backward through softmax: dLogits = dProbs * softmax_jacobian
    const dLogits = softmaxBackward(dProbs, this.probs);
    // Gradient for weights and biases
    this.dWeights = this.input.T().dot(dLogits);
    this.dBiases = columnSum(dLogits);
    // Gradient for input
    return dLogits.dot(this.weights.T());
  }

  update(learningRate) {
    if (!this.dWeights) return;
    this.weights = this.weights.sub(this.dWeights.mul(learningRate));
    this.biases = this.biases.sub(this.dBiases.mul(learningRate));
  }

  paramCount() {
    return this.inputSize * this.numExperts + this.numExperts;
  }
}

// ===== Expert Network =====
// Each expert is a small feed-forward network
class Expert {
  constructor(inputSize, hiddenSize, outputSize) {
    this.fc1 = new Dense(inputSize, hiddenSize, 'relu');
    this.fc2 = new Dense(hiddenSize, outputSize, 'linear');
  }

  forward(input) {
    return this.fc2.forward(this.fc1.forward(input));
  }

  backward(dOutput) {
    return this.fc1.backward(this.fc2.backward(dOutput));
  }

  update(learningRate) {
    // Only update if backward was called (gradients exist)
    if (this.fc1.dWeights) this.fc1.update(learningRate, 0, 'sgd');
    if (this.fc2.dWeights) this.fc2.update(learningRate, 0, 'sgd');
  }

  paramCount() {
    return this.fc1.paramCount() + this.fc2.paramCount();
  }
}

// ===== Mixture of Experts Layer =====
export class MixtureOfExperts {
  constructor(inputSize, numExperts, expertHiddenSize, outputSize, topK = 2) {
    this.inputSize = inputSize;
    this.numExperts = numExperts;
    this.outputSize = outputSize;
    this.topK = Math.min(topK, numExperts);

    this.gate = new GatingNetwork(inputSize, numExperts);
    this.experts = Array.from({ length: numExperts }, () =>
      new Expert(inputSize, expertHiddenSize, outputSize)
    );

    // Track routing for load balancing
    this.routingCounts = new Array(numExperts).fill(0);
    this.totalRouted = 0;
    
    // Cache for backward
    this.input = null;
    this.gateProbs = null;
    this.topKIndices = null; // [batchSize][topK]
    this.topKWeights = null; // [batchSize][topK] — normalized weights
    this.expertOutputs = null; // [batchSize][topK] — Matrix outputs
  }

  forward(input) {
    this.input = input;
    const batchSize = input.rows;

    // Get gating probabilities
    this.gateProbs = this.gate.forward(input);

    // Top-K selection per sample
    this.topKIndices = [];
    this.topKWeights = [];
    this.expertOutputs = [];

    const output = Matrix.zeros(batchSize, this.outputSize);

    for (let b = 0; b < batchSize; b++) {
      // Get probabilities for this sample
      const probs = [];
      for (let e = 0; e < this.numExperts; e++) {
        probs.push({ idx: e, prob: this.gateProbs.get(b, e) });
      }
      probs.sort((a, b_) => b_.prob - a.prob);

      const indices = probs.slice(0, this.topK).map(p => p.idx);
      const rawWeights = probs.slice(0, this.topK).map(p => p.prob);

      // Renormalize top-K weights
      const weightSum = rawWeights.reduce((s, w) => s + w, 0);
      const weights = rawWeights.map(w => weightSum > 0 ? w / weightSum : 1 / this.topK);

      this.topKIndices.push(indices);
      this.topKWeights.push(weights);

      // Extract single sample
      const sample = extractRow(input, b);

      // Run selected experts and combine
      const expertOuts = [];
      for (let k = 0; k < this.topK; k++) {
        const expertIdx = indices[k];
        const expertOut = this.experts[expertIdx].forward(sample);
        expertOuts.push(expertOut);

        // Weighted sum into output
        for (let d = 0; d < this.outputSize; d++) {
          output.set(b, d, output.get(b, d) + weights[k] * expertOut.get(0, d));
        }

        // Track routing
        this.routingCounts[expertIdx]++;
        this.totalRouted++;
      }
      this.expertOutputs.push(expertOuts);
    }

    return output;
  }

  backward(dOutput) {
    const batchSize = dOutput.rows;
    const dInput = Matrix.zeros(batchSize, this.inputSize);
    const dGateProbs = Matrix.zeros(batchSize, this.numExperts);

    for (let b = 0; b < batchSize; b++) {
      const indices = this.topKIndices[b];
      const weights = this.topKWeights[b];

      for (let k = 0; k < this.topK; k++) {
        const expertIdx = indices[k];
        const weight = weights[k];

        // Gradient for this expert's output
        const dExpertOut = new Matrix(1, this.outputSize);
        for (let d = 0; d < this.outputSize; d++) {
          dExpertOut.set(0, d, dOutput.get(b, d) * weight);
        }

        // Backward through expert
        // Must re-run forward to set correct internal caches for this sample
        // (expert may have been called with a different sample later in the batch)
        const sample = extractRow(this.input, b);
        this.experts[expertIdx].forward(sample);
        const dExpertInput = this.experts[expertIdx].backward(dExpertOut);

        // Accumulate input gradient
        for (let d = 0; d < this.inputSize; d++) {
          dInput.set(b, d, dInput.get(b, d) + dExpertInput.get(0, d));
        }

        // Gradient for gate probabilities
        let gateGrad = 0;
        for (let d = 0; d < this.outputSize; d++) {
          gateGrad += dOutput.get(b, d) * this.expertOutputs[b][k].get(0, d);
        }
        dGateProbs.set(b, expertIdx, dGateProbs.get(b, expertIdx) + gateGrad);
      }
    }

    // Backward through gating network
    const dGateInput = this.gate.backward(dGateProbs);

    // Combine gradients
    for (let b = 0; b < batchSize; b++) {
      for (let d = 0; d < this.inputSize; d++) {
        dInput.set(b, d, dInput.get(b, d) + dGateInput.get(b, d));
      }
    }

    return dInput;
  }

  update(learningRate) {
    this.gate.update(learningRate);
    for (const expert of this.experts) expert.update(learningRate);
  }

  // Load balancing loss (auxiliary loss to encourage even routing)
  loadBalanceLoss() {
    if (this.totalRouted === 0) return 0;
    const fractions = this.routingCounts.map(c => c / this.totalRouted);
    const target = 1 / this.numExperts;
    // Coefficient of variation — penalizes uneven distribution
    const variance = fractions.reduce((s, f) => s + (f - target) ** 2, 0) / this.numExperts;
    return this.numExperts * variance; // Scale so perfectly balanced = 0
  }

  resetRoutingStats() {
    this.routingCounts.fill(0);
    this.totalRouted = 0;
  }

  routingDistribution() {
    if (this.totalRouted === 0) return this.routingCounts.map(() => 0);
    return this.routingCounts.map(c => c / this.totalRouted);
  }

  paramCount() {
    return this.gate.paramCount() +
      this.experts.reduce((s, e) => s + e.paramCount(), 0);
  }
}

// ===== Helper Functions =====

function softmax(logits) {
  const result = new Matrix(logits.rows, logits.cols);
  for (let i = 0; i < logits.rows; i++) {
    let max = -Infinity;
    for (let j = 0; j < logits.cols; j++) {
      if (logits.get(i, j) > max) max = logits.get(i, j);
    }
    let sum = 0;
    for (let j = 0; j < logits.cols; j++) {
      const e = Math.exp(logits.get(i, j) - max);
      result.set(i, j, e);
      sum += e;
    }
    for (let j = 0; j < logits.cols; j++) {
      result.set(i, j, result.get(i, j) / sum);
    }
  }
  return result;
}

function softmaxBackward(dProbs, probs) {
  const result = new Matrix(dProbs.rows, dProbs.cols);
  for (let i = 0; i < dProbs.rows; i++) {
    for (let j = 0; j < dProbs.cols; j++) {
      let sum = 0;
      for (let k = 0; k < dProbs.cols; k++) {
        const indicator = j === k ? 1 : 0;
        sum += dProbs.get(i, k) * probs.get(i, k) * (indicator - probs.get(i, j));
      }
      result.set(i, j, sum);
    }
  }
  return result;
}

function columnSum(m) {
  const result = new Matrix(1, m.cols);
  for (let j = 0; j < m.cols; j++) {
    let sum = 0;
    for (let i = 0; i < m.rows; i++) sum += m.get(i, j);
    result.set(0, j, sum);
  }
  return result;
}

function extractRow(m, row) {
  const result = new Matrix(1, m.cols);
  for (let j = 0; j < m.cols; j++) result.set(0, j, m.get(row, j));
  return result;
}
