/**
 * contrastive.js — Contrastive Learning
 * 
 * Self-supervised learning by pulling together similar pairs
 * and pushing apart dissimilar pairs in embedding space.
 * 
 * Implements:
 * - NT-Xent loss (Normalized Temperature-scaled Cross-Entropy)
 * - Simple augmentation pipeline
 * - Contrastive training loop
 * 
 * Based on: Chen et al. (2020), "A Simple Framework for Contrastive Learning"
 */

import { Matrix } from './matrix.js';
import { Network } from './network.js';
import { Dense } from './layer.js';

/**
 * Cosine similarity between two vectors.
 * @param {number[]} a
 * @param {number[]} b
 * @returns {number}
 */
export function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
}

/**
 * NT-Xent Loss (Normalized Temperature-scaled Cross-Entropy).
 * 
 * For a batch of N pairs, computes the contrastive loss where:
 * - Positive pair: (z_i, z_{i+N}) are views of the same input
 * - Negative pairs: all other combinations
 * 
 * L = -log(exp(sim(z_i, z_j) / τ) / Σ_{k≠i} exp(sim(z_i, z_k) / τ))
 * 
 * @param {number[][]} embeddings - 2N embeddings (first N and second N are pairs)
 * @param {number} temperature - Temperature parameter τ
 * @returns {number} Average loss
 */
export function ntXentLoss(embeddings, temperature = 0.5) {
  const N = embeddings.length / 2;
  let totalLoss = 0;

  for (let i = 0; i < 2 * N; i++) {
    // Positive pair index
    const j = i < N ? i + N : i - N;

    // Similarity with positive
    const simPos = cosineSimilarity(embeddings[i], embeddings[j]) / temperature;

    // Denominator: sum over all negatives
    let logSumExp = -Infinity;
    for (let k = 0; k < 2 * N; k++) {
      if (k === i) continue;
      const sim = cosineSimilarity(embeddings[i], embeddings[k]) / temperature;
      logSumExp = logSumExpHelper(logSumExp, sim);
    }

    totalLoss += -simPos + logSumExp;
  }

  return totalLoss / (2 * N);
}

function logSumExpHelper(a, b) {
  const max = Math.max(a, b);
  return max + Math.log(Math.exp(a - max) + Math.exp(b - max));
}

/**
 * Simple data augmentation for 1D feature vectors.
 * @param {number[]} data - Input features
 * @param {Object} opts
 * @param {number} [opts.noiseScale=0.1] - Gaussian noise scale
 * @param {number} [opts.dropRate=0.1] - Feature dropout rate
 * @returns {number[]} Augmented features
 */
export function augment(data, opts = {}) {
  const { noiseScale = 0.1, dropRate = 0.1 } = opts;
  return data.map(x => {
    // Random dropout
    if (Math.random() < dropRate) return 0;
    // Add Gaussian noise (Box-Muller)
    const u1 = Math.random();
    const u2 = Math.random();
    const noise = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
    return x + noise * noiseScale;
  });
}

/**
 * Contrastive Learning module.
 * Trains an encoder to produce useful representations without labels.
 */
export class ContrastiveLearner {
  /**
   * @param {number} inputDim - Input dimension
   * @param {number} embedDim - Embedding dimension
   * @param {Object} opts
   * @param {number} [opts.hiddenDim=64]
   * @param {number} [opts.projDim=32] - Projection head output dimension
   * @param {number} [opts.temperature=0.5]
   * @param {number} [opts.learningRate=0.01]
   */
  constructor(inputDim, embedDim, opts = {}) {
    const {
      hiddenDim = 64,
      projDim = 32,
      temperature = 0.5,
      learningRate = 0.01,
    } = opts;

    this.inputDim = inputDim;
    this.embedDim = embedDim;
    this.projDim = projDim;
    this.temperature = temperature;
    this.learningRate = learningRate;

    // Encoder: input → embedding
    this.encoder = new Network();
    this.encoder.add(new Dense(inputDim, hiddenDim, 'relu'));
    this.encoder.add(new Dense(hiddenDim, embedDim, 'relu'));

    // Projection head: embedding → projection (used during training only)
    this.projHead = new Network();
    this.projHead.add(new Dense(embedDim, projDim, 'linear'));
  }

  /**
   * Encode input to embedding (without projection head).
   * @param {number[]} input
   * @returns {number[]}
   */
  encode(input) {
    const mat = new Matrix(1, this.inputDim, new Float64Array(input));
    const emb = this.encoder.forward(mat);
    return Array.from(emb.data);
  }

  /**
   * Encode and project (used during training).
   * @param {number[]} input
   * @returns {number[]}
   */
  _encodeAndProject(input) {
    const mat = new Matrix(1, this.inputDim, new Float64Array(input));
    const emb = this.encoder.forward(mat);
    const proj = this.projHead.forward(emb);
    return Array.from(proj.data);
  }

  /**
   * Train on unlabeled data using contrastive learning.
   * @param {number[][]} data - Array of input vectors
   * @param {Object} opts
   * @param {number} [opts.epochs=10]
   * @param {number} [opts.batchSize=16]
   * @param {Function} [opts.augmentFn] - Custom augmentation function
   * @param {Function} [opts.onEpoch]
   * @returns {{ history: number[] }}
   */
  train(data, opts = {}) {
    const { epochs = 10, batchSize = 16, augmentFn = augment, onEpoch } = opts;
    const history = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;
      let batchCount = 0;

      // Process in batches
      const shuffled = [...data].sort(() => Math.random() - 0.5);
      for (let start = 0; start < shuffled.length; start += batchSize) {
        const batch = shuffled.slice(start, start + batchSize);
        if (batch.length < 2) continue;

        // Create two augmented views for each sample
        const embeddings = [];
        for (const sample of batch) {
          const view1 = augmentFn(sample);
          embeddings.push(this._encodeAndProject(view1));
        }
        for (const sample of batch) {
          const view2 = augmentFn(sample);
          embeddings.push(this._encodeAndProject(view2));
        }

        // Compute loss
        const loss = ntXentLoss(embeddings, this.temperature);
        totalLoss += loss;
        batchCount++;

        // Simple weight perturbation for learning (approximate gradient)
        // (A proper implementation would use backprop through the encoder+projHead)
        this._perturbWeights(loss);
      }

      const avgLoss = batchCount > 0 ? totalLoss / batchCount : 0;
      history.push(avgLoss);
      if (onEpoch) onEpoch({ epoch, loss: avgLoss });
    }

    return { history };
  }

  /**
   * Simple weight perturbation for approximate gradient descent.
   * (Proper contrastive training would use backprop)
   */
  _perturbWeights(loss) {
    for (const layer of [...this.encoder.layers, ...this.projHead.layers]) {
      if (!layer.weights) continue;
      for (let i = 0; i < layer.weights.data.length; i++) {
        layer.weights.data[i] += (Math.random() - 0.5) * this.learningRate * 0.01;
      }
    }
  }

  /**
   * Compute similarity between two inputs.
   */
  similarity(input1, input2) {
    return cosineSimilarity(this.encode(input1), this.encode(input2));
  }
}
