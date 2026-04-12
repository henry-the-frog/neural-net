// hypernetwork.js — Networks that generate weights for other networks
// The "hypernetwork" produces weights conditioned on task/context embeddings

import { Matrix } from './matrix.js';
import { Dense } from './layer.js';

// ===== Simple Hypernetwork =====
// Given a task embedding z, generate weights for a target network
export class HyperNetwork {
  constructor(embeddingDim, targetLayerSizes) {
    this.embeddingDim = embeddingDim;
    this.targetLayerSizes = targetLayerSizes;

    // Compute total number of parameters in target network
    this.targetParamCount = 0;
    this.layerParamCounts = [];
    for (let l = 0; l < targetLayerSizes.length - 1; l++) {
      const params = targetLayerSizes[l] * targetLayerSizes[l + 1] + targetLayerSizes[l + 1];
      this.layerParamCounts.push(params);
      this.targetParamCount += params;
    }

    // Weight generator: embedding → target params
    // Use chunked generation for scalability
    this.generators = [];
    for (let l = 0; l < targetLayerSizes.length - 1; l++) {
      const paramCount = this.layerParamCounts[l];
      // Small MLP per layer: embedding → hidden → params
      const hiddenSize = Math.min(64, Math.max(16, Math.floor(paramCount / 4)));
      this.generators.push({
        w1: Array.from({ length: hiddenSize }, () =>
          Array.from({ length: embeddingDim }, () => (Math.random() - 0.5) * Math.sqrt(2 / embeddingDim))
        ),
        b1: new Array(hiddenSize).fill(0),
        w2: Array.from({ length: paramCount }, () =>
          Array.from({ length: hiddenSize }, () => (Math.random() - 0.5) * Math.sqrt(2 / hiddenSize) * 0.1)
        ),
        b2: new Array(paramCount).fill(0),
      });
    }
  }

  // Generate target network weights from embedding
  generateWeights(embedding) {
    const allParams = [];

    for (let l = 0; l < this.generators.length; l++) {
      const gen = this.generators[l];

      // Hidden layer
      const hidden = gen.w1.map((row, h) => {
        let sum = gen.b1[h];
        for (let i = 0; i < this.embeddingDim; i++) sum += row[i] * embedding[i];
        return Math.tanh(sum);
      });

      // Output layer (target params)
      const params = gen.w2.map((row, p) => {
        let sum = gen.b2[p];
        for (let h = 0; h < hidden.length; h++) sum += row[h] * hidden[h];
        return sum;
      });

      allParams.push(params);
    }

    return allParams;
  }

  // Create a target network with generated weights
  createTargetNetwork(embedding) {
    const paramSets = this.generateWeights(embedding);
    const layers = [];

    for (let l = 0; l < this.targetLayerSizes.length - 1; l++) {
      const inSize = this.targetLayerSizes[l];
      const outSize = this.targetLayerSizes[l + 1];
      const isLast = l === this.targetLayerSizes.length - 2;
      const layer = new Dense(inSize, outSize, isLast ? 'linear' : 'tanh');

      // Set weights from generated params
      const params = paramSets[l];
      let idx = 0;
      for (let i = 0; i < inSize; i++)
        for (let j = 0; j < outSize; j++)
          layer.weights.set(i, j, params[idx++]);
      for (let j = 0; j < outSize; j++)
        layer.biases.set(0, j, params[idx++]);

      layers.push(layer);
    }

    return {
      layers,
      forward(input) {
        let x = input;
        for (const layer of this.layers) x = layer.forward(x);
        return x;
      },
    };
  }

  // Forward: embedding + input → output through generated network
  forward(embedding, input) {
    const network = this.createTargetNetwork(embedding);
    return network.forward(input);
  }

  hyperParamCount() {
    let total = 0;
    for (const gen of this.generators) {
      total += gen.w1.length * gen.w1[0].length + gen.b1.length;
      total += gen.w2.length * gen.w2[0].length + gen.b2.length;
    }
    return total;
  }
}

// ===== Task-Conditioned HyperNetwork =====
// Learns task embeddings and generates specialized networks per task
export class TaskConditionedHyperNetwork {
  constructor(numTasks, embeddingDim, targetLayerSizes) {
    this.numTasks = numTasks;
    this.embeddingDim = embeddingDim;

    // Learnable task embeddings
    this.taskEmbeddings = Array.from({ length: numTasks }, () =>
      Array.from({ length: embeddingDim }, () => (Math.random() - 0.5) * 0.5)
    );

    this.hyperNet = new HyperNetwork(embeddingDim, targetLayerSizes);
  }

  // Forward for a specific task
  forward(taskId, input) {
    if (taskId < 0 || taskId >= this.numTasks) throw new Error(`Invalid task: ${taskId}`);
    return this.hyperNet.forward(this.taskEmbeddings[taskId], input);
  }

  // Get the network for a specific task
  getTaskNetwork(taskId) {
    return this.hyperNet.createTargetNetwork(this.taskEmbeddings[taskId]);
  }

  // Interpolate between task networks
  interpolate(taskId1, taskId2, alpha) {
    const e1 = this.taskEmbeddings[taskId1];
    const e2 = this.taskEmbeddings[taskId2];
    const interp = e1.map((v, i) => v * (1 - alpha) + e2[i] * alpha);
    return this.hyperNet.createTargetNetwork(interp);
  }
}

// ===== FiLM (Feature-wise Linear Modulation) =====
// Lightweight conditioning: generate scale + shift per feature
export class FiLM {
  constructor(conditionDim, featureDim) {
    this.conditionDim = conditionDim;
    this.featureDim = featureDim;

    // Generate gamma (scale) and beta (shift) from conditioning input
    this.gammaWeights = Array.from({ length: featureDim }, () =>
      Array.from({ length: conditionDim }, () => (Math.random() - 0.5) * 0.3)
    );
    this.gammaBias = new Array(featureDim).fill(1); // Initialize to identity

    this.betaWeights = Array.from({ length: featureDim }, () =>
      Array.from({ length: conditionDim }, () => (Math.random() - 0.5) * 0.3)
    );
    this.betaBias = new Array(featureDim).fill(0);
  }

  // Compute gamma and beta from conditioning
  computeParams(condition) {
    const gamma = this.gammaWeights.map((row, f) => {
      let sum = this.gammaBias[f];
      for (let i = 0; i < this.conditionDim; i++) sum += row[i] * condition[i];
      return sum;
    });
    const beta = this.betaWeights.map((row, f) => {
      let sum = this.betaBias[f];
      for (let i = 0; i < this.conditionDim; i++) sum += row[i] * condition[i];
      return sum;
    });
    return { gamma, beta };
  }

  // Modulate features: output = gamma * features + beta
  modulate(features, condition) {
    const { gamma, beta } = this.computeParams(condition);
    return features.map((v, i) => gamma[i] * v + beta[i]);
  }
}
