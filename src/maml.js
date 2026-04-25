// maml.js — Model-Agnostic Meta-Learning (MAML)
// Learn initialization that can quickly adapt to new tasks
// Based on "Model-Agnostic Meta-Learning for Fast Adaptation" (Finn et al., 2017)

import { Matrix } from './matrix.js';
import { Dense } from './layer.js';

// ===== Simple Network for MAML =====
// Thin wrapper that supports weight cloning and fast adaptation
export class MAMLNetwork {
  constructor(layerSizes) {
    this.layerSizes = layerSizes;
    this.layers = [];
    for (let l = 0; l < layerSizes.length - 1; l++) {
      this.layers.push(new Dense(layerSizes[l], layerSizes[l + 1],
        l < layerSizes.length - 2 ? 'relu' : 'linear'));
    }
  }

  forward(input) {
    let x = input;
    for (const layer of this.layers) x = layer.forward(x);
    return x;
  }

  // Get all parameters as flat array
  getParams() {
    const params = [];
    for (const layer of this.layers) {
      for (let i = 0; i < layer.weights.rows; i++)
        for (let j = 0; j < layer.weights.cols; j++)
          params.push(layer.weights.get(i, j));
      for (let j = 0; j < layer.biases.cols; j++)
        params.push(layer.biases.get(0, j));
    }
    return params;
  }

  // Set all parameters from flat array
  setParams(params) {
    let idx = 0;
    for (const layer of this.layers) {
      for (let i = 0; i < layer.weights.rows; i++)
        for (let j = 0; j < layer.weights.cols; j++)
          layer.weights.set(i, j, params[idx++]);
      for (let j = 0; j < layer.biases.cols; j++)
        layer.biases.set(0, j, params[idx++]);
    }
  }

  clone() {
    const net = new MAMLNetwork(this.layerSizes);
    net.setParams(this.getParams());
    return net;
  }

  paramCount() {
    return this.getParams().length;
  }

  // Compute MSE loss and gradients
  computeLoss(inputs, targets) {
    const output = this.forward(inputs);
    let loss = 0;
    const dOutput = new Matrix(output.rows, output.cols);
    for (let i = 0; i < output.rows; i++) {
      for (let j = 0; j < output.cols; j++) {
        const diff = output.get(i, j) - targets.get(i, j);
        loss += diff * diff;
        dOutput.set(i, j, 2 * diff / output.rows);
      }
    }
    loss /= output.rows;

    // Backward
    let dx = dOutput;
    for (let l = this.layers.length - 1; l >= 0; l--) {
      dx = this.layers[l].backward(dx);
    }

    return { loss, output };
  }

  // Get gradient as flat array (after backward)
  getGradients() {
    const grads = [];
    for (const layer of this.layers) {
      if (!layer.dWeights) {
        grads.push(...new Array(layer.weights.rows * layer.weights.cols + layer.biases.cols).fill(0));
        continue;
      }
      for (let i = 0; i < layer.dWeights.rows; i++)
        for (let j = 0; j < layer.dWeights.cols; j++)
          grads.push(layer.dWeights.get(i, j));
      for (let j = 0; j < layer.dBiases.cols; j++)
        grads.push(layer.dBiases.get(0, j));
    }
    return grads;
  }

  // SGD update
  sgdStep(learningRate) {
    for (const layer of this.layers) {
      if (layer.dWeights) layer.update(learningRate, 0, 'sgd');
    }
  }
}

// ===== MAML Algorithm =====
export class MAML {
  constructor(layerSizes, {
    innerLR = 0.01,
    outerLR = 0.001,
    innerSteps = 5,
    gradClipNorm = 10.0,
  } = {}) {
    this.model = new MAMLNetwork(layerSizes);
    this.innerLR = innerLR;
    this.outerLR = outerLR;
    this.innerSteps = innerSteps;
    this.gradClipNorm = gradClipNorm;
  }

  // Inner loop: adapt to a single task
  adapt(task, steps = null) {
    const adaptSteps = steps || this.innerSteps;
    const adapted = this.model.clone();

    for (let step = 0; step < adaptSteps; step++) {
      adapted.computeLoss(task.supportInputs, task.supportTargets);
      adapted.sgdStep(this.innerLR);
    }

    return adapted;
  }

  // Outer loop: meta-train on batch of tasks
  metaTrainStep(tasks) {
    const metaParams = this.model.getParams();
    const metaGradients = new Array(metaParams.length).fill(0);
    let totalLoss = 0;

    for (const task of tasks) {
      // Inner loop: adapt
      const adapted = this.adapt(task);

      // Evaluate on query set
      const { loss } = adapted.computeLoss(task.queryInputs, task.queryTargets);
      totalLoss += loss;

      // Approximate meta-gradient:
      // The true MAML differentiates through the inner loop (second-order).
      // First-order approximation (FOMAML): just use adapted network's gradient
      const adaptedParams = adapted.getParams();
      const adaptedGrads = adapted.getGradients();

      // Meta-gradient: direction that improves post-adaptation performance
      for (let i = 0; i < metaParams.length; i++) {
        metaGradients[i] += adaptedGrads[i] / tasks.length;
      }
    }

    // Gradient clipping (L2 norm)
    if (this.gradClipNorm > 0) {
      const norm = Math.sqrt(metaGradients.reduce((s, g) => s + g * g, 0));
      if (norm > this.gradClipNorm) {
        const scale = this.gradClipNorm / norm;
        for (let i = 0; i < metaGradients.length; i++) {
          metaGradients[i] *= scale;
        }
      }
    }

    // Meta-update
    const newParams = metaParams.map((p, i) => p - this.outerLR * metaGradients[i]);
    this.model.setParams(newParams);

    return totalLoss / tasks.length;
  }

  // Meta-train for multiple iterations
  metaTrain(taskGenerator, iterations = 100, tasksPerBatch = 4) {
    const losses = [];
    for (let iter = 0; iter < iterations; iter++) {
      const tasks = Array.from({ length: tasksPerBatch }, () => taskGenerator());
      const loss = this.metaTrainStep(tasks);
      losses.push(loss);
    }
    return losses;
  }

  // Test: adapt to new task and evaluate
  test(task, adaptSteps = null) {
    const adapted = this.adapt(task, adaptSteps);
    const { loss } = adapted.computeLoss(task.queryInputs, task.queryTargets);
    return { loss, adaptedModel: adapted };
  }
}

// ===== Task Generators =====

// Sinusoidal regression tasks (classic MAML benchmark)
export function sinusoidTaskGenerator(numSupport = 5, numQuery = 5) {
  return () => {
    const amplitude = Math.random() * 4.5 + 0.1;
    const phase = Math.random() * Math.PI * 2;

    const makeData = (n) => {
      const inputs = new Matrix(n, 1);
      const targets = new Matrix(n, 1);
      for (let i = 0; i < n; i++) {
        const x = Math.random() * 10 - 5;
        inputs.set(i, 0, x);
        targets.set(i, 0, amplitude * Math.sin(x + phase));
      }
      return { inputs, targets };
    };

    const support = makeData(numSupport);
    const query = makeData(numQuery);

    return {
      supportInputs: support.inputs,
      supportTargets: support.targets,
      queryInputs: query.inputs,
      queryTargets: query.targets,
    };
  };
}

// Linear regression tasks
export function linearTaskGenerator(inputDim = 1, numSupport = 5, numQuery = 5) {
  return () => {
    // Random linear function
    const slope = Array.from({ length: inputDim }, () => Math.random() * 4 - 2);
    const bias = Math.random() * 2 - 1;

    const makeData = (n) => {
      const inputs = new Matrix(n, inputDim);
      const targets = new Matrix(n, 1);
      for (let i = 0; i < n; i++) {
        let y = bias;
        for (let d = 0; d < inputDim; d++) {
          const x = Math.random() * 4 - 2;
          inputs.set(i, d, x);
          y += slope[d] * x;
        }
        targets.set(i, 0, y);
      }
      return { inputs, targets };
    };

    const support = makeData(numSupport);
    const query = makeData(numQuery);

    return {
      supportInputs: support.inputs,
      supportTargets: support.targets,
      queryInputs: query.inputs,
      queryTargets: query.targets,
    };
  };
}
