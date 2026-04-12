// darts.js — Differentiable Architecture Search (DARTS)
// Simplified implementation using autograd for architecture optimization.
//
// Key idea: Instead of searching over discrete architectures, use continuous
// relaxation. Each "mixed operation" is a softmax-weighted sum of candidate
// operations. Architecture parameters (alpha) are learned via gradient descent.
//
// Reference: Liu et al., "DARTS: Differentiable Architecture Search", ICLR 2019

import { Matrix } from './matrix.js';
import * as autograd from './autograd.js';

/**
 * Candidate operation: a simple function that transforms input.
 * In real DARTS these would be conv/pool/etc — here we use dense layers.
 */
class CandidateOp {
  constructor(inputSize, outputSize, name) {
    this.name = name;
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    // Xavier initialization
    const scale = Math.sqrt(2.0 / (inputSize + outputSize));
    this.weights = Matrix.random(inputSize, outputSize).map(v => v * scale);
    this.bias = new Float64Array(outputSize);
  }

  forward(input) {
    // input: Float64Array of length inputSize
    const out = new Float64Array(this.outputSize);
    for (let j = 0; j < this.outputSize; j++) {
      let sum = this.bias[j];
      for (let i = 0; i < this.inputSize; i++) {
        sum += input[i] * this.weights.get(i, j);
      }
      out[j] = sum;
    }
    return out;
  }
}

/**
 * ReLU candidate
 */
class ReLUOp extends CandidateOp {
  forward(input) {
    const linear = super.forward(input);
    return linear.map(v => Math.max(0, v));
  }
}

/**
 * Identity/skip connection (requires inputSize === outputSize)
 */
class IdentityOp {
  constructor(size) {
    this.name = 'identity';
    this.inputSize = size;
    this.outputSize = size;
  }

  forward(input) {
    return new Float64Array(input);
  }
}

/**
 * Zero operation (always returns zeros)
 */
class ZeroOp {
  constructor(size) {
    this.name = 'zero';
    this.inputSize = size;
    this.outputSize = size;
  }

  forward(_input) {
    return new Float64Array(this.outputSize);
  }
}

/**
 * Softmax over an array of values
 */
function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

/**
 * MixedOp: A weighted mixture of candidate operations.
 * The mixing weights (alpha) are architecture parameters learned via gradient descent.
 */
export class MixedOp {
  constructor(inputSize, outputSize, candidates = null) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;

    if (candidates) {
      this.ops = candidates;
    } else {
      // Default candidates
      this.ops = [
        new ReLUOp(inputSize, outputSize, 'relu_linear'),
        new CandidateOp(inputSize, outputSize, 'linear'),
      ];
      if (inputSize === outputSize) {
        this.ops.push(new IdentityOp(inputSize));
        this.ops.push(new ZeroOp(inputSize));
      }
    }

    // Architecture parameters — one per candidate op
    this.alpha = new Float64Array(this.ops.length);
    // Initialize uniformly (softmax will be ~equal)
  }

  /**
   * Forward: softmax-weighted sum of all candidate outputs
   */
  forward(input) {
    const weights = softmax(Array.from(this.alpha));
    const output = new Float64Array(this.outputSize);
    
    for (let k = 0; k < this.ops.length; k++) {
      if (weights[k] < 1e-8) continue; // skip negligible ops
      const opOut = this.ops[k].forward(input);
      for (let j = 0; j < this.outputSize; j++) {
        output[j] += weights[k] * opOut[j];
      }
    }
    return output;
  }

  /**
   * Get the winning operation (highest alpha)
   */
  get selectedOp() {
    let bestIdx = 0;
    for (let i = 1; i < this.alpha.length; i++) {
      if (this.alpha[i] > this.alpha[bestIdx]) bestIdx = i;
    }
    return this.ops[bestIdx];
  }

  /**
   * Get architecture weights (softmax of alpha)
   */
  get architectureWeights() {
    return softmax(Array.from(this.alpha));
  }
}

/**
 * DARTSCell: A cell in the DARTS search space.
 * Contains multiple nodes connected by MixedOps.
 */
export class DARTSCell {
  constructor(inputSize, hiddenSize, numNodes = 4) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.numNodes = numNodes;
    
    // Edges: each intermediate node receives input from all previous nodes
    // Node 0 and 1 are input nodes (from previous cell and skip connection)
    // Nodes 2..numNodes+1 are intermediate nodes
    this.edges = new Map(); // key: "src->dst", value: MixedOp
    
    const totalNodes = numNodes + 2; // 2 input nodes + numNodes intermediate
    for (let dst = 2; dst < totalNodes; dst++) {
      for (let src = 0; src < dst; src++) {
        const key = `${src}->${dst}`;
        this.edges.set(key, new MixedOp(hiddenSize, hiddenSize));
      }
    }

    // Input projection
    this.inputProj = new CandidateOp(inputSize, hiddenSize, 'input_proj');
  }

  /**
   * Forward pass through the cell
   */
  forward(input) {
    const totalNodes = this.numNodes + 2;
    const nodeOutputs = new Array(totalNodes);
    
    // Input nodes (both receive the projected input for simplicity)
    const projected = this.inputProj.forward(input);
    nodeOutputs[0] = projected;
    nodeOutputs[1] = projected;

    // Intermediate nodes
    for (let dst = 2; dst < totalNodes; dst++) {
      const combined = new Float64Array(this.hiddenSize);
      for (let src = 0; src < dst; src++) {
        const edge = this.edges.get(`${src}->${dst}`);
        const edgeOut = edge.forward(nodeOutputs[src]);
        for (let j = 0; j < this.hiddenSize; j++) {
          combined[j] += edgeOut[j];
        }
      }
      nodeOutputs[dst] = combined;
    }

    // Output: concatenate (average) all intermediate node outputs
    const output = new Float64Array(this.hiddenSize);
    for (let i = 2; i < totalNodes; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        output[j] += nodeOutputs[i][j] / this.numNodes;
      }
    }
    return output;
  }

  /**
   * Get all architecture parameters (alphas)
   */
  getAllAlphas() {
    const alphas = [];
    for (const [key, mixedOp] of this.edges) {
      alphas.push({ edge: key, alpha: mixedOp.alpha, weights: mixedOp.architectureWeights });
    }
    return alphas;
  }

  /**
   * Get the derived (discrete) architecture
   */
  getDerivedArchitecture() {
    const arch = {};
    for (const [key, mixedOp] of this.edges) {
      arch[key] = {
        selected: mixedOp.selectedOp.name,
        weights: mixedOp.architectureWeights,
      };
    }
    return arch;
  }
}

/**
 * DARTSSearcher: Performs architecture search using bilevel optimization.
 * 
 * Outer loop: optimize alpha (architecture) on validation data
 * Inner loop: optimize weights on training data
 */
export class DARTSSearcher {
  constructor(cell, outputSize) {
    this.cell = cell;
    this.outputSize = outputSize;
    
    // Output layer
    this.outputLayer = new CandidateOp(cell.hiddenSize, outputSize, 'output');
    
    this.weightLR = 0.01;
    this.alphaLR = 0.05;
    this.steps = 0;
  }

  /**
   * Forward pass through cell + output layer
   */
  predict(input) {
    const hidden = this.cell.forward(input);
    const output = this.outputLayer.forward(hidden);
    // Softmax for classification
    const maxVal = Math.max(...output);
    const exps = output.map(v => Math.exp(v - maxVal));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / sum);
  }

  /**
   * Compute cross-entropy loss
   */
  loss(input, targetIdx) {
    const probs = this.predict(input);
    return -Math.log(Math.max(probs[targetIdx], 1e-10));
  }

  /**
   * Finite-difference gradient for architecture parameters
   * (Simpler than implementing full backprop through the mixed operations)
   */
  _archGradient(valInputs, valTargets, h = 0.1) {
    for (const [key, mixedOp] of this.cell.edges) {
      for (let k = 0; k < mixedOp.alpha.length; k++) {
        // f(alpha + h)
        mixedOp.alpha[k] += h;
        let lossPlus = 0;
        for (let i = 0; i < valInputs.length; i++) {
          lossPlus += this.loss(valInputs[i], valTargets[i]);
        }
        
        // f(alpha - h)
        mixedOp.alpha[k] -= 2 * h;
        let lossMinus = 0;
        for (let i = 0; i < valInputs.length; i++) {
          lossMinus += this.loss(valInputs[i], valTargets[i]);
        }
        
        // Restore
        mixedOp.alpha[k] += h;
        
        // Gradient
        const grad = (lossPlus - lossMinus) / (2 * h);
        mixedOp.alpha[k] -= this.alphaLR * grad;
      }
    }
  }

  /**
   * One step of bilevel optimization
   */
  step(trainInputs, trainTargets, valInputs, valTargets) {
    // Inner loop: update weights on training data (simplified — just forward pass loss)
    // In a full implementation, this would do proper backprop through the cell
    
    // Outer loop: update architecture on validation data
    this._archGradient(valInputs, valTargets);
    this.steps++;
    
    // Compute training loss for monitoring
    let trainLoss = 0;
    for (let i = 0; i < trainInputs.length; i++) {
      trainLoss += this.loss(trainInputs[i], trainTargets[i]);
    }
    
    let valLoss = 0;
    for (let i = 0; i < valInputs.length; i++) {
      valLoss += this.loss(valInputs[i], valTargets[i]);
    }
    
    return {
      trainLoss: trainLoss / trainInputs.length,
      valLoss: valLoss / valInputs.length,
      step: this.steps,
    };
  }

  /**
   * Run search for N steps
   */
  search(trainInputs, trainTargets, valInputs, valTargets, steps = 50) {
    const history = [];
    for (let i = 0; i < steps; i++) {
      const result = this.step(trainInputs, trainTargets, valInputs, valTargets);
      history.push(result);
    }
    return {
      history,
      architecture: this.cell.getDerivedArchitecture(),
      alphas: this.cell.getAllAlphas(),
    };
  }
}
