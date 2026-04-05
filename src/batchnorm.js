// batchnorm.js — Batch Normalization layer
// Ioffe & Szegedy, 2015
//
// Normalizes activations across the batch dimension:
//   y = gamma * (x - mean) / sqrt(var + eps) + beta
//
// During training: uses batch statistics
// During inference: uses running mean/variance

import { Matrix } from './matrix.js';

export class BatchNorm {
  constructor(size, { momentum = 0.1, epsilon = 1e-5 } = {}) {
    this.size = size;
    this.inputSize = size;
    this.outputSize = size;
    this.momentum = momentum;
    this.epsilon = epsilon;
    this.training = true;

    // Learnable parameters
    this.gamma = new Matrix(1, size).fill(1);  // Scale
    this.beta = new Matrix(1, size).fill(0);   // Shift

    // Running statistics (for inference)
    this.runningMean = new Matrix(1, size).fill(0);
    this.runningVar = new Matrix(1, size).fill(1);

    // Cached values for backward pass
    this.xNorm = null;
    this.mean = null;
    this.variance = null;
    this.input = null;
    this.std = null;

    // Gradients
    this.dGamma = null;
    this.dBeta = null;
  }

  forward(input) {
    this.input = input;
    const batchSize = input.rows;
    const output = new Matrix(batchSize, this.size);

    if (this.training) {
      // Compute batch mean: mean[j] = (1/N) * sum_i(x[i][j])
      this.mean = new Matrix(1, this.size);
      for (let j = 0; j < this.size; j++) {
        let sum = 0;
        for (let i = 0; i < batchSize; i++) {
          sum += input.get(i, j);
        }
        this.mean.set(0, j, sum / batchSize);
      }

      // Compute batch variance: var[j] = (1/N) * sum_i((x[i][j] - mean[j])^2)
      this.variance = new Matrix(1, this.size);
      for (let j = 0; j < this.size; j++) {
        let sum = 0;
        const m = this.mean.get(0, j);
        for (let i = 0; i < batchSize; i++) {
          const d = input.get(i, j) - m;
          sum += d * d;
        }
        this.variance.set(0, j, sum / batchSize);
      }

      // Compute std = sqrt(var + eps)
      this.std = new Matrix(1, this.size);
      for (let j = 0; j < this.size; j++) {
        this.std.set(0, j, Math.sqrt(this.variance.get(0, j) + this.epsilon));
      }

      // Normalize: xNorm[i][j] = (x[i][j] - mean[j]) / std[j]
      this.xNorm = new Matrix(batchSize, this.size);
      for (let i = 0; i < batchSize; i++) {
        for (let j = 0; j < this.size; j++) {
          this.xNorm.set(i, j, (input.get(i, j) - this.mean.get(0, j)) / this.std.get(0, j));
        }
      }

      // Update running statistics
      for (let j = 0; j < this.size; j++) {
        this.runningMean.set(0, j,
          (1 - this.momentum) * this.runningMean.get(0, j) + this.momentum * this.mean.get(0, j));
        this.runningVar.set(0, j,
          (1 - this.momentum) * this.runningVar.get(0, j) + this.momentum * this.variance.get(0, j));
      }
    } else {
      // Inference mode: use running statistics
      this.xNorm = new Matrix(batchSize, this.size);
      for (let i = 0; i < batchSize; i++) {
        for (let j = 0; j < this.size; j++) {
          this.xNorm.set(i, j,
            (input.get(i, j) - this.runningMean.get(0, j)) /
            Math.sqrt(this.runningVar.get(0, j) + this.epsilon));
        }
      }
    }

    // Scale and shift: y = gamma * xNorm + beta
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < this.size; j++) {
        output.set(i, j, this.gamma.get(0, j) * this.xNorm.get(i, j) + this.beta.get(0, j));
      }
    }

    return output;
  }

  backward(dOutput) {
    const batchSize = dOutput.rows;
    const N = batchSize;

    // dBeta = sum over batch of dOutput
    this.dBeta = new Matrix(1, this.size);
    for (let j = 0; j < this.size; j++) {
      let sum = 0;
      for (let i = 0; i < N; i++) sum += dOutput.get(i, j);
      this.dBeta.set(0, j, sum);
    }

    // dGamma = sum over batch of (dOutput * xNorm)
    this.dGamma = new Matrix(1, this.size);
    for (let j = 0; j < this.size; j++) {
      let sum = 0;
      for (let i = 0; i < N; i++) sum += dOutput.get(i, j) * this.xNorm.get(i, j);
      this.dGamma.set(0, j, sum);
    }

    // dxNorm = dOutput * gamma
    const dxNorm = new Matrix(N, this.size);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < this.size; j++) {
        dxNorm.set(i, j, dOutput.get(i, j) * this.gamma.get(0, j));
      }
    }

    // dInput = (1/N) * (1/std) * (N * dxNorm - sum(dxNorm) - xNorm * sum(dxNorm * xNorm))
    const dInput = new Matrix(N, this.size);
    for (let j = 0; j < this.size; j++) {
      let sumDx = 0, sumDxXn = 0;
      for (let i = 0; i < N; i++) {
        sumDx += dxNorm.get(i, j);
        sumDxXn += dxNorm.get(i, j) * this.xNorm.get(i, j);
      }
      const invStd = 1.0 / this.std.get(0, j);
      for (let i = 0; i < N; i++) {
        dInput.set(i, j,
          invStd / N * (N * dxNorm.get(i, j) - sumDx - this.xNorm.get(i, j) * sumDxXn));
      }
    }

    return dInput;
  }

  update(learningRate) {
    const batchSize = this.input ? this.input.rows : 1;
    this.gamma = this.gamma.sub(this.dGamma.mul(learningRate / batchSize));
    this.beta = this.beta.sub(this.dBeta.mul(learningRate / batchSize));
  }

  paramCount() {
    return this.size * 2; // gamma + beta
  }
}
