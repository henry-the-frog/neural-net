// layer.js — Dense (fully connected) layer

import { Matrix } from './matrix.js';
import { getActivation } from './activation.js';

export class Dense {
  constructor(inputSize, outputSize, activation = 'relu', { dropout = 0 } = {}) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.activation = getActivation(activation);
    this.dropoutRate = dropout;

    // Weights and biases (Xavier initialized)
    this.weights = Matrix.random(inputSize, outputSize);
    this.biases = Matrix.zeros(1, outputSize);

    // Cache for backpropagation
    this.input = null;
    this.z = null;  // Pre-activation
    this.a = null;  // Post-activation (output)
    this.dropoutMask = null;
    this.training = true;

    // Momentum (velocity)
    this.vWeights = Matrix.zeros(inputSize, outputSize);
    this.vBiases = Matrix.zeros(1, outputSize);
    
    // Adam state (second moment)
    this.sWeights = Matrix.zeros(inputSize, outputSize);
    this.sBiases = Matrix.zeros(1, outputSize);
    this.adamT = 0; // Time step

    // Gradients
    this.dWeights = null;
    this.dBiases = null;
  }

  // Forward pass: output = activation(input · weights + bias)
  forward(input) {
    this.input = input;
    this.z = input.dot(this.weights).add(this.biases);
    this.a = this.activation.forward(this.z);

    // Apply dropout during training
    if (this.dropoutRate > 0 && this.training) {
      this.dropoutMask = this.a.map(() => Math.random() > this.dropoutRate ? 1 / (1 - this.dropoutRate) : 0);
      this.a = this.a.mul(this.dropoutMask);
    }

    return this.a;
  }

  // Backward pass: compute gradients and return gradient for previous layer
  backward(dOutput) {
    // If softmax, dOutput is already the combined gradient
    let dz;
    if (this.activation.name === 'softmax') {
      dz = dOutput; // Cross-entropy + softmax: dz = output - target
    } else {
      // Element-wise: dz = dOutput * activation'(z)
      const activGrad = this.activation.backward(this.a);
      dz = dOutput.mul(activGrad);
    }

    // Gradient for weights: input^T · dz
    this.dWeights = this.input.T().dot(dz);
    // Gradient for biases: sum of dz along batch axis
    this.dBiases = dz.sumAxis(0);
    // Gradient for input (to pass to previous layer): dz · weights^T
    return dz.dot(this.weights.T());
  }

  // Update weights with optimizer
  update(learningRate, momentum = 0, optimizer = 'sgd') {
    const batchSize = this.input.rows;
    const gradW = this.dWeights.mul(1.0 / batchSize);
    const gradB = this.dBiases.mul(1.0 / batchSize);

    if (optimizer === 'adam') {
      this.adamT++;
      const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;

      // First moment (momentum)
      this.vWeights = this.vWeights.mul(beta1).add(gradW.mul(1 - beta1));
      this.vBiases = this.vBiases.mul(beta1).add(gradB.mul(1 - beta1));

      // Second moment (RMSProp)
      this.sWeights = this.sWeights.mul(beta2).add(gradW.mul(gradW).mul(1 - beta2));
      this.sBiases = this.sBiases.mul(beta2).add(gradB.mul(gradB).mul(1 - beta2));

      // Bias correction
      const bc1 = 1 - Math.pow(beta1, this.adamT);
      const bc2 = 1 - Math.pow(beta2, this.adamT);
      const vwHat = this.vWeights.mul(1.0 / bc1);
      const vbHat = this.vBiases.mul(1.0 / bc1);
      const swHat = this.sWeights.mul(1.0 / bc2);
      const sbHat = this.sBiases.mul(1.0 / bc2);

      // Update
      this.weights = this.weights.sub(vwHat.mul(learningRate).mul(swHat.map(v => 1 / (Math.sqrt(v) + eps))));
      this.biases = this.biases.sub(vbHat.mul(learningRate).mul(sbHat.map(v => 1 / (Math.sqrt(v) + eps))));
    } else if (momentum > 0) {
      this.vWeights = this.vWeights.mul(momentum).add(gradW.mul(learningRate));
      this.vBiases = this.vBiases.mul(momentum).add(gradB.mul(learningRate));
      this.weights = this.weights.sub(this.vWeights);
      this.biases = this.biases.sub(this.vBiases);
    } else {
      this.weights = this.weights.sub(gradW.mul(learningRate));
      this.biases = this.biases.sub(gradB.mul(learningRate));
    }
  }

  // Parameter count
  paramCount() {
    return this.inputSize * this.outputSize + this.outputSize;
  }
}
