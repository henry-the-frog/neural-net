// dropout.js — Standalone Dropout layer
// Can be inserted between any layers in a network

import { Matrix } from './matrix.js';

export class Dropout {
  constructor(rate = 0.5) {
    this.rate = rate; // probability of dropping a neuron
    this.training = true;
    this.mask = null;
    this.inputSize = null;
    this.outputSize = null;
  }

  forward(input) {
    this.inputSize = input.cols;
    this.outputSize = input.cols;
    
    if (!this.training || this.rate === 0) {
      return input;
    }

    // Create dropout mask: 1/(1-rate) for kept, 0 for dropped (inverted dropout)
    const scale = 1.0 / (1 - this.rate);
    this.mask = input.map(() => Math.random() > this.rate ? scale : 0);
    return input.mul(this.mask);
  }

  backward(dOutput) {
    if (!this.mask) return dOutput;
    return dOutput.mul(this.mask);
  }

  update() {} // No learnable parameters

  paramCount() { return 0; }
}
