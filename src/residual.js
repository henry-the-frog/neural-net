// residual.js — Residual (Skip) connection wrapper

import { Matrix } from './matrix.js';

/**
 * Residual connection — wraps a sublayer with a skip connection
 * output = input + sublayer(input)
 * 
 * This is the key insight from ResNet (He et al. 2015) that enables
 * training very deep networks by allowing gradients to flow directly
 * through skip connections.
 */
export class Residual {
  constructor(sublayer) {
    this.sublayer = sublayer;
    this.outputSize = sublayer.outputSize;
    this.training = true;
    this._input = null;
    this.dWeights = null;
    this.dBiases = null;
  }
  
  forward(input) {
    this._input = input;
    const sublayerOutput = this.sublayer.forward(input);
    
    // Residual connection: output = input + sublayer(input)
    const result = new Matrix(input.rows, input.cols);
    for (let i = 0; i < input.rows; i++) {
      for (let j = 0; j < input.cols; j++) {
        result.set(i, j, input.get(i, j) + sublayerOutput.get(i, j));
      }
    }
    return result;
  }
  
  backward(dOutput) {
    // Gradient flows through both paths:
    // dInput = dOutput (skip) + sublayer.backward(dOutput)
    const dSublayer = this.sublayer.backward(dOutput);
    
    const dInput = new Matrix(dOutput.rows, dOutput.cols);
    for (let i = 0; i < dOutput.rows; i++) {
      for (let j = 0; j < dOutput.cols; j++) {
        dInput.set(i, j, dOutput.get(i, j) + dSublayer.get(i, j));
      }
    }
    
    // Propagate sublayer gradients for optimizer
    this.dWeights = this.sublayer.dWeights;
    this.dBiases = this.sublayer.dBiases;
    
    return dInput;
  }
  
  update(learningRate, momentum, optimizer) {
    if (this.sublayer.update) {
      this.sublayer.update(learningRate, momentum, optimizer);
    }
  }
  
  paramCount() {
    return this.sublayer.paramCount ? this.sublayer.paramCount() : 0;
  }
}

/**
 * Sequential — chains multiple layers together as a single layer
 * Useful for building sublayers for Residual connections
 */
export class Sequential {
  constructor(layers) {
    this.layers = layers;
    this.outputSize = layers[layers.length - 1].outputSize;
    this.training = true;
    this.dWeights = null;
    this.dBiases = null;
  }
  
  forward(input) {
    let x = input;
    for (const layer of this.layers) {
      x = layer.forward(x);
    }
    return x;
  }
  
  backward(dOutput) {
    let grad = dOutput;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      grad = this.layers[i].backward(grad);
    }
    // Store first layer's gradients for optimizer compat
    this.dWeights = this.layers[0].dWeights;
    this.dBiases = this.layers[0].dBiases;
    return grad;
  }
  
  update(learningRate, momentum, optimizer) {
    for (const layer of this.layers) {
      if (layer.update) layer.update(learningRate, momentum, optimizer);
    }
  }
  
  paramCount() {
    return this.layers.reduce((sum, l) => sum + (l.paramCount ? l.paramCount() : 0), 0);
  }
}
