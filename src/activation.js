// activation.js — Activation functions and their derivatives

import { Matrix } from './matrix.js';

// Sigmoid: 1 / (1 + e^-x)
export const sigmoid = {
  name: 'sigmoid',
  forward(x) {
    return x.map(v => 1 / (1 + Math.exp(-v)));
  },
  backward(output) {
    // derivative: σ(x) * (1 - σ(x)) — computed from output
    return output.mul(output.map(v => 1 - v));
  }
};

// ReLU: max(0, x)
export const relu = {
  name: 'relu',
  forward(x) {
    return x.map(v => Math.max(0, v));
  },
  backward(output) {
    return output.map(v => v > 0 ? 1 : 0);
  }
};

// Leaky ReLU: x > 0 ? x : 0.01x
export const leakyRelu = {
  name: 'leaky_relu',
  forward(x) {
    return x.map(v => v > 0 ? v : 0.01 * v);
  },
  backward(output) {
    return output.map(v => v > 0 ? 1 : 0.01);
  }
};

// Tanh
export const tanh = {
  name: 'tanh',
  forward(x) {
    return x.map(v => Math.tanh(v));
  },
  backward(output) {
    // derivative: 1 - tanh²(x)
    return output.map(v => 1 - v * v);
  }
};

// Softmax (applied per row)
export const softmax = {
  name: 'softmax',
  forward(x) {
    const result = new Matrix(x.rows, x.cols);
    for (let i = 0; i < x.rows; i++) {
      // Numerical stability: subtract max
      let maxVal = -Infinity;
      for (let j = 0; j < x.cols; j++) {
        const v = x.get(i, j);
        if (v > maxVal) maxVal = v;
      }
      let sumExp = 0;
      for (let j = 0; j < x.cols; j++) {
        sumExp += Math.exp(x.get(i, j) - maxVal);
      }
      for (let j = 0; j < x.cols; j++) {
        result.set(i, j, Math.exp(x.get(i, j) - maxVal) / sumExp);
      }
    }
    return result;
  },
  backward(output) {
    // For softmax + cross-entropy, the gradient is simplified
    // We'll handle this in the loss function
    return Matrix.ones(output.rows, output.cols);
  }
};

// Linear (identity) — no activation
export const linear = {
  name: 'linear',
  forward(x) { return x.clone(); },
  backward(output) { return Matrix.ones(output.rows, output.cols); }
};

// Get activation by name
export function getActivation(name) {
  const activations = { sigmoid, relu, leaky_relu: leakyRelu, tanh, softmax, linear };
  return activations[name] || linear;
}
