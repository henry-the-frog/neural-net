// initializers.js — Weight initialization strategies for neural networks
//
// Proper initialization prevents vanishing/exploding gradients:
//   Xavier/Glorot: for sigmoid/tanh (Var = 2/(fanIn + fanOut))
//   He/Kaiming: for ReLU (Var = 2/fanIn)
//   LeCun: for SELU/sigmoid (Var = 1/fanIn)

import { Matrix } from './matrix.js';

/**
 * Xavier/Glorot initialization (uniform)
 * Suitable for: sigmoid, tanh, softmax
 * W ~ U(-limit, limit) where limit = sqrt(6 / (fanIn + fanOut))
 */
export function xavierUniform(rows, cols, fanIn, fanOut) {
  const limit = Math.sqrt(6.0 / (fanIn + fanOut));
  const m = new Matrix(rows, cols);
  for (let i = 0; i < m.data.length; i++) {
    m.data[i] = (Math.random() * 2 - 1) * limit;
  }
  return m;
}

/**
 * Xavier/Glorot initialization (normal)
 * W ~ N(0, sqrt(2 / (fanIn + fanOut)))
 */
export function xavierNormal(rows, cols, fanIn, fanOut) {
  const std = Math.sqrt(2.0 / (fanIn + fanOut));
  const m = new Matrix(rows, cols);
  for (let i = 0; i < m.data.length; i++) {
    m.data[i] = randomNormal() * std;
  }
  return m;
}

/**
 * He/Kaiming initialization (uniform)
 * Suitable for: ReLU, Leaky ReLU
 * W ~ U(-limit, limit) where limit = sqrt(6 / fanIn)
 */
export function heUniform(rows, cols, fanIn) {
  const limit = Math.sqrt(6.0 / fanIn);
  const m = new Matrix(rows, cols);
  for (let i = 0; i < m.data.length; i++) {
    m.data[i] = (Math.random() * 2 - 1) * limit;
  }
  return m;
}

/**
 * He/Kaiming initialization (normal)
 * W ~ N(0, sqrt(2 / fanIn))
 */
export function heNormal(rows, cols, fanIn) {
  const std = Math.sqrt(2.0 / fanIn);
  const m = new Matrix(rows, cols);
  for (let i = 0; i < m.data.length; i++) {
    m.data[i] = randomNormal() * std;
  }
  return m;
}

/**
 * LeCun initialization
 * Suitable for: SELU, sigmoid
 * W ~ N(0, sqrt(1 / fanIn))
 */
export function lecunNormal(rows, cols, fanIn) {
  const std = Math.sqrt(1.0 / fanIn);
  const m = new Matrix(rows, cols);
  for (let i = 0; i < m.data.length; i++) {
    m.data[i] = randomNormal() * std;
  }
  return m;
}

/**
 * Zeros initialization (for biases)
 */
export function zeros(rows, cols) {
  return new Matrix(rows, cols); // Already zeroed
}

/**
 * Ones initialization
 */
export function ones(rows, cols) {
  const m = new Matrix(rows, cols);
  for (let i = 0; i < m.data.length; i++) m.data[i] = 1;
  return m;
}

/**
 * Create initializer by name
 */
export function createInitializer(name) {
  switch (name) {
    case 'xavier_uniform': case 'glorot_uniform': return xavierUniform;
    case 'xavier_normal': case 'glorot_normal': return xavierNormal;
    case 'he_uniform': case 'kaiming_uniform': return heUniform;
    case 'he_normal': case 'kaiming_normal': return heNormal;
    case 'lecun': case 'lecun_normal': return lecunNormal;
    case 'zeros': return zeros;
    case 'ones': return ones;
    default: throw new Error(`Unknown initializer: ${name}`);
  }
}

// Box-Muller transform for normal distribution
function randomNormal() {
  let u1, u2;
  do { u1 = Math.random(); } while (u1 === 0);
  u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
