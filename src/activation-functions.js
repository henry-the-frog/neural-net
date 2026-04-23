// activation-functions.js — Comprehensive Activation Functions
// All major activations used in deep learning, with derivatives.

/**
 * ReLU: max(0, x)
 */
export function relu(x) { return Math.max(0, x); }
export function reluDerivative(x) { return x > 0 ? 1 : 0; }

/**
 * Leaky ReLU: max(αx, x)
 */
export function leakyRelu(x, alpha = 0.01) { return x > 0 ? x : alpha * x; }
export function leakyReluDerivative(x, alpha = 0.01) { return x > 0 ? 1 : alpha; }

/**
 * GELU (Hendrycks & Gimpel, 2016): x * Φ(x)
 * Used in GPT, BERT.
 */
export function gelu(x) {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}

/**
 * SiLU / Swish: x * σ(x) = x / (1 + e^{-x})
 * Used in LLaMA, Mistral.
 */
export function silu(x) { return x / (1 + Math.exp(-x)); }

/**
 * Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
 */
export function mish(x) { return x * Math.tanh(Math.log(1 + Math.exp(x))); }

/**
 * Softplus: ln(1 + e^x)
 */
export function softplus(x) { return Math.log(1 + Math.exp(x)); }

/**
 * ELU (Clevert et al., 2016): x if x>0, else α(e^x - 1)
 */
export function elu(x, alpha = 1.0) { return x > 0 ? x : alpha * (Math.exp(x) - 1); }

/**
 * SELU (Klambauer et al., 2017): self-normalizing
 */
export function selu(x) {
  const alpha = 1.6732632423543772;
  const scale = 1.0507009873554805;
  return scale * (x > 0 ? x : alpha * (Math.exp(x) - 1));
}

/**
 * Sigmoid: 1 / (1 + e^{-x})
 */
export function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

/**
 * Tanh
 */
export function tanh(x) { return Math.tanh(x); }

/**
 * Hard Swish (Howard et al., 2019): used in MobileNetV3
 * x * ReLU6(x + 3) / 6
 */
export function hardSwish(x) {
  return x * Math.max(0, Math.min(6, x + 3)) / 6;
}

/**
 * Hard Sigmoid: clamp((x+3)/6, 0, 1)
 */
export function hardSigmoid(x) {
  return Math.max(0, Math.min(1, (x + 3) / 6));
}

/**
 * Softmax (for arrays)
 */
export function softmax(arr) {
  const max = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b);
  return exp.map(x => x / sum);
}

/**
 * Get activation by name.
 */
export function getActivation(name) {
  const activations = {
    relu, leakyRelu, gelu, silu, mish, softplus, elu, selu,
    sigmoid, tanh, hardSwish, hardSigmoid,
  };
  return activations[name] || relu;
}
