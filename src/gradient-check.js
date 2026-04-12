// gradient-check.js — Numerical gradient verification utility
// Cross-checks analytic gradients against finite difference approximations

import { Matrix } from './matrix.js';

// ===== Numerical Gradient (Central Difference) =====
// df/dx ≈ (f(x + ε) - f(x - ε)) / (2ε)
export function numericalGradient(f, x, epsilon = 1e-5) {
  const grad = new Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const xPlus = [...x]; xPlus[i] += epsilon;
    const xMinus = [...x]; xMinus[i] -= epsilon;
    grad[i] = (f(xPlus) - f(xMinus)) / (2 * epsilon);
  }
  return grad;
}

// ===== Relative Error =====
export function relativeError(analytic, numerical) {
  const aAbs = Math.abs(analytic);
  const nAbs = Math.abs(numerical);
  const diff = Math.abs(analytic - numerical);
  return diff / (Math.max(aAbs, nAbs) + 1e-8);
}

// ===== Gradient Check for Dense Layer =====
export function checkDenseGradient(layer, input, dOutput, epsilon = 1e-5) {
  const results = { weight: [], bias: [], maxError: 0 };

  // Forward + backward to get analytic gradients
  layer.forward(input);
  layer.backward(dOutput);

  // Check weight gradients
  for (let i = 0; i < layer.weights.rows; i++) {
    for (let j = 0; j < layer.weights.cols; j++) {
      const original = layer.weights.get(i, j);

      // f(w + ε)
      layer.weights.set(i, j, original + epsilon);
      const outPlus = layer.forward(input);
      let lossPlus = 0;
      for (let r = 0; r < dOutput.rows; r++)
        for (let c = 0; c < dOutput.cols; c++)
          lossPlus += outPlus.get(r, c) * dOutput.get(r, c);

      // f(w - ε)
      layer.weights.set(i, j, original - epsilon);
      const outMinus = layer.forward(input);
      let lossMinus = 0;
      for (let r = 0; r < dOutput.rows; r++)
        for (let c = 0; c < dOutput.cols; c++)
          lossMinus += outMinus.get(r, c) * dOutput.get(r, c);

      layer.weights.set(i, j, original);

      const numGrad = (lossPlus - lossMinus) / (2 * epsilon);
      const analGrad = layer.dWeights.get(i, j);
      const err = relativeError(analGrad, numGrad);
      results.weight.push({ i, j, analytic: analGrad, numerical: numGrad, error: err });
      results.maxError = Math.max(results.maxError, err);
    }
  }

  // Check bias gradients
  for (let j = 0; j < layer.biases.cols; j++) {
    const original = layer.biases.get(0, j);

    layer.biases.set(0, j, original + epsilon);
    const outPlus = layer.forward(input);
    let lossPlus = 0;
    for (let r = 0; r < dOutput.rows; r++)
      for (let c = 0; c < dOutput.cols; c++)
        lossPlus += outPlus.get(r, c) * dOutput.get(r, c);

    layer.biases.set(0, j, original - epsilon);
    const outMinus = layer.forward(input);
    let lossMinus = 0;
    for (let r = 0; r < dOutput.rows; r++)
      for (let c = 0; c < dOutput.cols; c++)
        lossMinus += outMinus.get(r, c) * dOutput.get(r, c);

    layer.biases.set(0, j, original);

    const numGrad = (lossPlus - lossMinus) / (2 * epsilon);
    const analGrad = layer.dBiases.get(0, j);
    const err = relativeError(analGrad, numGrad);
    results.bias.push({ j, analytic: analGrad, numerical: numGrad, error: err });
    results.maxError = Math.max(results.maxError, err);
  }

  return results;
}

// ===== Generic Gradient Check =====
// For any function f(params) → scalar loss
// Checks if provided gradient matches numerical approximation
export function gradientCheck(getParams, setParams, forward, analyticGrad, epsilon = 1e-5) {
  const params = getParams();
  const errors = [];
  let maxError = 0;

  for (let i = 0; i < params.length; i++) {
    const original = params[i];

    // f(param + ε)
    params[i] = original + epsilon;
    setParams(params);
    const lossPlus = forward();

    // f(param - ε)
    params[i] = original - epsilon;
    setParams(params);
    const lossMinus = forward();

    // Restore
    params[i] = original;
    setParams(params);

    const numGrad = (lossPlus - lossMinus) / (2 * epsilon);
    const analGrad = analyticGrad[i];
    const err = relativeError(analGrad, numGrad);
    errors.push({ index: i, analytic: analGrad, numerical: numGrad, error: err });
    maxError = Math.max(maxError, err);
  }

  return { errors, maxError, passed: maxError < 1e-2 };
}

// ===== Report =====
export function gradientReport(results) {
  const lines = [];
  lines.push(`Gradient Check: maxError = ${results.maxError.toFixed(6)}`);
  lines.push(results.maxError < 1e-2 ? '✅ PASSED' :
             results.maxError < 0.1 ? '⚠️ WARNING (error > 1%)' : '❌ FAILED (error > 10%)');

  const worst = [...(results.errors || results.weight || [])]
    .sort((a, b) => b.error - a.error)
    .slice(0, 5);

  if (worst.length > 0) {
    lines.push('\nWorst errors:');
    for (const w of worst) {
      lines.push(`  [${w.index ?? `${w.i},${w.j}`}] analytic=${(w.analytic||0).toFixed(6)} ` +
        `numerical=${(w.numerical||0).toFixed(6)} error=${w.error.toFixed(6)}`);
    }
  }

  return lines.join('\n');
}
