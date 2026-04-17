// kan.js — Kolmogorov-Arnold Network
// Learnable B-spline activation functions on edges instead of nodes
// Based on KAN: Kolmogorov-Arnold Networks (Liu et al., 2024)

import { Matrix } from './matrix.js';

// ===== B-Spline Basis Functions =====

// Evaluate B-spline basis functions of given order at point x
// knots: array of knot positions (must be sorted)
// order: spline order (degree + 1), e.g., order=4 for cubic B-splines
function bsplineBasis(x, knots, order) {
  const n = knots.length - order; // number of basis functions
  const basis = new Array(n).fill(0);

  if (order === 1) {
    for (let i = 0; i < n; i++) {
      if (x >= knots[i] && x < knots[i + 1]) basis[i] = 1;
    }
    // Handle right endpoint: assign to last non-degenerate interval
    if (x >= knots[n]) {
      for (let i = n - 1; i >= 0; i--) {
        if (knots[i] < knots[i + 1]) {
          basis[i] = 1;
          break;
        }
      }
    }
    return basis;
  }

  const lower = bsplineBasis(x, knots, order - 1);
  const nLower = knots.length - (order - 1);

  for (let i = 0; i < n; i++) {
    let left = 0, right = 0;
    const denom1 = knots[i + order - 1] - knots[i];
    if (denom1 > 0 && i < nLower) {
      left = (x - knots[i]) / denom1 * lower[i];
    }
    const denom2 = knots[i + order] - knots[i + 1];
    if (denom2 > 0 && (i + 1) < nLower) {
      right = (knots[i + order] - x) / denom2 * lower[i + 1];
    }
    basis[i] = left + right;
  }

  return basis;
}

// Create uniform knot vector with augmented ends
function uniformKnots(numBasis, order, xMin = -1, xMax = 1) {
  const numInternalKnots = numBasis - order + 2;
  const step = (xMax - xMin) / (numInternalKnots - 1);
  const internal = Array.from({ length: numInternalKnots }, (_, i) => xMin + i * step);

  // Augment: repeat first and last knots (order-1) times
  const knots = [];
  for (let i = 0; i < order - 1; i++) knots.push(xMin);
  knots.push(...internal);
  for (let i = 0; i < order - 1; i++) knots.push(xMax);

  return knots;
}

// ===== KAN Layer =====
// Each edge (i,j) has a learnable B-spline function
export class KANLayer {
  constructor(inputSize, outputSize, numBasis = 8, splineOrder = 4, gridRange = [-1, 1]) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.numBasis = numBasis;
    this.splineOrder = splineOrder;
    this.gridRange = gridRange;

    // Create knot vector
    this.knots = uniformKnots(numBasis, splineOrder, gridRange[0], gridRange[1]);

    // Spline coefficients: one set per edge (inputSize * outputSize * numBasis)
    // coeffs[i][j] = array of numBasis weights for edge from input i to output j
    this.coeffs = [];
    for (let i = 0; i < inputSize; i++) {
      this.coeffs[i] = [];
      for (let j = 0; j < outputSize; j++) {
        // Initialize with small random values + residual (SiLU-like)
        this.coeffs[i][j] = Array.from({ length: numBasis },
          () => (Math.random() - 0.5) * 0.2);
      }
    }

    // Residual connection weight (linear component)
    this.residualWeights = [];
    for (let i = 0; i < inputSize; i++) {
      this.residualWeights[i] = [];
      for (let j = 0; j < outputSize; j++) {
        this.residualWeights[i][j] = (Math.random() - 0.5) * 0.5;
      }
    }

    // Cache
    this.input = null;
    this.basisValues = null; // [batchSize][inputSize] = array of basis values
    this.dCoeffs = null;
    this.dResidual = null;
  }

  forward(input) {
    this.input = input;
    const batchSize = input.rows;
    const output = Matrix.zeros(batchSize, this.outputSize);

    this.basisValues = [];

    for (let b = 0; b < batchSize; b++) {
      const batchBasis = [];
      for (let i = 0; i < this.inputSize; i++) {
        const x = input.get(b, i);
        // Clamp to grid range
        const xClamped = Math.max(this.gridRange[0], Math.min(this.gridRange[1], x));
        const basis = bsplineBasis(xClamped, this.knots, this.splineOrder);
        batchBasis.push(basis);

        for (let j = 0; j < this.outputSize; j++) {
          // Spline activation: sum of coeffs * basis
          let splineVal = 0;
          for (let k = 0; k < this.numBasis; k++) {
            splineVal += this.coeffs[i][j][k] * basis[k];
          }
          // Residual: linear component (like SiLU residual in original KAN)
          const residual = this.residualWeights[i][j] * x;
          output.set(b, j, output.get(b, j) + splineVal + residual);
        }
      }
      this.basisValues.push(batchBasis);
    }

    return output;
  }

  backward(dOutput) {
    const batchSize = dOutput.rows;
    const dInput = Matrix.zeros(batchSize, this.inputSize);

    // Initialize gradient accumulators
    this.dCoeffs = [];
    for (let i = 0; i < this.inputSize; i++) {
      this.dCoeffs[i] = [];
      for (let j = 0; j < this.outputSize; j++) {
        this.dCoeffs[i][j] = new Array(this.numBasis).fill(0);
      }
    }
    this.dResidual = [];
    for (let i = 0; i < this.inputSize; i++) {
      this.dResidual[i] = [];
      for (let j = 0; j < this.outputSize; j++) {
        this.dResidual[i][j] = 0;
      }
    }

    for (let b = 0; b < batchSize; b++) {
      for (let j = 0; j < this.outputSize; j++) {
        const dOut = dOutput.get(b, j);

        for (let i = 0; i < this.inputSize; i++) {
          const basis = this.basisValues[b][i];
          const x = this.input.get(b, i);

          // Gradient for spline coefficients
          for (let k = 0; k < this.numBasis; k++) {
            this.dCoeffs[i][j][k] += dOut * basis[k];
          }

          // Gradient for residual weights
          this.dResidual[i][j] += dOut * x;

          // Gradient for input (through both spline and residual)
          // dSpline/dx requires derivative of basis functions — approximate with finite difference
          // If input is outside the grid range, the spline output is constant (clamped),
          // so dSpline/dx = 0 — only the residual contributes.
          const xClamped = Math.max(this.gridRange[0], Math.min(this.gridRange[1], x));
          let dSplineDx = 0;
          if (x > this.gridRange[0] && x < this.gridRange[1]) {
            // Inside range: compute basis derivative via finite difference
            const fdEps = 1e-5;
            const basisPlus = bsplineBasis(
              Math.min(this.gridRange[1], xClamped + fdEps), this.knots, this.splineOrder);
            const basisMinus = bsplineBasis(
              Math.max(this.gridRange[0], xClamped - fdEps), this.knots, this.splineOrder);

            for (let k = 0; k < this.numBasis; k++) {
              dSplineDx += this.coeffs[i][j][k] * (basisPlus[k] - basisMinus[k]) / (2 * fdEps);
            }
          }
          // Outside range: dSplineDx stays 0 (spline is flat due to clamping)

          dInput.set(b, i, dInput.get(b, i) + dOut * (dSplineDx + this.residualWeights[i][j]));
        }
      }
    }

    return dInput;
  }

  update(learningRate) {
    if (!this.dCoeffs) return;
    const batchSize = this.input.rows;

    for (let i = 0; i < this.inputSize; i++) {
      for (let j = 0; j < this.outputSize; j++) {
        for (let k = 0; k < this.numBasis; k++) {
          this.coeffs[i][j][k] -= learningRate * this.dCoeffs[i][j][k] / batchSize;
        }
        this.residualWeights[i][j] -= learningRate * this.dResidual[i][j] / batchSize;
      }
    }
  }

  paramCount() {
    return this.inputSize * this.outputSize * (this.numBasis + 1); // +1 for residual
  }

  // Get the learned activation function for edge (i, j)
  getActivation(i, j, numPoints = 100) {
    const points = [];
    const step = (this.gridRange[1] - this.gridRange[0]) / (numPoints - 1);
    for (let p = 0; p < numPoints; p++) {
      const x = this.gridRange[0] + p * step;
      const basis = bsplineBasis(x, this.knots, this.splineOrder);
      let y = 0;
      for (let k = 0; k < this.numBasis; k++) {
        y += this.coeffs[i][j][k] * basis[k];
      }
      y += this.residualWeights[i][j] * x;
      points.push({ x, y });
    }
    return points;
  }
}

// ===== KAN Network (multi-layer) =====
export class KAN {
  constructor(layerSizes, numBasis = 8, splineOrder = 4) {
    this.layers = [];
    for (let l = 0; l < layerSizes.length - 1; l++) {
      this.layers.push(new KANLayer(layerSizes[l], layerSizes[l + 1], numBasis, splineOrder));
    }
  }

  forward(input) {
    let x = input;
    for (const layer of this.layers) {
      x = layer.forward(x);
    }
    return x;
  }

  backward(dOutput) {
    let dx = dOutput;
    for (let l = this.layers.length - 1; l >= 0; l--) {
      dx = this.layers[l].backward(dx);
    }
    return dx;
  }

  update(learningRate) {
    for (const layer of this.layers) layer.update(learningRate);
  }

  paramCount() {
    return this.layers.reduce((s, l) => s + l.paramCount(), 0);
  }

  // Train on data with MSE loss
  train(inputs, targets, epochs = 100, learningRate = 0.01) {
    const losses = [];
    for (let epoch = 0; epoch < epochs; epoch++) {
      const output = this.forward(inputs);

      // MSE loss
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
      losses.push(loss);

      this.backward(dOutput);
      this.update(learningRate);
    }
    return losses;
  }
}

// Export B-spline utilities for testing
export { bsplineBasis, uniformKnots };
