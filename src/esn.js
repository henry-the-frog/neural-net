// esn.js — Echo State Network (Reservoir Computing)
// Random fixed reservoir + trainable output layer via ridge regression
// Great for time-series, fast to train (no backprop through time)

import { Matrix } from './matrix.js';

// ===== Echo State Network =====
export class EchoStateNetwork {
  constructor(inputSize, reservoirSize, outputSize, {
    spectralRadius = 0.9,
    inputScaling = 0.5,
    leakingRate = 0.3,
    sparsity = 0.1,
    noise = 0.001,
    ridgeParam = 1e-6,
  } = {}) {
    this.inputSize = inputSize;
    this.reservoirSize = reservoirSize;
    this.outputSize = outputSize;
    this.leakingRate = leakingRate;
    this.noise = noise;
    this.ridgeParam = ridgeParam;

    // Input weights (random, fixed)
    this.Win = Matrix.random(reservoirSize, inputSize).mul(inputScaling);

    // Reservoir weights (sparse random, scaled to spectral radius)
    this.W = createSparseReservoir(reservoirSize, sparsity, spectralRadius);

    // Output weights (learned)
    this.Wout = null;

    // Reservoir state
    this.state = Matrix.zeros(1, reservoirSize);

    // Feedback weights (optional, not always used)
    this.Wfb = null;
  }

  // Reset reservoir state
  reset() {
    this.state = Matrix.zeros(1, this.state.cols);
  }

  // Update reservoir state with input
  step(input) {
    // input: [1, inputSize] Matrix
    const preActivation = input.dot(this.Win.T()).add(this.state.dot(this.W.T()));

    // Leaky integration
    const newState = new Matrix(1, this.reservoirSize);
    for (let j = 0; j < this.reservoirSize; j++) {
      const activated = Math.tanh(preActivation.get(0, j));
      newState.set(0, j,
        (1 - this.leakingRate) * this.state.get(0, j) + this.leakingRate * activated
      );
      // Add small noise for regularization
      if (this.noise > 0) {
        newState.set(0, j, newState.get(0, j) + (Math.random() * 2 - 1) * this.noise);
      }
    }

    this.state = newState;
    return this.state;
  }

  // Collect reservoir states for all inputs in a sequence
  collectStates(inputs, washout = 0) {
    this.reset();
    const states = [];

    for (let t = 0; t < inputs.length; t++) {
      const input = inputs[t] instanceof Matrix ? inputs[t] :
        new Matrix(1, this.inputSize, new Float64Array(inputs[t]));
      this.step(input);

      if (t >= washout) {
        // Extended state: [input, reservoir_state]
        const extended = new Array(this.inputSize + this.reservoirSize);
        for (let i = 0; i < this.inputSize; i++) {
          extended[i] = input instanceof Matrix ? input.get(0, i) : inputs[t][i];
        }
        for (let i = 0; i < this.reservoirSize; i++) {
          extended[this.inputSize + i] = this.state.get(0, i);
        }
        states.push(extended);
      }
    }

    return states;
  }

  // Train output weights via ridge regression
  // inputs: array of [inputSize] arrays
  // targets: array of [outputSize] arrays
  train(inputs, targets, washout = 100) {
    const states = this.collectStates(inputs, washout);
    const effectiveTargets = targets.slice(washout);

    const nSamples = states.length;
    const featSize = this.inputSize + this.reservoirSize;

    // Build matrices for ridge regression: Wout = (S^T S + λI)^(-1) S^T T
    // S: [nSamples, featSize], T: [nSamples, outputSize]
    const S = new Matrix(nSamples, featSize);
    const T = new Matrix(nSamples, this.outputSize);

    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < featSize; j++) S.set(i, j, states[i][j]);
      for (let j = 0; j < this.outputSize; j++) T.set(i, j, effectiveTargets[i][j]);
    }

    // Ridge regression: Wout = (S^T S + λI)^(-1) S^T T
    const StS = S.T().dot(S);
    // Add regularization
    for (let i = 0; i < featSize; i++) {
      StS.set(i, i, StS.get(i, i) + this.ridgeParam);
    }

    const StT = S.T().dot(T);

    // Solve via Cholesky or direct inversion (use simple Gauss elimination)
    this.Wout = solveLinearSystem(StS, StT);

    return this.Wout;
  }

  // Predict output for given input
  predict(input) {
    const inp = input instanceof Matrix ? input :
      new Matrix(1, this.inputSize, new Float64Array(input));
    this.step(inp);

    if (!this.Wout) throw new Error('ESN not trained');

    // Extended state
    const extended = new Matrix(1, this.inputSize + this.reservoirSize);
    for (let i = 0; i < this.inputSize; i++) {
      extended.set(0, i, inp instanceof Matrix ? inp.get(0, i) : input[i]);
    }
    for (let i = 0; i < this.reservoirSize; i++) {
      extended.set(0, this.inputSize + i, this.state.get(0, i));
    }

    return extended.dot(this.Wout);
  }

  // Predict sequence (autonomous mode — feed predictions back as input)
  predictSequence(seed, steps) {
    const outputs = [];
    let current = seed;

    for (let t = 0; t < steps; t++) {
      const output = this.predict(current);
      const out = [];
      for (let j = 0; j < this.outputSize; j++) out.push(output.get(0, j));
      outputs.push(out);
      current = out.slice(0, this.inputSize); // Use output as next input
    }

    return outputs;
  }

  paramCount() {
    const fixed = this.reservoirSize * this.inputSize + this.reservoirSize * this.reservoirSize;
    const trainable = this.Wout ? (this.inputSize + this.reservoirSize) * this.outputSize : 0;
    return { fixed, trainable, total: fixed + trainable };
  }
}

// ===== Sparse Reservoir Creation =====
function createSparseReservoir(size, sparsity, spectralRadius) {
  const W = new Matrix(size, size);

  // Create sparse random matrix
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      if (Math.random() < sparsity) {
        W.set(i, j, (Math.random() * 2 - 1));
      }
    }
  }

  // Scale to desired spectral radius (approximate with power iteration)
  const maxEig = estimateSpectralRadius(W, 50);
  if (maxEig > 0) {
    const scale = spectralRadius / maxEig;
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        W.set(i, j, W.get(i, j) * scale);
      }
    }
  }

  return W;
}

// Power iteration to estimate largest eigenvalue magnitude
function estimateSpectralRadius(W, iterations) {
  const n = W.rows;
  let v = new Matrix(n, 1);
  for (let i = 0; i < n; i++) v.set(i, 0, Math.random());

  let eigenvalue = 0;
  for (let iter = 0; iter < iterations; iter++) {
    const Wv = W.dot(v);
    let norm = 0;
    for (let i = 0; i < n; i++) norm += Wv.get(i, 0) ** 2;
    norm = Math.sqrt(norm);
    if (norm < 1e-10) return 0;

    eigenvalue = norm;
    for (let i = 0; i < n; i++) v.set(i, 0, Wv.get(i, 0) / norm);
  }

  return eigenvalue;
}

// Simple linear system solver (Gaussian elimination with partial pivoting)
function solveLinearSystem(A, B) {
  const n = A.rows;
  const m = B.cols;

  // Augmented matrix [A | B]
  const aug = new Matrix(n, n + m);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) aug.set(i, j, A.get(i, j));
    for (let j = 0; j < m; j++) aug.set(i, n + j, B.get(i, j));
  }

  // Forward elimination with partial pivoting
  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxVal = Math.abs(aug.get(col, col));
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug.get(row, col)) > maxVal) {
        maxVal = Math.abs(aug.get(row, col));
        maxRow = row;
      }
    }

    // Swap rows
    if (maxRow !== col) {
      for (let j = 0; j < n + m; j++) {
        const tmp = aug.get(col, j);
        aug.set(col, j, aug.get(maxRow, j));
        aug.set(maxRow, j, tmp);
      }
    }

    const pivot = aug.get(col, col);
    if (Math.abs(pivot) < 1e-12) continue;

    // Eliminate below
    for (let row = col + 1; row < n; row++) {
      const factor = aug.get(row, col) / pivot;
      for (let j = col; j < n + m; j++) {
        aug.set(row, j, aug.get(row, j) - factor * aug.get(col, j));
      }
    }
  }

  // Back substitution
  const X = new Matrix(n, m);
  for (let row = n - 1; row >= 0; row--) {
    const pivot = aug.get(row, row);
    if (Math.abs(pivot) < 1e-12) continue;

    for (let j = 0; j < m; j++) {
      let sum = aug.get(row, n + j);
      for (let k = row + 1; k < n; k++) {
        sum -= aug.get(row, k) * X.get(k, j);
      }
      X.set(row, j, sum / pivot);
    }
  }

  return X;
}

// ===== Liquid State Machine (Spiking Reservoir) =====
// Similar to ESN but uses spiking neurons in the reservoir
export class LiquidStateMachine {
  constructor(inputSize, reservoirSize, outputSize, { connectivity = 0.1 } = {}) {
    this.inputSize = inputSize;
    this.reservoirSize = reservoirSize;
    this.outputSize = outputSize;

    // Sparse connections
    this.connections = [];
    for (let i = 0; i < reservoirSize; i++) {
      this.connections[i] = [];
      for (let j = 0; j < reservoirSize; j++) {
        if (Math.random() < connectivity) {
          this.connections[i].push({ target: j, weight: (Math.random() * 2 - 1) * 0.5 });
        }
      }
    }

    // Neuron states (membrane potential)
    this.potentials = new Array(reservoirSize).fill(0);
    this.spikes = new Array(reservoirSize).fill(false);

    // Input weights
    this.inputWeights = Array.from({ length: reservoirSize }, () =>
      Array.from({ length: inputSize }, () => (Math.random() * 2 - 1) * 0.3)
    );

    this.Wout = null;
  }

  step(input) {
    const newPotentials = [...this.potentials];

    // Input current
    for (let i = 0; i < this.reservoirSize; i++) {
      let current = 0;
      for (let j = 0; j < this.inputSize; j++) {
        current += input[j] * this.inputWeights[i][j];
      }
      // Recurrent current
      for (const conn of this.connections[i]) {
        if (this.spikes[conn.target]) {
          current += conn.weight;
        }
      }
      // LIF dynamics
      newPotentials[i] = newPotentials[i] * 0.9 + current;
    }

    // Check for spikes
    this.spikes = newPotentials.map(p => {
      if (p > 1) return true;
      return false;
    });

    // Reset spiked neurons
    this.potentials = newPotentials.map((p, i) => this.spikes[i] ? 0 : p);

    // Return spike rates (low-pass filtered)
    return this.potentials;
  }

  reset() {
    this.potentials.fill(0);
    this.spikes.fill(false);
  }
}
