// neural-ode.js — Neural Ordinary Differential Equations
// Continuous-depth neural networks with ODE solvers
// Based on "Neural Ordinary Differential Equations" (Chen et al., NeurIPS 2018)

import { Matrix } from './matrix.js';
import { Dense } from './layer.js';

// ===== ODE Solvers =====

// Euler method: y_{n+1} = y_n + h * f(t_n, y_n)
export function eulerSolve(f, y0, t0, t1, steps) {
  const h = (t1 - t0) / steps;
  let y = y0;
  let t = t0;
  const trajectory = [{ t, y: cloneMatrix(y) }];

  for (let i = 0; i < steps; i++) {
    const dy = f(t, y);
    y = matAdd(y, matScale(dy, h));
    t += h;
    trajectory.push({ t, y: cloneMatrix(y) });
  }

  return { final: y, trajectory };
}

// RK4 (4th-order Runge-Kutta): classical method
export function rk4Solve(f, y0, t0, t1, steps) {
  const h = (t1 - t0) / steps;
  let y = y0;
  let t = t0;
  const trajectory = [{ t, y: cloneMatrix(y) }];

  for (let i = 0; i < steps; i++) {
    const k1 = f(t, y);
    const k2 = f(t + h / 2, matAdd(y, matScale(k1, h / 2)));
    const k3 = f(t + h / 2, matAdd(y, matScale(k2, h / 2)));
    const k4 = f(t + h, matAdd(y, matScale(k3, h)));

    // y_{n+1} = y_n + (h/6)(k1 + 2*k2 + 2*k3 + k4)
    const update = matScale(
      matAdd(matAdd(k1, matScale(k2, 2)), matAdd(matScale(k3, 2), k4)),
      h / 6
    );
    y = matAdd(y, update);
    t += h;
    trajectory.push({ t, y: cloneMatrix(y) });
  }

  return { final: y, trajectory };
}

// Adaptive step solver using embedded RK methods (Bogacki-Shampine 2/3)
export function rk45Solve(f, y0, t0, t1, { tol = 1e-6, maxSteps = 1000, hInit = null } = {}) {
  let h = hInit || (t1 - t0) / 20;
  let y = y0;
  let t = t0;
  const trajectory = [{ t, y: cloneMatrix(y) }];
  let steps = 0;

  while (t < t1 - 1e-12 && steps < maxSteps) {
    h = Math.min(h, t1 - t);

    // Bogacki-Shampine (embedded RK 2/3)
    const k1 = f(t, y);
    const k2 = f(t + h / 2, matAdd(y, matScale(k1, h / 2)));
    const k3 = f(t + 3 * h / 4, matAdd(y, matScale(k2, 3 * h / 4)));

    // 3rd-order solution
    const y3 = matAdd(y, matScale(
      matAdd(matScale(k1, 2 / 9), matAdd(matScale(k2, 1 / 3), matScale(k3, 4 / 9))),
      h
    ));

    const k4 = f(t + h, y3);

    // 2nd-order solution (for error estimate)
    const y2 = matAdd(y, matScale(
      matAdd(matScale(k1, 7 / 24), matAdd(matScale(k2, 1 / 4),
        matAdd(matScale(k3, 1 / 3), matScale(k4, 1 / 8)))),
      h
    ));

    // Error estimate
    const err = matMaxAbs(matAdd(y3, matScale(y2, -1)));
    const errRatio = err / (tol + 1e-10);

    if (errRatio <= 1 || h < 1e-10) {
      // Accept step
      y = y3;
      t += h;
      trajectory.push({ t, y: cloneMatrix(y) });
      steps++;
    }

    // Adjust step size
    const safety = 0.8;
    const factor = Math.max(0.3, Math.min(3, safety * Math.pow(1 / (errRatio + 1e-10), 1 / 3)));
    h *= factor;
  }

  return { final: y, trajectory, steps };
}

// ===== Neural ODE Dynamics =====
// The "dynamics function" f(t, y) is a neural network

export class ODEFunc {
  constructor(hiddenSize, numLayers = 2) {
    this.hiddenSize = hiddenSize;
    this.layers = [];

    // Build a small MLP for the dynamics
    for (let i = 0; i < numLayers; i++) {
      this.layers.push(new Dense(hiddenSize, hiddenSize, i < numLayers - 1 ? 'tanh' : 'linear'));
    }
  }

  // Evaluate f(t, y) — the time derivative of the hidden state
  evaluate(t, y) {
    let x = y;
    for (const layer of this.layers) {
      x = layer.forward(x);
    }
    return x;
  }

  paramCount() {
    return this.layers.reduce((s, l) => s + l.paramCount(), 0);
  }
}

// ===== Neural ODE Layer =====
export class NeuralODELayer {
  constructor(hiddenSize, numLayers = 2, solver = 'rk4', steps = 10) {
    this.hiddenSize = hiddenSize;
    this.func = new ODEFunc(hiddenSize, numLayers);
    this.solver = solver;
    this.steps = steps;
    this.t0 = 0;
    this.t1 = 1;

    // Cache for backward
    this.input = null;
    this.output = null;
    this.trajectory = null;
  }

  forward(input) {
    this.input = input;
    const f = (t, y) => this.func.evaluate(t, y);

    let result;
    if (this.solver === 'euler') {
      result = eulerSolve(f, input, this.t0, this.t1, this.steps);
    } else if (this.solver === 'rk4') {
      result = rk4Solve(f, input, this.t0, this.t1, this.steps);
    } else {
      result = rk45Solve(f, input, this.t0, this.t1);
    }

    this.output = result.final;
    this.trajectory = result.trajectory;
    return this.output;
  }

  // Backward pass using the adjoint method
  // Instead of backpropagating through the solver steps,
  // we solve an augmented ODE backward in time
  backward(dOutput) {
    // Simplified adjoint: backprop through the discrete Euler steps
    // For Euler: y_{n+1} = y_n + h * f(t_n, y_n)
    // dy_loss/dy_n = dy_loss/dy_{n+1} + dy_loss/dy_{n+1} * h * df/dy_n
    // = adjoint + h * backprop(adjoint through f)
    const h = (this.t1 - this.t0) / this.steps;
    let adjoint = dOutput;

    // Walk backward through the trajectory
    for (let step = this.steps - 1; step >= 0; step--) {
      const y = this.trajectory[step].y;
      const t = this.trajectory[step].t;

      // Forward through dynamics at this point to set up layer caches
      this.func.evaluate(t, y);

      // Backprop adjoint through the dynamics network: compute df/dy * adjoint
      let dFunc = adjoint;
      for (let l = this.func.layers.length - 1; l >= 0; l--) {
        dFunc = this.func.layers[l].backward(dFunc);
      }

      // Update adjoint: adjoint_n = adjoint_{n+1} + h * (df/dy * adjoint_{n+1})
      adjoint = matAdd(adjoint, matScale(dFunc, h));
    }

    return adjoint; // Gradient w.r.t. initial state
  }

  update(learningRate) {
    for (const layer of this.func.layers) {
      if (layer.dWeights) layer.update(learningRate, 0, 'sgd');
    }
  }

  paramCount() {
    return this.func.paramCount();
  }
}

// ===== Neural ODE Network =====
// Combines encoder → ODE → decoder
export class NeuralODE {
  constructor(inputSize, hiddenSize, outputSize, { solver = 'rk4', steps = 10 } = {}) {
    this.encoder = new Dense(inputSize, hiddenSize, 'tanh');
    this.ode = new NeuralODELayer(hiddenSize, 2, solver, steps);
    this.decoder = new Dense(hiddenSize, outputSize, 'linear');
  }

  forward(input) {
    const h0 = this.encoder.forward(input);
    const h1 = this.ode.forward(h0);
    return this.decoder.forward(h1);
  }

  backward(dOutput) {
    const dH1 = this.decoder.backward(dOutput);
    const dH0 = this.ode.backward(dH1);
    return this.encoder.backward(dH0);
  }

  update(learningRate) {
    if (this.encoder.dWeights) this.encoder.update(learningRate, 0, 'sgd');
    this.ode.update(learningRate);
    if (this.decoder.dWeights) this.decoder.update(learningRate, 0, 'sgd');
  }

  train(inputs, targets, epochs = 100, learningRate = 0.01) {
    const losses = [];
    for (let epoch = 0; epoch < epochs; epoch++) {
      const output = this.forward(inputs);

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

  paramCount() {
    return this.encoder.paramCount() + this.ode.paramCount() + this.decoder.paramCount();
  }
}

// ===== Matrix Utilities =====

function cloneMatrix(m) {
  const result = new Matrix(m.rows, m.cols);
  for (let i = 0; i < m.rows; i++)
    for (let j = 0; j < m.cols; j++)
      result.set(i, j, m.get(i, j));
  return result;
}

function matAdd(a, b) {
  const result = new Matrix(a.rows, a.cols);
  for (let i = 0; i < a.rows; i++)
    for (let j = 0; j < a.cols; j++)
      result.set(i, j, a.get(i, j) + b.get(i, j));
  return result;
}

function matScale(m, s) {
  const result = new Matrix(m.rows, m.cols);
  for (let i = 0; i < m.rows; i++)
    for (let j = 0; j < m.cols; j++)
      result.set(i, j, m.get(i, j) * s);
  return result;
}

function matMaxAbs(m) {
  let max = 0;
  for (let i = 0; i < m.rows; i++)
    for (let j = 0; j < m.cols; j++)
      max = Math.max(max, Math.abs(m.get(i, j)));
  return max;
}
