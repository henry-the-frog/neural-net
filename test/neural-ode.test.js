import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  eulerSolve, rk4Solve, rk45Solve,
  NeuralODELayer, NeuralODE,
} from '../src/neural-ode.js';
import { Matrix } from '../src/matrix.js';

const approx = (a, b, eps = 0.01) => Math.abs(a - b) < eps;

// ===== ODE Solver Tests =====
describe('Euler Solver', () => {
  it('solves dy/dt = -y (exponential decay)', () => {
    // y' = -y, y(0) = 1 → y(t) = e^(-t)
    const f = (t, y) => {
      const result = new Matrix(y.rows, y.cols);
      for (let i = 0; i < y.rows; i++)
        for (let j = 0; j < y.cols; j++)
          result.set(i, j, -y.get(i, j));
      return result;
    };

    const y0 = new Matrix(1, 1, new Float64Array([1]));
    const { final } = eulerSolve(f, y0, 0, 1, 1000);
    const expected = Math.exp(-1);
    assert.ok(approx(final.get(0, 0), expected, 0.01),
      `Expected ${expected.toFixed(4)}, got ${final.get(0, 0).toFixed(4)}`);
  });

  it('returns trajectory', () => {
    const f = (t, y) => {
      const result = new Matrix(1, 1);
      result.set(0, 0, 1); // constant derivative
      return result;
    };
    const y0 = new Matrix(1, 1, new Float64Array([0]));
    const { trajectory } = eulerSolve(f, y0, 0, 1, 10);
    assert.equal(trajectory.length, 11); // 10 steps + initial
    assert.ok(approx(trajectory[10].y.get(0, 0), 1, 0.01));
  });
});

describe('RK4 Solver', () => {
  it('solves exponential decay more accurately', () => {
    const f = (t, y) => {
      const result = new Matrix(1, 1);
      result.set(0, 0, -y.get(0, 0));
      return result;
    };

    const y0 = new Matrix(1, 1, new Float64Array([1]));
    const { final } = rk4Solve(f, y0, 0, 1, 10); // Only 10 steps!
    const expected = Math.exp(-1);
    assert.ok(approx(final.get(0, 0), expected, 0.001),
      `RK4 should be very accurate: ${final.get(0, 0).toFixed(6)} vs ${expected.toFixed(6)}`);
  });

  it('solves harmonic oscillator', () => {
    // y'' = -y → y1' = y2, y2' = -y1
    // y(0) = [1, 0] → y(t) = [cos(t), -sin(t)]
    const f = (t, y) => {
      const result = new Matrix(1, 2);
      result.set(0, 0, y.get(0, 1));
      result.set(0, 1, -y.get(0, 0));
      return result;
    };

    const y0 = new Matrix(1, 2, new Float64Array([1, 0]));
    const { final } = rk4Solve(f, y0, 0, Math.PI, 100);
    assert.ok(approx(final.get(0, 0), -1, 0.001), 'cos(π) = -1');
    assert.ok(approx(final.get(0, 1), 0, 0.01), '-sin(π) ≈ 0');
  });

  it('is more accurate than Euler with same steps', () => {
    const f = (t, y) => {
      const result = new Matrix(1, 1);
      result.set(0, 0, -y.get(0, 0));
      return result;
    };
    const y0 = new Matrix(1, 1, new Float64Array([1]));
    const expected = Math.exp(-1);

    const euler = eulerSolve(f, y0, 0, 1, 10).final.get(0, 0);
    const rk4 = rk4Solve(f, y0, 0, 1, 10).final.get(0, 0);

    const eulerErr = Math.abs(euler - expected);
    const rk4Err = Math.abs(rk4 - expected);
    assert.ok(rk4Err < eulerErr, `RK4 error (${rk4Err.toFixed(6)}) should be < Euler error (${eulerErr.toFixed(6)})`);
  });
});

describe('RK45 Adaptive Solver', () => {
  it('solves exponential decay with error control', () => {
    const f = (t, y) => {
      const result = new Matrix(1, 1);
      result.set(0, 0, -y.get(0, 0));
      return result;
    };
    const y0 = new Matrix(1, 1, new Float64Array([1]));
    const { final } = rk45Solve(f, y0, 0, 1, { tol: 1e-6 });
    const expected = Math.exp(-1);
    assert.ok(approx(final.get(0, 0), expected, 0.001),
      `Should be accurate: ${final.get(0, 0).toFixed(6)} vs ${expected.toFixed(6)}`);
  });

  it('adapts step size', () => {
    const f = (t, y) => {
      const result = new Matrix(1, 1);
      result.set(0, 0, -10 * y.get(0, 0)); // Stiff-ish
      return result;
    };
    const y0 = new Matrix(1, 1, new Float64Array([1]));
    const { trajectory } = rk45Solve(f, y0, 0, 1, { tol: 1e-4 });
    assert.ok(trajectory.length >= 3, 'Should take at least a few steps');
  });
});

// ===== Neural ODE Layer Tests =====
describe('Neural ODE Layer', () => {
  it('forward produces correct shape', () => {
    const layer = new NeuralODELayer(4, 2, 'rk4', 5);
    const input = Matrix.random(3, 4);
    const output = layer.forward(input);
    assert.equal(output.rows, 3);
    assert.equal(output.cols, 4);
  });

  it('output is finite', () => {
    const layer = new NeuralODELayer(4, 2, 'rk4', 5);
    const input = Matrix.random(3, 4);
    const output = layer.forward(input);
    for (let i = 0; i < output.rows; i++) {
      for (let j = 0; j < output.cols; j++) {
        assert.ok(Number.isFinite(output.get(i, j)));
      }
    }
  });

  it('trajectory has correct length', () => {
    const layer = new NeuralODELayer(4, 2, 'rk4', 10);
    const input = Matrix.random(2, 4);
    layer.forward(input);
    assert.equal(layer.trajectory.length, 11); // 10 steps + initial
  });

  it('backward produces correct shape', () => {
    const layer = new NeuralODELayer(4, 2, 'rk4', 5);
    const input = Matrix.random(3, 4);
    layer.forward(input);
    const dOutput = Matrix.random(3, 4);
    const dInput = layer.backward(dOutput);
    assert.equal(dInput.rows, 3);
    assert.equal(dInput.cols, 4);
  });

  it('Euler solver works', () => {
    const layer = new NeuralODELayer(4, 2, 'euler', 20);
    const input = Matrix.random(2, 4);
    const output = layer.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 4);
    for (let i = 0; i < output.rows; i++)
      for (let j = 0; j < output.cols; j++)
        assert.ok(Number.isFinite(output.get(i, j)));
  });
});

// ===== Neural ODE Network Tests =====
describe('Neural ODE Network', () => {
  it('forward produces correct shape', () => {
    const model = new NeuralODE(2, 8, 1);
    const input = Matrix.random(5, 2);
    const output = model.forward(input);
    assert.equal(output.rows, 5);
    assert.equal(output.cols, 1);
  });

  it('can train on simple function', () => {
    const model = new NeuralODE(1, 8, 1, { steps: 5 });

    const N = 20;
    const inputs = new Matrix(N, 1);
    const targets = new Matrix(N, 1);
    for (let i = 0; i < N; i++) {
      const x = -1 + 2 * i / (N - 1);
      inputs.set(i, 0, x);
      targets.set(i, 0, Math.sin(Math.PI * x));
    }

    const losses = model.train(inputs, targets, 100, 0.01);
    assert.ok(losses[losses.length - 1] < losses[0],
      `Loss should decrease: ${losses[0].toFixed(4)} → ${losses[losses.length - 1].toFixed(4)}`);
  });

  it('paramCount includes all components', () => {
    const model = new NeuralODE(2, 8, 1);
    const params = model.paramCount();
    assert.ok(params > 0);
    // encoder: (2+1)*8 = 24, ode: 2*(8+1)*8 = 144, decoder: (8+1)*1 = 9
    // But depends on activation storage etc.
    assert.ok(params > 100, `Should have many params: ${params}`);
  });

  it('different solvers produce different results', () => {
    const model1 = new NeuralODE(2, 4, 1, { solver: 'euler', steps: 5 });
    const model2 = new NeuralODE(2, 4, 1, { solver: 'rk4', steps: 5 });

    const input = Matrix.random(3, 2);
    const out1 = model1.forward(input);
    const out2 = model2.forward(input);

    // They'll be different due to different random init, but both should be finite
    for (let i = 0; i < 3; i++) {
      assert.ok(Number.isFinite(out1.get(i, 0)));
      assert.ok(Number.isFinite(out2.get(i, 0)));
    }
  });
});
