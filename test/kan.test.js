import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { KANLayer, KAN, bsplineBasis, uniformKnots } from '../src/kan.js';
import { Matrix } from '../src/matrix.js';

const approx = (a, b, eps = 0.01) => Math.abs(a - b) < eps;

describe('B-Spline Basis', () => {
  it('partition of unity (sum = 1)', () => {
    const knots = uniformKnots(8, 4, -1, 1);
    for (const x of [-0.8, -0.3, 0, 0.3, 0.8]) {
      const basis = bsplineBasis(x, knots, 4);
      const sum = basis.reduce((a, b) => a + b, 0);
      assert.ok(approx(sum, 1, 0.01), `Basis should sum to 1 at x=${x}: ${sum}`);
    }
  });

  it('non-negative', () => {
    const knots = uniformKnots(8, 4, -1, 1);
    for (const x of [-0.9, -0.5, 0, 0.5, 0.9]) {
      const basis = bsplineBasis(x, knots, 4);
      assert.ok(basis.every(v => v >= -1e-10), `Basis should be non-negative at x=${x}`);
    }
  });

  it('compact support', () => {
    const knots = uniformKnots(8, 4, -1, 1);
    const basis = bsplineBasis(0, knots, 4);
    // Only some basis functions should be non-zero at any point
    const nonZero = basis.filter(v => v > 0.001).length;
    assert.ok(nonZero <= 4, `At most order(4) basis functions active: ${nonZero}`);
  });

  it('order 1 (piecewise constant)', () => {
    const knots = [0, 1, 2, 3];
    const basis = bsplineBasis(0.5, knots, 1);
    assert.ok(approx(basis[0], 1));
    assert.ok(approx(basis[1], 0));
    assert.ok(approx(basis[2], 0));
  });

  it('order 2 (piecewise linear)', () => {
    const knots = [0, 0, 1, 2, 2];
    const basis = bsplineBasis(0.5, knots, 2);
    assert.ok(approx(basis[0], 0.5));
    assert.ok(approx(basis[1], 0.5));
    assert.ok(approx(basis[2], 0));
  });
});

describe('Uniform Knots', () => {
  it('correct length', () => {
    const knots = uniformKnots(8, 4, -1, 1);
    // numBasis + order = 8 + 4 = 12
    assert.equal(knots.length, 12);
  });

  it('sorted', () => {
    const knots = uniformKnots(8, 4, -1, 1);
    for (let i = 1; i < knots.length; i++) {
      assert.ok(knots[i] >= knots[i - 1], `Knots should be sorted: ${knots[i-1]} > ${knots[i]}`);
    }
  });

  it('starts and ends at range', () => {
    const knots = uniformKnots(8, 4, -2, 3);
    assert.equal(knots[0], -2);
    assert.equal(knots[knots.length - 1], 3);
  });
});

describe('KAN Layer', () => {
  it('forward produces correct shape', () => {
    const layer = new KANLayer(3, 2, 8, 4);
    const input = Matrix.random(5, 3);
    const output = layer.forward(input);
    assert.equal(output.rows, 5);
    assert.equal(output.cols, 2);
  });

  it('output is finite', () => {
    const layer = new KANLayer(3, 2, 8, 4);
    const input = Matrix.random(5, 3);
    const output = layer.forward(input);
    for (let i = 0; i < output.rows; i++) {
      for (let j = 0; j < output.cols; j++) {
        assert.ok(Number.isFinite(output.get(i, j)));
      }
    }
  });

  it('backward produces correct shape', () => {
    const layer = new KANLayer(3, 2, 8, 4);
    const input = Matrix.random(5, 3);
    layer.forward(input);
    const dOutput = Matrix.random(5, 2);
    const dInput = layer.backward(dOutput);
    assert.equal(dInput.rows, 5);
    assert.equal(dInput.cols, 3);
  });

  it('backward gradients are finite', () => {
    const layer = new KANLayer(3, 2, 8, 4);
    const input = Matrix.random(3, 3);
    layer.forward(input);
    const dOutput = Matrix.random(3, 2);
    const dInput = layer.backward(dOutput);
    for (let i = 0; i < dInput.rows; i++) {
      for (let j = 0; j < dInput.cols; j++) {
        assert.ok(Number.isFinite(dInput.get(i, j)), `Non-finite grad at (${i},${j})`);
      }
    }
  });

  it('paramCount is correct', () => {
    const layer = new KANLayer(3, 2, 8, 4);
    // 3 * 2 * (8 + 1) = 54
    assert.equal(layer.paramCount(), 54);
  });

  it('getActivation returns points', () => {
    const layer = new KANLayer(3, 2, 8, 4);
    const points = layer.getActivation(0, 0, 50);
    assert.equal(points.length, 50);
    assert.ok(points[0].x <= points[49].x);
    assert.ok(points.every(p => Number.isFinite(p.y)));
  });
});

describe('KAN Network', () => {
  it('forward through multi-layer', () => {
    const kan = new KAN([3, 4, 2]);
    const input = Matrix.random(5, 3);
    const output = kan.forward(input);
    assert.equal(output.rows, 5);
    assert.equal(output.cols, 2);
  });

  it('can learn sine function', () => {
    const kan = new KAN([1, 4, 1], 8, 4);

    // Generate training data: y = sin(pi * x)
    const N = 50;
    const inputs = new Matrix(N, 1);
    const targets = new Matrix(N, 1);
    for (let i = 0; i < N; i++) {
      const x = -1 + 2 * i / (N - 1);
      inputs.set(i, 0, x);
      targets.set(i, 0, Math.sin(Math.PI * x));
    }

    const losses = kan.train(inputs, targets, 300, 0.01);
    assert.ok(losses[losses.length - 1] < losses[0],
      `Loss should decrease: ${losses[0].toFixed(4)} → ${losses[losses.length - 1].toFixed(4)}`);
  });

  it('can learn quadratic', () => {
    const kan = new KAN([1, 4, 1], 8, 4);

    const N = 40;
    const inputs = new Matrix(N, 1);
    const targets = new Matrix(N, 1);
    for (let i = 0; i < N; i++) {
      const x = -1 + 2 * i / (N - 1);
      inputs.set(i, 0, x);
      targets.set(i, 0, x * x);
    }

    const losses = kan.train(inputs, targets, 500, 0.01);
    assert.ok(losses[losses.length - 1] < losses[0],
      `Loss should decrease: ${losses[0].toFixed(4)} → ${losses[losses.length - 1].toFixed(4)}`);
  });

  it('can learn 2D function', () => {
    const kan = new KAN([2, 4, 1], 8, 4);

    const N = 40;
    const inputs = new Matrix(N, 2);
    const targets = new Matrix(N, 1);
    for (let i = 0; i < N; i++) {
      const x = -1 + 2 * Math.random();
      const y = -1 + 2 * Math.random();
      inputs.set(i, 0, x);
      inputs.set(i, 1, y);
      targets.set(i, 0, x + y); // Simple sum
    }

    const losses = kan.train(inputs, targets, 200, 0.01);
    assert.ok(losses[losses.length - 1] < losses[0],
      `Loss should decrease: ${losses[0].toFixed(4)} → ${losses[losses.length - 1].toFixed(4)}`);
  });

  it('paramCount sums across layers', () => {
    const kan = new KAN([3, 4, 2], 8, 4);
    const total = kan.paramCount();
    const expected = 3 * 4 * 9 + 4 * 2 * 9; // (inputSize * outputSize * (numBasis + 1)) per layer
    assert.equal(total, expected);
  });
});
