import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  numericalGradient, relativeError, checkDenseGradient,
  gradientCheck, gradientReport,
} from '../src/gradient-check.js';
import { Dense } from '../src/layer.js';
import { Matrix } from '../src/matrix.js';

describe('Numerical Gradient', () => {
  it('computes gradient of x^2', () => {
    const f = (x) => x[0] * x[0];
    const grad = numericalGradient(f, [3]);
    assert.ok(Math.abs(grad[0] - 6) < 0.01, `d/dx(x²) at x=3 should be 6: ${grad[0]}`);
  });

  it('computes gradient of sin(x)', () => {
    const f = (x) => Math.sin(x[0]);
    const grad = numericalGradient(f, [0]);
    assert.ok(Math.abs(grad[0] - 1) < 0.01, `d/dx(sin(x)) at x=0 should be 1: ${grad[0]}`);
  });

  it('computes multi-variable gradient', () => {
    const f = (x) => x[0] * x[1] + x[1] * x[1]; // df/dx0 = x1, df/dx1 = x0 + 2*x1
    const grad = numericalGradient(f, [2, 3]);
    assert.ok(Math.abs(grad[0] - 3) < 0.01, `df/dx0 at (2,3) should be 3: ${grad[0]}`);
    assert.ok(Math.abs(grad[1] - 8) < 0.01, `df/dx1 at (2,3) should be 8: ${grad[1]}`);
  });
});

describe('Relative Error', () => {
  it('zero for identical values', () => {
    assert.ok(relativeError(5, 5) < 1e-6);
  });

  it('small for close values', () => {
    assert.ok(relativeError(1.0, 1.001) < 0.01);
  });

  it('large for different values', () => {
    assert.ok(relativeError(1, 2) > 0.3);
  });
});

describe('Dense Gradient Check', () => {
  it('verifies Dense layer with linear activation', () => {
    const layer = new Dense(3, 2, 'linear');
    const input = Matrix.random(2, 3); // batch of 2
    const dOutput = Matrix.random(2, 2);

    const results = checkDenseGradient(layer, input, dOutput);
    assert.ok(results.maxError < 0.01,
      `Dense gradient error should be < 1%: ${results.maxError.toFixed(6)}`);
  });

  it('verifies Dense layer with tanh activation', () => {
    const layer = new Dense(4, 3, 'tanh');
    const input = Matrix.random(2, 4);
    const dOutput = Matrix.random(2, 3);

    const results = checkDenseGradient(layer, input, dOutput);
    assert.ok(results.maxError < 0.05,
      `Tanh gradient error should be small: ${results.maxError.toFixed(6)}`);
  });

  it('verifies Dense layer with relu activation', () => {
    const layer = new Dense(3, 2, 'relu');
    const input = Matrix.random(2, 3);
    const dOutput = Matrix.random(2, 2);

    const results = checkDenseGradient(layer, input, dOutput);
    // ReLU gradient can have issues at zero, allow slightly higher tolerance
    assert.ok(results.maxError < 0.1,
      `ReLU gradient error: ${results.maxError.toFixed(6)}`);
  });
});

describe('Generic Gradient Check', () => {
  it('verifies simple quadratic', () => {
    let params = [1, 2, 3];
    const result = gradientCheck(
      () => [...params],
      (p) => { params = p; },
      () => params[0] ** 2 + params[1] ** 2 + params[2] ** 2,
      [2 * params[0], 2 * params[1], 2 * params[2]],
    );
    assert.ok(result.passed, `Should pass: maxError=${result.maxError.toFixed(6)}`);
  });
});

describe('Gradient Report', () => {
  it('generates readable report', () => {
    const results = {
      maxError: 0.001,
      errors: [
        { index: 0, analytic: 1.0, numerical: 1.001, error: 0.001 },
        { index: 1, analytic: -0.5, numerical: -0.499, error: 0.002 },
      ],
    };
    const report = gradientReport(results);
    assert.ok(report.includes('✅ PASSED'));
    assert.ok(report.includes('maxError'));
  });

  it('shows warning for marginal results', () => {
    const report = gradientReport({ maxError: 0.05, errors: [] });
    assert.ok(report.includes('⚠️'));
  });

  it('shows failure for bad results', () => {
    const report = gradientReport({ maxError: 0.5, errors: [] });
    assert.ok(report.includes('❌'));
  });
});
