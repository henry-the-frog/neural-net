// optimizer-zoo.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { SGDMomentum, AdaGrad, RMSprop, Lion } from './optimizer-zoo.js';

describe('Optimizer Zoo', () => {
  // Simple quadratic: minimize f(x) = x²
  function testOptimizer(Opt, args, steps = 100) {
    const opt = new Opt(...args);
    const params = new Float64Array([5.0]); // Start at x=5
    for (let i = 0; i < steps; i++) {
      const grads = new Float64Array([2 * params[0]]); // df/dx = 2x
      opt.step(params, grads);
    }
    return params[0];
  }

  test('SGD with momentum converges to 0', () => {
    const result = testOptimizer(SGDMomentum, [0.01, 0.9]);
    assert.ok(Math.abs(result) < 0.1, `Should converge to 0, got ${result}`);
  });

  test('AdaGrad converges', () => {
    const result = testOptimizer(AdaGrad, [0.5]);
    assert.ok(Math.abs(result) < 1, `Should converge, got ${result}`);
  });

  test('RMSprop converges to 0', () => {
    const result = testOptimizer(RMSprop, [0.05, 0.99], 200);
    assert.ok(Math.abs(result) < 0.5, `Should converge, got ${result}`);
  });

  test('Lion converges (sign-based)', () => {
    const result = testOptimizer(Lion, [0.05, 0.9, 0.99, 0], 500);
    assert.ok(Math.abs(result) < 2, `Should converge, got ${result}`);
  });

  test('SGD momentum: faster than vanilla SGD', () => {
    const withMomentum = new SGDMomentum(0.01, 0.9);
    const without = new SGDMomentum(0.01, 0);
    
    const p1 = new Float64Array([5.0]);
    const p2 = new Float64Array([5.0]);
    
    for (let i = 0; i < 50; i++) {
      withMomentum.step(p1, new Float64Array([2 * p1[0]]));
      without.step(p2, new Float64Array([2 * p2[0]]));
    }
    
    assert.ok(Math.abs(p1[0]) < Math.abs(p2[0]), 
      `Momentum ${p1[0]} should be closer to 0 than without ${p2[0]}`);
  });
});
