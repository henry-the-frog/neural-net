// adamw.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { AdamW, SGDMomentum } from './adamw.js';
import { Matrix } from './matrix.js';

describe('AdamW Optimizer', () => {
  it('converges on simple quadratic', () => {
    // Minimize f(x) = x², gradient = 2x
    const opt = new AdamW({ lr: 0.1, weightDecay: 0 });
    const param = new Float64Array([5.0]);

    for (let i = 0; i < 100; i++) {
      const grad = new Float64Array([2 * param[0]]);
      opt.update('x', param, grad);
    }

    assert.ok(Math.abs(param[0]) < 0.1, `Should converge to 0: ${param[0]}`);
  });

  it('works with Matrix objects (regression: was silently no-op)', () => {
    const opt = new AdamW({ lr: 0.1, weightDecay: 0 });
    const param = new Matrix(2, 2, new Float64Array([5, -3, 2, -1]));
    const origValues = [...param.data];

    for (let i = 0; i < 50; i++) {
      // Gradient: 2 * param (minimize sum of squares)
      const grad = new Matrix(2, 2, new Float64Array(param.data.map(v => 2 * v)));
      opt.update('W', param, grad);
    }

    // All values should have moved toward 0
    for (let i = 0; i < 4; i++) {
      assert.ok(Math.abs(param.data[i]) < Math.abs(origValues[i]),
        `param[${i}] should decrease: ${origValues[i]} → ${param.data[i]}`);
    }
  });

  it('weight decay pushes params toward zero', () => {
    const opt = new AdamW({ lr: 0.1, weightDecay: 0.5 });
    const param = new Float64Array([10.0]);
    const grad = new Float64Array([0.01]); // tiny gradient to activate Adam updates

    for (let i = 0; i < 100; i++) {
      opt.update('x', param, grad);
    }

    assert.ok(Math.abs(param[0]) < 5, `Weight decay should shrink: ${param[0]}`);
  });

  it('bias correction helps early steps', () => {
    const opt = new AdamW({ lr: 0.1, beta1: 0.9, beta2: 0.999, weightDecay: 0 });
    const param = new Float64Array([5.0]);
    const grad = new Float64Array([1.0]); // constant gradient

    // First step should make a meaningful update despite small moment estimates
    opt.update('x', param, grad);
    const afterOne = param[0];
    assert.ok(afterOne < 5.0, 'Should decrease after one step');
    assert.ok(afterOne > 4.0, 'Should not overshoot');
  });

  it('handles multiple parameters independently', () => {
    const opt = new AdamW({ lr: 0.1, weightDecay: 0 });
    const p1 = new Float64Array([5.0]);
    const p2 = new Float64Array([-3.0]);

    for (let i = 0; i < 50; i++) {
      opt.update('p1', p1, new Float64Array([2 * p1[0]]));
      opt.update('p2', p2, new Float64Array([2 * p2[0]]));
    }

    assert.ok(Math.abs(p1[0]) < 0.5, `p1 should converge: ${p1[0]}`);
    assert.ok(Math.abs(p2[0]) < 0.5, `p2 should converge: ${p2[0]}`);
  });

  it('reset clears state', () => {
    const opt = new AdamW();
    opt.update('x', new Float64Array([1]), new Float64Array([1]));
    assert.equal(opt.step, 1);
    opt.reset();
    assert.equal(opt.step, 0);
    assert.equal(opt.states.size, 0);
  });
});

describe('SGD with Momentum', () => {
  it('converges with momentum', () => {
    const opt = new SGDMomentum({ lr: 0.01, momentum: 0.9 });
    const param = new Float64Array([5.0]);

    for (let i = 0; i < 200; i++) {
      opt.update('x', param, new Float64Array([2 * param[0]]));
    }

    assert.ok(Math.abs(param[0]) < 0.5, `Should converge: ${param[0]}`);
  });

  it('works with Matrix objects', () => {
    const opt = new SGDMomentum({ lr: 0.01, momentum: 0.9 });
    const param = new Matrix(1, 2, new Float64Array([5.0, -3.0]));

    for (let i = 0; i < 200; i++) {
      const grad = new Matrix(1, 2, new Float64Array(param.data.map(v => 2 * v)));
      opt.update('x', param, grad);
    }

    assert.ok(Math.abs(param.data[0]) < 0.5, `Should converge: ${param.data[0]}`);
    assert.ok(Math.abs(param.data[1]) < 0.5, `Should converge: ${param.data[1]}`);
  });
});
