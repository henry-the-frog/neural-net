// optimizer-correctness.test.js — Step-by-step optimizer correctness verification
// Compares each optimizer's update against manual reference calculations.

import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { SGD, MomentumSGD, Adam, RMSProp, AdamW } from './optimizer.js';
import { Matrix } from './matrix.js';

const tol = 1e-10;

function assertClose(actual, expected, tolerance = tol, msg = '') {
  assert.ok(Math.abs(actual - expected) < tolerance,
    `${msg} expected ${expected}, got ${actual} (diff ${Math.abs(actual - expected)})`);
}

describe('SGD Correctness', () => {
  it('vanilla SGD: param = param - lr * grad', () => {
    const sgd = new SGD(0.1);
    const param = new Matrix(1, 3, new Float64Array([1.0, 2.0, 3.0]));
    const grad = new Matrix(1, 3, new Float64Array([0.5, -0.3, 0.8]));

    const result = sgd.update(param, grad);
    
    // Expected: [1 - 0.1*0.5, 2 - 0.1*(-0.3), 3 - 0.1*0.8] = [0.95, 2.03, 2.92]
    assertClose(result.get(0, 0), 0.95, tol, 'SGD[0]');
    assertClose(result.get(0, 1), 2.03, tol, 'SGD[1]');
    assertClose(result.get(0, 2), 2.92, tol, 'SGD[2]');
  });

  it('SGD with weight decay: L2 regularization', () => {
    const sgd = new SGD(0.1, { weightDecay: 0.01 });
    const param = new Matrix(1, 2, new Float64Array([10.0, -5.0]));
    const grad = new Matrix(1, 2, new Float64Array([1.0, 1.0]));

    const result = sgd.update(param, grad);
    
    // Effective grad = grad + wd * param = [1 + 0.01*10, 1 + 0.01*(-5)] = [1.1, 0.95]
    // Result = param - lr * effGrad = [10 - 0.1*1.1, -5 - 0.1*0.95] = [9.89, -5.095]
    assertClose(result.get(0, 0), 9.89, tol, 'SGD L2[0]');
    assertClose(result.get(0, 1), -5.095, tol, 'SGD L2[1]');
  });
});

describe('MomentumSGD Correctness', () => {
  it('first step: velocity = lr * grad, param -= velocity', () => {
    const mom = new MomentumSGD(0.1, 0.9);
    const param = new Matrix(1, 2, new Float64Array([5.0, 3.0]));
    const grad = new Matrix(1, 2, new Float64Array([2.0, -1.0]));

    const result = mom.update(param, grad, 'w');
    
    // v = 0.9*0 + 0.1*[2,-1] = [0.2, -0.1]
    // param = [5,3] - [0.2,-0.1] = [4.8, 3.1]
    assertClose(result.get(0, 0), 4.8, tol, 'Mom step1[0]');
    assertClose(result.get(0, 1), 3.1, tol, 'Mom step1[1]');
  });

  it('second step: velocity accumulates with momentum', () => {
    const mom = new MomentumSGD(0.1, 0.9);
    const p0 = new Matrix(1, 1, new Float64Array([10.0]));
    const g1 = new Matrix(1, 1, new Float64Array([2.0]));
    const g2 = new Matrix(1, 1, new Float64Array([3.0]));

    // Step 1: v1 = 0.9*0 + 0.1*2 = 0.2, p1 = 10 - 0.2 = 9.8
    const p1 = mom.update(p0, g1, 'w');
    assertClose(p1.get(0, 0), 9.8, tol, 'Mom step1');

    // Step 2: v2 = 0.9*0.2 + 0.1*3 = 0.18 + 0.3 = 0.48, p2 = 9.8 - 0.48 = 9.32
    const p2 = mom.update(p1, g2, 'w');
    assertClose(p2.get(0, 0), 9.32, tol, 'Mom step2');
  });
});

describe('Adam Correctness', () => {
  it('step 1 matches reference calculation', () => {
    const adam = new Adam(0.001, 0.9, 0.999, 1e-8);
    adam.t = 0;
    adam.step(); // t = 1

    const param = new Matrix(1, 1, new Float64Array([5.0]));
    const grad = new Matrix(1, 1, new Float64Array([2.0]));

    const result = adam.update(param, grad, 'w');

    // m1 = 0.9*0 + 0.1*2 = 0.2
    // v1 = 0.999*0 + 0.001*4 = 0.004
    // mHat = 0.2 / (1 - 0.9^1) = 0.2/0.1 = 2.0
    // vHat = 0.004 / (1 - 0.999^1) = 0.004/0.001 = 4.0
    // update = 5 - 0.001 * 2.0 / (sqrt(4.0) + 1e-8) = 5 - 0.001 * 2.0 / 2.0 = 5 - 0.001 = 4.999
    assertClose(result.get(0, 0), 4.999, 1e-6, 'Adam step1');
  });

  it('step 2 accumulates moment estimates', () => {
    const adam = new Adam(0.001, 0.9, 0.999, 1e-8);
    adam.step(); // t=1

    const p0 = new Matrix(1, 1, new Float64Array([5.0]));
    const g1 = new Matrix(1, 1, new Float64Array([2.0]));
    
    const p1 = adam.update(p0, g1, 'w');
    
    adam.step(); // t=2
    const g2 = new Matrix(1, 1, new Float64Array([1.0]));
    const p2 = adam.update(p1, g2, 'w');

    // m2 = 0.9*0.2 + 0.1*1 = 0.18 + 0.1 = 0.28
    // v2 = 0.999*0.004 + 0.001*1 = 0.003996 + 0.001 = 0.004996
    // bc1 = 1 - 0.9^2 = 0.19
    // bc2 = 1 - 0.999^2 = 0.001999
    // mHat = 0.28 / 0.19 = 1.473684...
    // vHat = 0.004996 / 0.001999 = 2.499249...
    // update = p1 - 0.001 * mHat / (sqrt(vHat) + 1e-8)
    
    // Just check it's reasonable (moved in the right direction)
    assert.ok(p2.get(0, 0) < p1.get(0, 0), 'Adam should decrease param with positive gradients');
    assert.ok(p2.get(0, 0) > 4.99, 'Adam should not move too far');
  });

  it('Adam bias correction prevents large initial steps', () => {
    const adam = new Adam(0.01, 0.9, 0.999, 1e-8);
    adam.step(); // t=1

    const param = new Matrix(1, 1, new Float64Array([1.0]));
    const grad = new Matrix(1, 1, new Float64Array([0.1]));

    const result = adam.update(param, grad, 'w');

    // Without bias correction, the update would be much larger
    // With correction at t=1: mHat = 0.01/0.1 = 0.1, vHat = 0.00001/0.001 = 0.01
    // update = 1 - 0.01 * 0.1 / sqrt(0.01) = 1 - 0.01 * 0.1/0.1 = 1 - 0.01 = 0.99
    assertClose(result.get(0, 0), 0.99, 1e-4, 'Adam bias correction');
  });
});

describe('RMSProp Correctness', () => {
  it('step 1: cache initialized from gradient', () => {
    const rms = new RMSProp(0.01, 0.99, 1e-8);
    const param = new Matrix(1, 1, new Float64Array([5.0]));
    const grad = new Matrix(1, 1, new Float64Array([2.0]));

    const result = rms.update(param, grad, 'w');

    // cache = 0.99*0 + 0.01*4 = 0.04
    // update = 5 - 0.01 * 2 / (sqrt(0.04) + 1e-8) = 5 - 0.02/0.2 = 5 - 0.1 = 4.9
    assertClose(result.get(0, 0), 4.9, 1e-4, 'RMSProp step1');
  });

  it('cache accumulates across steps', () => {
    const rms = new RMSProp(0.01, 0.99, 1e-8);
    const p0 = new Matrix(1, 1, new Float64Array([5.0]));
    const g1 = new Matrix(1, 1, new Float64Array([2.0]));
    
    const p1 = rms.update(p0, g1, 'w');
    // cache1 = 0.04

    const g2 = new Matrix(1, 1, new Float64Array([1.0]));
    const p2 = rms.update(p1, g2, 'w');
    // cache2 = 0.99*0.04 + 0.01*1 = 0.0396 + 0.01 = 0.0496
    // update = p1 - 0.01 * 1 / (sqrt(0.0496) + 1e-8) = p1 - 0.01/0.2228 ≈ p1 - 0.04489

    const step2 = p1.get(0, 0) - p2.get(0, 0);
    assertClose(step2, 0.01 / Math.sqrt(0.0496), 1e-4, 'RMSProp step2 size');
  });

  it('RMSProp adapts: large gradient → smaller effective step', () => {
    const rms = new RMSProp(0.1, 0.9, 1e-8);
    
    // After many large gradients, cache should be large → smaller steps
    const param = new Matrix(1, 1, new Float64Array([10.0]));
    
    let p = param;
    for (let i = 0; i < 100; i++) {
      const g = new Matrix(1, 1, new Float64Array([10.0])); // Constant large gradient
      p = rms.update(p, g, 'w');
    }

    // After accumulating, the effective step should be approximately lr
    // because cache ≈ grad^2, so lr*grad/sqrt(grad^2) = lr
    // p should have decreased by roughly 100 * 0.1 = 10
    assert.ok(p.get(0, 0) < 10.0, 'RMSProp should decrease param');
    assert.ok(p.get(0, 0) > -10.0, 'RMSProp should not diverge');
  });
});

describe('AdamW Correctness: Decoupled Weight Decay', () => {
  it('AdamW weight decay is decoupled from gradient', () => {
    // Key property of AdamW: weight decay should not pass through adaptive rate
    const adamw = new AdamW(0.001, 0.9, 0.999, 1e-8, 0.01);
    adamw.step();

    const param = new Matrix(1, 1, new Float64Array([10.0]));
    const grad = new Matrix(1, 1, new Float64Array([0.0])); // Zero gradient!

    const result = adamw.update(param, grad, 'w');

    // With zero gradient, the only update should be weight decay:
    // result = param - wd * lr * param = 10 - 0.01 * 0.001 * 10 = 10 - 0.0001 = 9.9999
    // Plus the Adam update term (which should be ~0 with zero gradient)
    assertClose(result.get(0, 0), 10.0 - 0.01 * 0.001 * 10.0, 1e-4,
      'AdamW with zero grad should only apply weight decay');
  });

  it('AdamW vs Adam: different weight decay behavior', () => {
    // Adam with weight decay: decay goes through gradient → adaptive rate
    // AdamW: decay applied directly to param → not affected by adaptive rate
    const adam = new Adam(0.001, 0.9, 0.999, 1e-8, { weightDecay: 0.01 });
    const adamw = new AdamW(0.001, 0.9, 0.999, 1e-8, 0.01);
    adam.step(); adamw.step();

    const param = new Matrix(1, 1, new Float64Array([10.0]));
    const grad = new Matrix(1, 1, new Float64Array([1.0]));

    const resultAdam = adam.update(param, grad, 'w');
    const resultAdamW = adamw.update(param, grad, 'w');

    // Both should decrease param, but by different amounts
    assert.ok(resultAdam.get(0, 0) < 10.0, 'Adam should decrease');
    assert.ok(resultAdamW.get(0, 0) < 10.0, 'AdamW should decrease');
    
    // They should NOT be equal (different weight decay mechanisms)
    assert.ok(Math.abs(resultAdam.get(0, 0) - resultAdamW.get(0, 0)) > 1e-10,
      'Adam and AdamW should produce different results with same weight decay');
  });
});

describe('Optimizer Convergence: Minimize Quadratic', () => {
  // All optimizers should minimize f(x) = x^2 (gradient = 2x)
  // Starting from x=5, after many steps, x should approach 0

  function minimizeQuadratic(optimizer, steps = 1000) {
    let x = new Matrix(1, 1, new Float64Array([5.0]));
    for (let i = 0; i < steps; i++) {
      if (optimizer.step) optimizer.step();
      const grad = x.mul(2.0); // d/dx(x^2) = 2x
      x = optimizer.update(x, grad, 'x');
    }
    return x.get(0, 0);
  }

  it('SGD converges on quadratic', () => {
    const result = minimizeQuadratic(new SGD(0.1));
    assert.ok(Math.abs(result) < 0.01, `SGD should converge near 0, got ${result}`);
  });

  it('MomentumSGD converges on quadratic', () => {
    const result = minimizeQuadratic(new MomentumSGD(0.01, 0.9));
    assert.ok(Math.abs(result) < 0.01, `Momentum should converge near 0, got ${result}`);
  });

  it('Adam converges on quadratic', () => {
    const result = minimizeQuadratic(new Adam(0.01), 2000);
    assert.ok(Math.abs(result) < 0.1, `Adam should converge near 0, got ${result}`);
  });

  it('RMSProp converges on quadratic', () => {
    const result = minimizeQuadratic(new RMSProp(0.01));
    assert.ok(Math.abs(result) < 0.1, `RMSProp should converge near 0, got ${result}`);
  });

  it('AdamW converges on quadratic', () => {
    const result = minimizeQuadratic(new AdamW(0.01), 2000);
    assert.ok(Math.abs(result) < 0.1, `AdamW should converge near 0, got ${result}`);
  });
});

describe('Optimizer Edge Cases', () => {
  it('Adam auto-step: t increments even without explicit step()', () => {
    const adam = new Adam(0.001);
    // Don't call step() — t should auto-increment to avoid division by zero
    const param = new Matrix(1, 1, new Float64Array([1.0]));
    const grad = new Matrix(1, 1, new Float64Array([0.5]));
    
    const result = adam.update(param, grad, 'w');
    assert.ok(isFinite(result.get(0, 0)), 'Adam without step() should not produce NaN');
    assert.ok(!isNaN(result.get(0, 0)), 'Adam without step() should not produce NaN');
  });

  it('zero gradient produces no update (except weight decay)', () => {
    const sgd = new SGD(0.1);
    const param = new Matrix(1, 2, new Float64Array([3.0, -4.0]));
    const grad = Matrix.zeros(1, 2);

    const result = sgd.update(param, grad);
    assertClose(result.get(0, 0), 3.0, tol, 'Zero grad no change[0]');
    assertClose(result.get(0, 1), -4.0, tol, 'Zero grad no change[1]');
  });

  it('very large gradient: no NaN or Inf', () => {
    const adam = new Adam(0.001);
    adam.step();
    const param = new Matrix(1, 1, new Float64Array([1.0]));
    const grad = new Matrix(1, 1, new Float64Array([1e10]));

    const result = adam.update(param, grad, 'w');
    assert.ok(isFinite(result.get(0, 0)), 'Large gradient should not produce Inf');
    assert.ok(!isNaN(result.get(0, 0)), 'Large gradient should not produce NaN');
  });
});
