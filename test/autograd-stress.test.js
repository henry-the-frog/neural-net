// autograd-stress.test.js — Adversarial stress tests for autograd
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  Variable, add, mul, sub, div, pow, neg, relu, sigmoid, tanh_ad,
  exp_ad, log_ad, sin_ad, cos_ad, constant, parameter, sum, mean, mseLoss
} from '../src/autograd.js';

// Numerical gradient: f should be () => scalar, params is Float64Array or array
function numGrad(f, param, eps = 1e-6) {
  const orig = param.value;
  param.value = orig + eps;
  // Need to recompute — but autograd uses the param reference
  const fPlus = f().value;
  param.value = orig - eps;
  const fMinus = f().value;
  param.value = orig;
  return (fPlus - fMinus) / (2 * eps);
}

function relErr(analytical, numerical) {
  const denom = Math.max(Math.abs(analytical), Math.abs(numerical), 1e-8);
  return Math.abs(analytical - numerical) / denom;
}

describe('Autograd Stress — Numerical Gradient Verification', () => {
  it('deep chain: ((x^2 + 3x)^3 - 2)^2', () => {
    const x = parameter(2.5, 'x');
    const f = () => {
      const x2 = pow(x, 2);
      const x3 = mul(constant(3), x);
      const s = add(x2, x3);
      const s3 = pow(s, 3);
      const m = sub(s3, constant(2));
      return pow(m, 2);
    };
    const result = f();
    result.backward();
    const ng = numGrad(f, x);
    const err = relErr(x.grad, ng);
    assert.ok(err < 1e-4, `Deep chain gradient error: ${err.toExponential(2)} (analytical=${x.grad}, numerical=${ng})`);
  });

  it('diamond DAG: variable used in two paths', () => {
    const x = parameter(3.0, 'x');
    const f = () => {
      const a = pow(x, 2);     // x^2
      const b = mul(x, constant(5)); // 5x
      return add(a, b);        // x^2 + 5x, grad = 2x + 5 = 11
    };
    const result = f();
    result.backward();
    const ng = numGrad(f, x);
    const err = relErr(x.grad, ng);
    assert.ok(err < 1e-4, `Diamond DAG gradient error: ${err.toExponential(2)}`);
    // Check exact: d/dx(x^2 + 5x) at x=3 = 2(3) + 5 = 11
    assert.ok(Math.abs(x.grad - 11) < 1e-6, `Expected 11, got ${x.grad}`);
  });

  it('triple fan-out: x used three times', () => {
    const x = parameter(2.0, 'x');
    const f = () => {
      const a = pow(x, 3);     // x^3
      const b = pow(x, 2);     // x^2
      const c = mul(x, constant(7)); // 7x
      return add(add(a, b), c); // x^3 + x^2 + 7x, grad = 3x^2 + 2x + 7
    };
    const result = f();
    result.backward();
    const expected = 3 * 4 + 2 * 2 + 7; // 12 + 4 + 7 = 23
    assert.ok(Math.abs(x.grad - expected) < 1e-6, `Triple fan-out: expected ${expected}, got ${x.grad}`);
  });

  it('division by small number', () => {
    const x = parameter(1.0, 'x');
    const eps = parameter(0.001, 'eps');
    const f = () => div(x, eps);
    const result = f();
    result.backward();
    // d/dx(x/eps) = 1/eps = 1000
    assert.ok(Math.abs(x.grad - 1000) < 0.1, `div-by-small: x.grad=${x.grad}`);
    // d/deps(x/eps) = -x/eps^2 = -1/0.000001 = -1000000
    assert.ok(Math.abs(eps.grad - (-1000000)) < 100, `div-by-small: eps.grad=${eps.grad}`);
  });

  it('relu at zero: gradient should be 0', () => {
    const x = parameter(0.0, 'x');
    const result = relu(x);
    result.backward();
    assert.equal(x.grad, 0, 'ReLU at 0 should have gradient 0');
  });

  it('relu negative: gradient should be 0', () => {
    const x = parameter(-5.0, 'x');
    const result = relu(x);
    result.backward();
    assert.equal(x.grad, 0, 'ReLU at negative should have gradient 0');
  });

  it('relu positive: gradient should be 1', () => {
    const x = parameter(3.0, 'x');
    const result = relu(x);
    result.backward();
    assert.equal(x.grad, 1, 'ReLU at positive should have gradient 1');
  });

  it('sigmoid gradient matches numerical', () => {
    for (const val of [-10, -1, 0, 1, 10]) {
      const x = parameter(val, 'x');
      const f = () => sigmoid(x);
      const result = f();
      result.backward();
      const ng = numGrad(f, x);
      const err = relErr(x.grad, ng);
      assert.ok(err < 1e-3, `Sigmoid gradient at ${val}: err=${err.toExponential(2)}`);
    }
  });

  it('tanh gradient matches numerical', () => {
    for (const val of [-5, -1, 0, 1, 5]) {
      const x = parameter(val, 'x');
      const f = () => tanh_ad(x);
      const result = f();
      result.backward();
      const ng = numGrad(f, x);
      const err = relErr(x.grad, ng);
      assert.ok(err < 1e-3, `Tanh gradient at ${val}: err=${err.toExponential(2)}`);
    }
  });

  it('exp with large input', () => {
    const x = parameter(20.0, 'x');
    const result = exp_ad(x);
    result.backward();
    // d/dx(e^x) = e^x
    const expected = Math.exp(20);
    assert.ok(Math.abs(x.grad - expected) / expected < 1e-6, `exp grad at 20: ${x.grad} vs ${expected}`);
  });

  it('log with small input', () => {
    const x = parameter(0.001, 'x');
    const result = log_ad(x);
    result.backward();
    // d/dx(ln(x)) = 1/x = 1000
    assert.ok(Math.abs(x.grad - 1000) < 0.1, `log grad at 0.001: ${x.grad}`);
  });

  it('sin/cos composition: sin(cos(x))', () => {
    const x = parameter(1.5, 'x');
    const f = () => sin_ad(cos_ad(x));
    const result = f();
    result.backward();
    // d/dx(sin(cos(x))) = cos(cos(x)) * (-sin(x))
    const expected = Math.cos(Math.cos(1.5)) * (-Math.sin(1.5));
    assert.ok(Math.abs(x.grad - expected) < 1e-6, `sin(cos(x)) grad: ${x.grad} vs ${expected}`);
  });

  it('complex expression: sigmoid(x^2 + sin(x))', () => {
    const x = parameter(1.0, 'x');
    const f = () => sigmoid(add(pow(x, 2), sin_ad(x)));
    const result = f();
    result.backward();
    const ng = numGrad(f, x);
    const err = relErr(x.grad, ng);
    assert.ok(err < 1e-3, `Complex expression gradient error: ${err.toExponential(2)}`);
  });
});

describe('Autograd Stress — Edge Cases', () => {
  it('gradient accumulation: same variable used 10 times', () => {
    const x = parameter(1.0, 'x');
    // x + x + x + ... (10 times) = 10x, grad = 10
    let result = x;
    for (let i = 0; i < 9; i++) {
      result = add(result, x);
    }
    result.backward();
    assert.ok(Math.abs(x.grad - 10) < 1e-6, `10x grad should be 10, got ${x.grad}`);
  });

  it('zero gradient propagation', () => {
    const x = parameter(5.0, 'x');
    const y = parameter(0.0, 'y');
    // x * 0 = 0, grad w.r.t. x should be 0
    const result = mul(x, y);
    result.backward();
    assert.equal(x.grad, 0, `x*0 grad w.r.t. x should be 0`);
    assert.equal(y.grad, 5, `x*0 grad w.r.t. y should be x=5`);
  });

  it('deep chain: 20 nested additions', () => {
    const x = parameter(0.5, 'x');
    let result = x;
    for (let i = 0; i < 20; i++) {
      result = add(result, constant(0.1));
    }
    // result = x + 2.0, grad = 1
    result.backward();
    assert.ok(Math.abs(x.grad - 1) < 1e-6, `Deep add chain grad should be 1, got ${x.grad}`);
  });

  it('deep chain: 20 nested multiplications (vanishing gradient)', () => {
    const x = parameter(1.0, 'x');
    let result = x;
    for (let i = 0; i < 20; i++) {
      result = mul(result, constant(0.5));
    }
    // result = x * 0.5^20, grad = 0.5^20
    result.backward();
    const expected = Math.pow(0.5, 20);
    assert.ok(Math.abs(x.grad - expected) < 1e-10, `Deep mul chain: ${x.grad} vs ${expected}`);
  });

  it('deep chain: 20 nested multiplications (exploding gradient)', () => {
    const x = parameter(1.0, 'x');
    let result = x;
    for (let i = 0; i < 20; i++) {
      result = mul(result, constant(2.0));
    }
    // result = x * 2^20, grad = 2^20 = 1048576
    result.backward();
    const expected = Math.pow(2, 20);
    assert.ok(Math.abs(x.grad - expected) < 1, `Exploding: ${x.grad} vs ${expected}`);
  });

  it('pow(x, 0) should have gradient 0', () => {
    const x = parameter(5.0, 'x');
    const result = pow(x, 0);
    result.backward();
    // d/dx(x^0) = 0 * x^(-1) = 0
    assert.equal(x.grad, 0, `pow(x,0) grad should be 0, got ${x.grad}`);
  });

  it('pow(x, 1) should have gradient 1', () => {
    const x = parameter(5.0, 'x');
    const result = pow(x, 1);
    result.backward();
    assert.ok(Math.abs(x.grad - 1) < 1e-6, `pow(x,1) grad should be 1, got ${x.grad}`);
  });

  it('neg twice is identity gradient', () => {
    const x = parameter(3.0, 'x');
    const result = neg(neg(x));
    result.backward();
    assert.ok(Math.abs(x.grad - 1) < 1e-6, `neg(neg(x)) grad should be 1, got ${x.grad}`);
  });

  it('multiple backward calls accumulate gradients', () => {
    const x = parameter(2.0, 'x');
    const y = mul(x, x); // x^2
    y.backward();
    // First backward: grad = 2x = 4
    assert.ok(Math.abs(x.grad - 4) < 1e-6);
    // Second backward WITHOUT zeroing: should accumulate
    y.backward();
    assert.ok(Math.abs(x.grad - 8) < 1e-6, `Accumulated grad should be 8, got ${x.grad}`);
  });

  it('zeroGrad resets properly', () => {
    const x = parameter(2.0, 'x');
    const y = mul(x, x);
    y.backward();
    assert.ok(Math.abs(x.grad - 4) < 1e-6);
    x.zeroGrad();
    assert.equal(x.grad, 0, 'zeroGrad should reset to 0');
  });
});

describe('Autograd Stress — MSE Loss', () => {
  it('MSE loss gradient for simple regression', () => {
    const w = parameter(0.5, 'w');
    const b = parameter(0.1, 'b');
    // y = w*x + b, target = 3.0, x = 2.0
    // pred = 0.5*2 + 0.1 = 1.1
    // loss = (1.1 - 3)^2 = 3.61
    // d_loss/d_pred = 2*(1.1-3) = -3.8
    // d_pred/d_w = x = 2
    // d_loss/d_w = -3.8 * 2 = -7.6
    const f = () => {
      const pred = add(mul(w, constant(2.0)), b);
      return pow(sub(pred, constant(3.0)), 2);
    };
    const loss = f();
    loss.backward();
    const ngW = numGrad(f, w);
    const ngB = numGrad(f, b);
    assert.ok(relErr(w.grad, ngW) < 1e-4, `w grad: ${w.grad} vs numerical ${ngW}`);
    assert.ok(relErr(b.grad, ngB) < 1e-4, `b grad: ${b.grad} vs numerical ${ngB}`);
  });

  it('mseLoss with multiple predictions', () => {
    const w = parameter(1.0, 'w');
    const xs = [1, 2, 3, 4, 5];
    const ys = [2, 4, 6, 8, 10]; // y = 2x
    const f = () => {
      const preds = xs.map(x => mul(w, constant(x)));
      return mseLoss(preds, ys);
    };
    const loss = f();
    loss.backward();
    const ng = numGrad(f, w);
    const err = relErr(w.grad, ng);
    assert.ok(err < 1e-3, `MSE loss gradient error: ${err.toExponential(2)}`);
    // w=1 predicts [1,2,3,4,5], targets [2,4,6,8,10]
    // Gradient should push w toward 2 (negative gradient means increase w)
    assert.ok(w.grad < 0, `Gradient should be negative to increase w toward 2, got ${w.grad}`);
  });

  it('gradient descent converges', () => {
    const w = parameter(0.0, 'w');
    const b = parameter(0.0, 'b');
    const xs = [1, 2, 3, 4];
    const ys = [3, 5, 7, 9]; // y = 2x + 1

    for (let step = 0; step < 200; step++) {
      w.zeroGrad();
      b.zeroGrad();
      const preds = xs.map(x => add(mul(w, constant(x)), b));
      const loss = mseLoss(preds, ys);
      loss.backward();
      w.value -= 0.01 * w.grad;
      b.value -= 0.01 * b.grad;
    }

    assert.ok(Math.abs(w.value - 2) < 0.1, `w should converge to ~2, got ${w.value}`);
    assert.ok(Math.abs(b.value - 1) < 0.5, `b should converge to ~1, got ${b.value}`);
  });
});

describe('Autograd Stress — Adversarial', () => {
  it('NaN propagation: log(0)', () => {
    const x = parameter(0.0, 'x');
    const result = log_ad(x);
    // log(0) = -Infinity, grad = 1/0 = Infinity
    assert.equal(result.value, -Infinity);
    result.backward();
    assert.equal(x.grad, Infinity, 'log(0) gradient should be Infinity');
  });

  it('NaN propagation: 0/0', () => {
    const x = parameter(0.0, 'x');
    const y = parameter(0.0, 'y');
    const result = div(x, y);
    assert.ok(isNaN(result.value), '0/0 should be NaN');
  });

  it('very large values: exp(100)', () => {
    const x = parameter(100.0, 'x');
    const result = exp_ad(x);
    assert.ok(isFinite(result.value), `exp(100) should be finite: ${result.value}`);
    result.backward();
    assert.ok(isFinite(x.grad), `exp(100) grad should be finite: ${x.grad}`);
  });

  it('very negative sigmoid: sigmoid(-100)', () => {
    const x = parameter(-100, 'x');
    const result = sigmoid(x);
    assert.ok(result.value < 1e-40, `sigmoid(-100) should be ~0: ${result.value}`);
    result.backward();
    assert.ok(isFinite(x.grad), `sigmoid(-100) grad should be finite: ${x.grad}`);
  });

  it('deeply nested: 100 operations', () => {
    const x = parameter(0.01, 'x');
    const f = () => {
      let result = x;
      for (let i = 0; i < 50; i++) {
        result = add(result, constant(0.01));
        result = mul(result, constant(1.01));
      }
      return result;
    };
    const result = f();
    result.backward();
    const ng = numGrad(f, x);
    const err = relErr(x.grad, ng);
    assert.ok(err < 1e-3, `100-op chain gradient error: ${err.toExponential(2)}`);
  });
});
