import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  Variable, add, mul, sub, div, pow, neg,
  relu, sigmoid, tanh_ad, exp_ad, log_ad, sin_ad, cos_ad,
  constant, parameter, sum, mean, mseLoss,
} from '../src/autograd.js';

const approx = (a, b, eps = 0.001) => Math.abs(a - b) < eps;

describe('Basic Operations', () => {
  it('add computes correctly and backprops', () => {
    const a = parameter(3);
    const b = parameter(5);
    const c = add(a, b);
    assert.equal(c.value, 8);
    c.backward();
    assert.equal(a.grad, 1);
    assert.equal(b.grad, 1);
  });

  it('mul computes correctly and backprops', () => {
    const a = parameter(3);
    const b = parameter(4);
    const c = mul(a, b);
    assert.equal(c.value, 12);
    c.backward();
    assert.equal(a.grad, 4); // dc/da = b
    assert.equal(b.grad, 3); // dc/db = a
  });

  it('sub computes correctly', () => {
    const a = parameter(7);
    const b = parameter(3);
    const c = sub(a, b);
    assert.equal(c.value, 4);
    c.backward();
    assert.equal(a.grad, 1);
    assert.equal(b.grad, -1);
  });

  it('div computes correctly', () => {
    const a = parameter(6);
    const b = parameter(3);
    const c = div(a, b);
    assert.equal(c.value, 2);
    c.backward();
    assert.ok(approx(a.grad, 1 / 3)); // dc/da = 1/b
    assert.ok(approx(b.grad, -6 / 9)); // dc/db = -a/b²
  });

  it('pow computes correctly', () => {
    const a = parameter(3);
    const c = pow(a, 2);
    assert.equal(c.value, 9);
    c.backward();
    assert.equal(a.grad, 6); // d(x²)/dx = 2x
  });

  it('neg computes correctly', () => {
    const a = parameter(5);
    const c = neg(a);
    assert.equal(c.value, -5);
    c.backward();
    assert.equal(a.grad, -1);
  });
});

describe('Chain Rule', () => {
  it('f(x) = (x + 3) * 2', () => {
    const x = parameter(4);
    const three = constant(3);
    const sum_ = add(x, three);
    const two = constant(2);
    const result = mul(sum_, two);
    assert.equal(result.value, 14);
    result.backward();
    assert.equal(x.grad, 2); // df/dx = 2
  });

  it('f(x) = x² + 2x + 1', () => {
    const x = parameter(3);
    const x2 = pow(x, 2);
    const x2_ = mul(constant(2), x);
    const result = add(add(x2, x2_), constant(1));
    assert.equal(result.value, 16);
    result.backward();
    assert.equal(x.grad, 8); // df/dx = 2x + 2 = 8
  });

  it('f(x, y) = x*y + x²', () => {
    const x = parameter(2);
    const y = parameter(3);
    const xy = mul(x, y);
    const x2 = pow(x, 2);
    const result = add(xy, x2);
    assert.equal(result.value, 10);
    result.backward();
    assert.equal(x.grad, 7); // df/dx = y + 2x = 3 + 4 = 7
    assert.equal(y.grad, 2); // df/dy = x = 2
  });
});

describe('Activations', () => {
  it('relu positive', () => {
    const x = parameter(3);
    const r = relu(x);
    assert.equal(r.value, 3);
    r.backward();
    assert.equal(x.grad, 1);
  });

  it('relu negative', () => {
    const x = parameter(-2);
    const r = relu(x);
    assert.equal(r.value, 0);
    r.backward();
    assert.equal(x.grad, 0);
  });

  it('sigmoid', () => {
    const x = parameter(0);
    const s = sigmoid(x);
    assert.ok(approx(s.value, 0.5));
    s.backward();
    assert.ok(approx(x.grad, 0.25)); // σ'(0) = 0.25
  });

  it('tanh', () => {
    const x = parameter(0);
    const t = tanh_ad(x);
    assert.ok(approx(t.value, 0));
    t.backward();
    assert.ok(approx(x.grad, 1)); // tanh'(0) = 1
  });

  it('exp', () => {
    const x = parameter(1);
    const e = exp_ad(x);
    assert.ok(approx(e.value, Math.E));
    e.backward();
    assert.ok(approx(x.grad, Math.E));
  });

  it('log', () => {
    const x = parameter(Math.E);
    const l = log_ad(x);
    assert.ok(approx(l.value, 1));
    l.backward();
    assert.ok(approx(x.grad, 1 / Math.E));
  });

  it('sin', () => {
    const x = parameter(0);
    const s = sin_ad(x);
    assert.ok(approx(s.value, 0));
    s.backward();
    assert.ok(approx(x.grad, 1)); // cos(0) = 1
  });

  it('cos', () => {
    const x = parameter(0);
    const c = cos_ad(x);
    assert.ok(approx(c.value, 1));
    c.backward();
    assert.ok(approx(x.grad, 0)); // -sin(0) = 0
  });
});

describe('MSE Loss', () => {
  it('zero for perfect predictions', () => {
    const preds = [parameter(1), parameter(2), parameter(3)];
    const targets = [1, 2, 3];
    const loss = mseLoss(preds, targets);
    assert.ok(approx(loss.value, 0));
  });

  it('positive for imperfect predictions', () => {
    const preds = [parameter(1), parameter(2)];
    const targets = [2, 3]; // diff of 1 each → MSE = 1
    const loss = mseLoss(preds, targets);
    assert.ok(approx(loss.value, 1));
  });

  it('gradients point in correct direction', () => {
    const pred = parameter(3);
    const target = 1;
    const loss = mseLoss([pred], [target]);
    loss.backward();
    // d/dpred of (pred - 1)² = 2(pred - 1) = 4, divided by N=1 → 4
    assert.ok(approx(pred.grad, 4));
    assert.ok(pred.grad > 0, 'Gradient should push prediction down toward target');
  });
});

describe('Gradient Descent with Autograd', () => {
  it('optimizes x² to find minimum', () => {
    let x = parameter(5);
    const lr = 0.1;

    for (let step = 0; step < 50; step++) {
      x.grad = 0;
      const loss = pow(x, 2);
      loss.backward();
      x.value -= lr * x.grad;
    }

    assert.ok(approx(x.value, 0, 0.1), `Should converge to 0: ${x.value}`);
  });

  it('fits linear function y = 2x + 1', () => {
    const w = parameter(0);
    const b = parameter(0);
    const lr = 0.01;

    const data = [[1, 3], [2, 5], [3, 7], [4, 9]]; // y = 2x + 1

    for (let epoch = 0; epoch < 200; epoch++) {
      w.grad = 0;
      b.grad = 0;

      const preds = data.map(([x]) => add(mul(w, constant(x)), b));
      const targets = data.map(([, y]) => y);
      const loss = mseLoss(preds, targets);
      loss.backward();

      w.value -= lr * w.grad;
      b.value -= lr * b.grad;
    }

    assert.ok(approx(w.value, 2, 0.2), `w should be ~2: ${w.value.toFixed(2)}`);
    assert.ok(approx(b.value, 1, 0.5), `b should be ~1: ${b.value.toFixed(2)}`);
  });
});
