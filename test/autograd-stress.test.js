// autograd-stress.test.js — Deep stress tests for autograd: complex graphs, numerical gradient checks
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  Variable, add, mul, sub, div, pow, neg,
  relu, sigmoid, tanh_ad, exp_ad, log_ad, sin_ad, cos_ad,
  constant, parameter, sum, mean, mseLoss,
} from '../src/autograd.js';

// Numerical gradient check: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
function numericalGrad(buildGraph, param, h = 1e-5) {
  const orig = param.value;

  param.value = orig + h;
  const fPlus = buildGraph().value;

  param.value = orig - h;
  const fMinus = buildGraph().value;

  param.value = orig; // restore
  return (fPlus - fMinus) / (2 * h);
}

function checkGrad(buildGraph, param, eps = 0.01) {
  // Analytical
  const out = buildGraph();
  param.grad = 0;
  out.backward();
  const analytical = param.grad;

  // Numerical
  const numerical = numericalGrad(buildGraph, param);

  const diff = Math.abs(analytical - numerical);
  const scale = Math.max(Math.abs(analytical), Math.abs(numerical), 1e-8);
  assert.ok(
    diff / scale < eps,
    `Gradient mismatch: analytical=${analytical.toFixed(6)}, numerical=${numerical.toFixed(6)}, relDiff=${(diff / scale).toFixed(6)}`
  );
  return { analytical, numerical };
}

describe('Numerical Gradient Verification', () => {
  it('add', () => {
    const x = parameter(3.7);
    checkGrad(() => add(x, constant(2)), x);
  });

  it('mul', () => {
    const x = parameter(2.5);
    checkGrad(() => mul(x, constant(4.1)), x);
  });

  it('sub', () => {
    const x = parameter(1.2);
    checkGrad(() => sub(x, constant(0.8)), x);
  });

  it('div', () => {
    const x = parameter(5.0);
    checkGrad(() => div(x, constant(3.0)), x);
  });

  it('div denominator', () => {
    const y = parameter(3.0);
    checkGrad(() => div(constant(5.0), y), y);
  });

  it('pow(x, 3)', () => {
    const x = parameter(2.0);
    checkGrad(() => pow(x, 3), x);
  });

  it('pow(x, 0.5) — sqrt', () => {
    const x = parameter(4.0);
    checkGrad(() => pow(x, 0.5), x);
  });

  it('neg', () => {
    const x = parameter(3.5);
    checkGrad(() => neg(x), x);
  });

  it('relu positive region', () => {
    const x = parameter(2.0);
    checkGrad(() => relu(x), x);
  });

  it('sigmoid', () => {
    const x = parameter(1.5);
    checkGrad(() => sigmoid(x), x);
  });

  it('tanh', () => {
    const x = parameter(0.8);
    checkGrad(() => tanh_ad(x), x);
  });

  it('exp', () => {
    const x = parameter(1.5);
    checkGrad(() => exp_ad(x), x);
  });

  it('log', () => {
    const x = parameter(2.0);
    checkGrad(() => log_ad(x), x);
  });

  it('sin', () => {
    const x = parameter(1.0);
    checkGrad(() => sin_ad(x), x);
  });

  it('cos', () => {
    const x = parameter(1.0);
    checkGrad(() => cos_ad(x), x);
  });
});

describe('Complex Computation Graphs', () => {
  it('f(x) = sin(x²) + cos(2x)', () => {
    const x = parameter(1.5);
    checkGrad(() => {
      const x2 = pow(x, 2);
      const sinx2 = sin_ad(x2);
      const twox = mul(constant(2), x);
      const cos2x = cos_ad(twox);
      return add(sinx2, cos2x);
    }, x);
  });

  it('f(x) = sigmoid(tanh(x * 3 - 1))', () => {
    const x = parameter(0.5);
    checkGrad(() => {
      const scaled = mul(x, constant(3));
      const shifted = sub(scaled, constant(1));
      const t = tanh_ad(shifted);
      return sigmoid(t);
    }, x);
  });

  it('f(x) = log(exp(x) + 1) — softplus', () => {
    const x = parameter(2.0);
    checkGrad(() => {
      const ex = exp_ad(x);
      const s = add(ex, constant(1));
      return log_ad(s);
    }, x);
  });

  it('f(x) = x / (1 + x²) — Witch of Agnesi', () => {
    const x = parameter(1.5);
    checkGrad(() => {
      const x2 = pow(x, 2);
      const denom = add(constant(1), x2);
      return div(x, denom);
    }, x);
  });

  it('f(x, y) = sin(x*y) + x²*y³', () => {
    const x = parameter(1.0);
    const y = parameter(2.0);
    checkGrad(() => {
      const xy = mul(x, y);
      const sinxy = sin_ad(xy);
      const x2 = pow(x, 2);
      const y3 = pow(y, 3);
      const x2y3 = mul(x2, y3);
      return add(sinxy, x2y3);
    }, x);
    checkGrad(() => {
      const xy = mul(x, y);
      const sinxy = sin_ad(xy);
      const x2 = pow(x, 2);
      const y3 = pow(y, 3);
      const x2y3 = mul(x2, y3);
      return add(sinxy, x2y3);
    }, y);
  });

  it('deeply nested: sigmoid(sigmoid(sigmoid(x)))', () => {
    const x = parameter(0.5);
    checkGrad(() => sigmoid(sigmoid(sigmoid(x))), x);
  });

  it('diamond graph: f(x) = (x + x) * (x - x + 1)', () => {
    // x feeds into multiple paths that recombine
    const x = parameter(3.0);
    checkGrad(() => {
      const a = add(x, x);
      const b = add(sub(x, x), constant(1));
      return mul(a, b);
    }, x);
  });

  it('long chain: 10 sequential operations', () => {
    const x = parameter(0.5);
    checkGrad(() => {
      let v = x;
      v = add(v, constant(1));
      v = mul(v, constant(0.5));
      v = pow(v, 2);
      v = sub(v, constant(0.3));
      v = sigmoid(v);
      v = mul(v, constant(3));
      v = tanh_ad(v);
      v = add(v, constant(0.1));
      v = exp_ad(v);
      v = log_ad(v);
      return v;
    }, x);
  });

  it('multi-input sum: f(x1..x5) = Σ(xi²)', () => {
    const params = [1.0, 2.0, 3.0, 4.0, 5.0].map(v => parameter(v));
    for (const p of params) {
      checkGrad(() => {
        const squares = params.map(pi => pow(pi, 2));
        return sum(squares);
      }, p);
    }
  });

  it('neural network forward: 2-layer MLP', () => {
    // Simulate a tiny 2→2→1 network
    const x1 = parameter(0.5);
    const x2 = parameter(-0.3);
    const w11 = parameter(0.1);
    const w12 = parameter(0.2);
    const w21 = parameter(-0.1);
    const w22 = parameter(0.3);
    const w31 = parameter(0.5);
    const w32 = parameter(-0.4);
    const b1 = parameter(0.0);
    const b2 = parameter(0.0);
    const b3 = parameter(0.0);

    const buildMLP = () => {
      // Hidden layer
      const h1 = relu(add(add(mul(w11, x1), mul(w12, x2)), b1));
      const h2 = relu(add(add(mul(w21, x1), mul(w22, x2)), b2));
      // Output
      return sigmoid(add(add(mul(w31, h1), mul(w32, h2)), b3));
    };

    // Check gradient for all weights
    for (const p of [w11, w12, w21, w22, w31, w32, b1, b2, b3]) {
      checkGrad(buildMLP, p);
    }
  });
});

describe('Edge Cases', () => {
  it('gradient through relu at exactly zero', () => {
    const x = parameter(0);
    const r = relu(x);
    r.backward();
    // At 0, relu gradient is 0 (using strict > 0 check)
    assert.equal(x.grad, 0);
  });

  it('very large values: exp(20)', () => {
    const x = parameter(20);
    // exp(20) is large but finite
    const e = exp_ad(x);
    assert.ok(isFinite(e.value));
    e.backward();
    assert.ok(isFinite(x.grad));
    assert.ok(Math.abs(x.grad - e.value) < 1); // d/dx exp(x) = exp(x)
  });

  it('very small positive value for log', () => {
    const x = parameter(1e-10);
    const l = log_ad(x);
    assert.ok(isFinite(l.value));
    l.backward();
    assert.ok(isFinite(x.grad));
    assert.ok(x.grad > 0); // 1/x should be large positive
  });

  it('gradient accumulation: x used 3 times', () => {
    const x = parameter(2);
    // f(x) = x + x + x = 3x → df/dx = 3
    const result = add(add(x, x), x);
    result.backward();
    assert.equal(x.grad, 3);
  });

  it('gradient accumulation: x * x', () => {
    const x = parameter(3);
    // f(x) = x * x = x² → df/dx = 2x = 6
    const result = mul(x, x);
    result.backward();
    assert.equal(x.grad, 6);
  });

  it('zero gradient does not corrupt computation graph', () => {
    const x = parameter(5);
    const y = parameter(3);
    // f = x * y
    const f = mul(x, y);
    f.backward();
    assert.equal(x.grad, 3);
    assert.equal(y.grad, 5);

    // Zero and re-backward
    x.grad = 0;
    y.grad = 0;
    const f2 = mul(x, y);
    f2.backward();
    assert.equal(x.grad, 3);
    assert.equal(y.grad, 5);
  });

  it('constant has no effect on gradients', () => {
    const c = constant(42);
    const x = parameter(5);
    const f = add(x, c);
    f.backward();
    assert.equal(x.grad, 1);
    // constant's grad can be anything — it shouldn't matter
  });
});

describe('Gradient Descent Stress', () => {
  it('Rosenbrock function optimization', () => {
    // f(x,y) = (1-x)² + 100(y-x²)²
    // Minimum at (1, 1)
    const x = parameter(-1.0);
    const y = parameter(-1.0);
    const lr = 0.001;

    for (let step = 0; step < 2000; step++) {
      x.grad = 0;
      y.grad = 0;

      const term1 = pow(sub(constant(1), x), 2);
      const term2 = mul(constant(100), pow(sub(y, pow(x, 2)), 2));
      const loss = add(term1, term2);
      loss.backward();

      x.value -= lr * x.grad;
      y.value -= lr * y.grad;
    }

    // Should move toward (1, 1) — may not fully converge with basic SGD
    // but should be in the right neighborhood
    const finalLoss = Math.pow(1 - x.value, 2) + 100 * Math.pow(y.value - x.value ** 2, 2);
    assert.ok(finalLoss < 5, `Rosenbrock should decrease substantially: loss=${finalLoss.toFixed(4)}`);
  });

  it('fits quadratic: y = ax² + bx + c', () => {
    const a = parameter(0.0);
    const b = parameter(0.0);
    const c = parameter(0.0);
    const lr = 0.001;

    // Target: y = 2x² - 3x + 1
    const data = [-2, -1, 0, 1, 2, 3].map(x => [x, 2 * x * x - 3 * x + 1]);

    for (let epoch = 0; epoch < 500; epoch++) {
      a.grad = 0;
      b.grad = 0;
      c.grad = 0;

      const preds = data.map(([xv]) => {
        const xc = constant(xv);
        return add(add(mul(a, pow(xc, 2)), mul(b, xc)), c);
      });
      const targets = data.map(([, y]) => y);
      const loss = mseLoss(preds, targets);
      loss.backward();

      a.value -= lr * a.grad;
      b.value -= lr * b.grad;
      c.value -= lr * c.grad;
    }

    assert.ok(Math.abs(a.value - 2) < 0.5, `a should be ~2: ${a.value.toFixed(2)}`);
    assert.ok(Math.abs(b.value - (-3)) < 0.5, `b should be ~-3: ${b.value.toFixed(2)}`);
    assert.ok(Math.abs(c.value - 1) < 0.5, `c should be ~1: ${c.value.toFixed(2)}`);
  });
});
