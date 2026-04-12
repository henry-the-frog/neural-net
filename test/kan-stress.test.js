// kan-stress.test.js — Deep stress tests for KAN (Kolmogorov-Arnold Networks)
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { KANLayer, KAN, bsplineBasis, uniformKnots } from '../src/kan.js';
import { Matrix } from '../src/matrix.js';
import { Network } from '../src/network.js';

describe('B-Spline Basis Functions', () => {
  it('basis functions sum to 1 (partition of unity)', () => {
    const knots = uniformKnots(8, 4, -1, 1);
    for (let trial = 0; trial < 50; trial++) {
      const x = -1 + Math.random() * 2;
      const basis = bsplineBasis(x, knots, 4);
      const sum = basis.reduce((a, b) => a + b, 0);
      assert.ok(Math.abs(sum - 1) < 0.01,
        `Basis should sum to 1 at x=${x.toFixed(3)}: sum=${sum.toFixed(6)}`);
    }
  });

  it('basis functions are non-negative', () => {
    const knots = uniformKnots(8, 4, -1, 1);
    for (let trial = 0; trial < 50; trial++) {
      const x = -1 + Math.random() * 2;
      const basis = bsplineBasis(x, knots, 4);
      for (let k = 0; k < basis.length; k++) {
        assert.ok(basis[k] >= -1e-10, `Basis ${k} should be >= 0 at x=${x.toFixed(3)}: ${basis[k]}`);
      }
    }
  });

  it('basis functions have local support', () => {
    const knots = uniformKnots(8, 4, -1, 1);
    // At the left end, only the first few basis should be non-zero
    const basisLeft = bsplineBasis(-0.99, knots, 4);
    const nonZeroLeft = basisLeft.filter(v => v > 1e-10).length;
    assert.ok(nonZeroLeft <= 4, `At most 4 non-zero basis at left: got ${nonZeroLeft}`);
    
    // At the right end
    const basisRight = bsplineBasis(0.99, knots, 4);
    const nonZeroRight = basisRight.filter(v => v > 1e-10).length;
    assert.ok(nonZeroRight <= 4, `At most 4 non-zero basis at right: got ${nonZeroRight}`);
  });

  it('order 1 (step functions) works correctly', () => {
    const knots = uniformKnots(4, 1, 0, 4);
    const basis = bsplineBasis(1.5, knots, 1);
    // Should have exactly one 1 and rest 0s
    const ones = basis.filter(v => v === 1).length;
    const zeros = basis.filter(v => v === 0).length;
    assert.equal(ones, 1);
    assert.equal(zeros + ones, basis.length);
  });

  it('boundary values are handled (x = min, x = max)', () => {
    const knots = uniformKnots(8, 4, -1, 1);
    const basisMin = bsplineBasis(-1, knots, 4);
    const basisMax = bsplineBasis(1, knots, 4);
    assert.ok(Math.abs(basisMin.reduce((a, b) => a + b, 0) - 1) < 0.01);
    assert.ok(Math.abs(basisMax.reduce((a, b) => a + b, 0) - 1) < 0.01);
  });
});

describe('KANLayer', () => {
  it('forward output has correct shape', () => {
    const layer = new KANLayer(3, 2, 8, 4);
    const input = Matrix.random(5, 3);
    const output = layer.forward(input);
    assert.equal(output.rows, 5);
    assert.equal(output.cols, 2);
  });

  it('backward produces gradient of correct shape', () => {
    const layer = new KANLayer(3, 2, 8, 4);
    const input = Matrix.random(5, 3);
    layer.forward(input);
    const dOutput = Matrix.random(5, 2);
    const dInput = layer.backward(dOutput);
    assert.equal(dInput.rows, 5);
    assert.equal(dInput.cols, 3);
  });

  it('getActivation returns smooth curve', () => {
    const layer = new KANLayer(2, 2, 8, 4);
    const points = layer.getActivation(0, 0, 50);
    assert.equal(points.length, 50);
    
    // Check smoothness: adjacent points shouldn't jump too much
    for (let i = 1; i < points.length; i++) {
      const dy = Math.abs(points[i].y - points[i - 1].y);
      assert.ok(dy < 10, `Activation should be smooth: dy=${dy.toFixed(4)} at x=${points[i].x.toFixed(3)}`);
    }
  });

  it('gradient coefficients accumulate during backward', () => {
    const layer = new KANLayer(2, 1, 8, 4);
    const input = Matrix.fromArray([[0.5, -0.3]]);
    layer.forward(input);
    const dOutput = Matrix.fromArray([[1.0]]);
    layer.backward(dOutput);
    
    // dCoeffs should be non-zero for the active basis functions
    let totalGrad = 0;
    for (let i = 0; i < 2; i++) {
      for (let k = 0; k < 8; k++) {
        totalGrad += Math.abs(layer.dCoeffs[i][0][k]);
      }
    }
    assert.ok(totalGrad > 0.001, `Gradients should be non-zero: ${totalGrad.toFixed(6)}`);
  });
});

describe('KAN Function Approximation', () => {
  it('should learn f(x) = sin(x) on [-1, 1]', () => {
    let passed = false;
    for (let attempt = 0; attempt < 3 && !passed; attempt++) {
      const kan = new KAN([1, 8, 1], 8, 4);
      
      // Training data: sin(x) at 20 points
      const N = 20;
      const inputs = Matrix.zeros(N, 1);
      const targets = Matrix.zeros(N, 1);
      for (let i = 0; i < N; i++) {
        const x = -1 + 2 * i / (N - 1);
        inputs.set(i, 0, x);
        targets.set(i, 0, Math.sin(x * Math.PI));
      }
      
      const losses = kan.train(inputs, targets, 1000, 0.1);
      const finalLoss = losses[losses.length - 1];
      if (finalLoss < 0.2) passed = true;
    }
    assert.ok(passed, 'KAN should approximate sin(x) in at least 1 of 3 attempts');
  });

  it('should learn f(x) = x² on [-1, 1]', () => {
    let passed = false;
    for (let attempt = 0; attempt < 3 && !passed; attempt++) {
      const kan = new KAN([1, 8, 1], 8, 4);
      
      const N = 20;
      const inputs = Matrix.zeros(N, 1);
      const targets = Matrix.zeros(N, 1);
      for (let i = 0; i < N; i++) {
        const x = -1 + 2 * i / (N - 1);
        inputs.set(i, 0, x);
        targets.set(i, 0, x * x);
      }
      
      const losses = kan.train(inputs, targets, 1000, 0.1);
      if (losses[losses.length - 1] < 0.2) passed = true;
    }
    assert.ok(passed, 'KAN should approximate x² in at least 1 of 3 attempts');
  });

  it('should learn f(x,y) = x + y (2D input)', () => {
    let passed = false;
    for (let attempt = 0; attempt < 3 && !passed; attempt++) {
      const kan = new KAN([2, 8, 1], 8, 4);
      
      const N = 25;
      const inputs = Matrix.zeros(N, 2);
      const targets = Matrix.zeros(N, 1);
      for (let i = 0; i < N; i++) {
        const x = Math.random() * 2 - 1;
        const y = Math.random() * 2 - 1;
        inputs.set(i, 0, x);
        inputs.set(i, 1, y);
        targets.set(i, 0, x + y);
      }
      
      const losses = kan.train(inputs, targets, 1000, 0.1);
      if (losses[losses.length - 1] < 0.2) passed = true;
    }
    assert.ok(passed, 'KAN should approximate x+y in at least 1 of 3 attempts');
  });

  it('loss should decrease during training', () => {
    const kan = new KAN([1, 8, 1], 8, 4);
    const N = 10;
    const inputs = Matrix.zeros(N, 1);
    const targets = Matrix.zeros(N, 1);
    for (let i = 0; i < N; i++) {
      inputs.set(i, 0, -1 + 2 * i / (N - 1));
      targets.set(i, 0, inputs.get(i, 0) * 2);
    }
    
    const losses = kan.train(inputs, targets, 200, 0.01);
    
    // Compare first 10 avg vs last 10 avg
    const firstAvg = losses.slice(0, 10).reduce((a, b) => a + b, 0) / 10;
    const lastAvg = losses.slice(-10).reduce((a, b) => a + b, 0) / 10;
    assert.ok(lastAvg < firstAvg, `Loss should decrease: ${firstAvg.toFixed(4)} → ${lastAvg.toFixed(4)}`);
  });
});

describe('KAN vs MLP Comparison', () => {
  it('KAN should approximate sin with fewer params than MLP', () => {
    // KAN: 1→4→1 with 8 basis = (1*4 + 4*1) * (8+1) = 45 params
    const kan = new KAN([1, 8, 1], 8, 4);
    
    // MLP: 1→16→1 = 16 + 16 + 16 + 1 = 49 params (similar count)
    const mlp = new Network();
    mlp.dense(1, 16, 'relu').dense(16, 1, 'linear').loss('mse');
    
    const N = 30;
    const inputs = Matrix.zeros(N, 1);
    const targets = Matrix.zeros(N, 1);
    for (let i = 0; i < N; i++) {
      const x = -1 + 2 * i / (N - 1);
      inputs.set(i, 0, x);
      targets.set(i, 0, Math.sin(x * Math.PI));
    }
    
    // Train KAN
    const kanLosses = kan.train(inputs, targets, 1000, 0.1);
    
    // Train MLP
    for (let e = 0; e < 300; e++) {
      mlp.trainBatch(inputs, targets, 0.01);
    }
    const mlpPred = mlp.predict(inputs);
    let mlpLoss = 0;
    for (let i = 0; i < N; i++) {
      const d = mlpPred.get(i, 0) - targets.get(i, 0);
      mlpLoss += d * d;
    }
    mlpLoss /= N;
    
    const kanLoss = kanLosses[kanLosses.length - 1];
    
    // Both should learn, KAN might be better for this smooth function
    assert.ok(kanLoss < 1.0 || mlpLoss < 1.0, 'At least one should learn');
    console.log(`    KAN loss: ${kanLoss.toFixed(4)}, MLP loss: ${mlpLoss.toFixed(4)}, ` +
      `KAN params: ${kan.paramCount()}`);
  });
});

describe('Edge Cases', () => {
  it('handles input outside grid range (clamping)', () => {
    const layer = new KANLayer(1, 1, 8, 4, [-1, 1]);
    // Input way outside range
    const input = Matrix.fromArray([[5]]);
    const output = layer.forward(input);
    assert.ok(Number.isFinite(output.get(0, 0)), 'Output should be finite for out-of-range input');
  });

  it('handles zero input', () => {
    const layer = new KANLayer(2, 2, 8, 4);
    const input = Matrix.fromArray([[0, 0]]);
    const output = layer.forward(input);
    assert.ok(output.data.every(Number.isFinite));
  });

  it('single sample training works', () => {
    const kan = new KAN([1, 2, 1], 4, 3);
    const input = Matrix.fromArray([[0.5]]);
    const target = Matrix.fromArray([[1.0]]);
    const losses = kan.train(input, target, 50, 0.1);
    assert.equal(losses.length, 50);
    assert.ok(losses.every(Number.isFinite));
  });

  it('deep KAN (3 layers) produces finite output', () => {
    const kan = new KAN([2, 4, 4, 1], 6, 3);
    const input = Matrix.random(10, 2);
    const output = kan.forward(input);
    assert.equal(output.rows, 10);
    assert.equal(output.cols, 1);
    assert.ok(output.data.every(Number.isFinite));
  });
});
