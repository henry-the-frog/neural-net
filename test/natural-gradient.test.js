import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { NaturalGradient, KFAC, createOptimizer } from '../src/optimizer.js';
import { Matrix } from '../src/matrix.js';

describe('NaturalGradient Optimizer', () => {
  it('should create with default parameters', () => {
    const opt = new NaturalGradient();
    assert.equal(opt.lr, 0.01);
    assert.equal(opt.damping, 0.01);
    assert.equal(opt.decay, 0.95);
    assert.equal(opt.name, 'natural');
  });

  it('should update parameters', () => {
    const opt = new NaturalGradient(0.1, 0.01, 0.95);
    const param = new Matrix(3, 1, new Float64Array([1, 2, 3]));
    const grad = new Matrix(3, 1, new Float64Array([0.5, -0.3, 0.1]));
    
    const updated = opt.update(param, grad, 'test');
    assert.equal(updated.rows, 3);
    // Parameters should change
    assert.notDeepStrictEqual(updated.data, param.data);
  });

  it('should accumulate Fisher estimates', () => {
    const opt = new NaturalGradient(0.1, 0.01, 0.9);
    const param = new Matrix(2, 1, new Float64Array([1, 2]));
    const grad = new Matrix(2, 1, new Float64Array([0.5, 0.3]));

    opt.update(param, grad, 'w1');
    const fisher = opt.getFisherEstimate('w1');
    assert.ok(fisher !== null);
    assert.ok(fisher.data[0] > 0); // Squared gradient should be positive
  });

  it('should scale updates by inverse Fisher', () => {
    const opt = new NaturalGradient(0.1, 0.001, 0.9);
    const param = new Matrix(2, 1, new Float64Array([0, 0]));

    // Large gradient on param 0, small on param 1
    for (let i = 0; i < 10; i++) {
      const grad = new Matrix(2, 1, new Float64Array([10, 0.1]));
      opt.update(param, grad, 'w');
    }

    // Now apply a uniform gradient — natural gradient should scale them differently
    const grad = new Matrix(2, 1, new Float64Array([1, 1]));
    const updated = opt.update(new Matrix(2, 1, new Float64Array([0, 0])), grad, 'w');
    
    // Param 0 should change less (it has a larger Fisher estimate, so step is smaller)
    const step0 = Math.abs(updated.data[0]);
    const step1 = Math.abs(updated.data[1]);
    assert.ok(step0 < step1,
      `Step for high-Fisher param should be smaller: ${step0.toFixed(6)} vs ${step1.toFixed(6)}`);
  });

  it('should be available via createOptimizer', () => {
    const opt = createOptimizer('natural', { lr: 0.05, damping: 0.1 });
    assert.equal(opt.name, 'natural');
    assert.equal(opt.lr, 0.05);
    assert.equal(opt.damping, 0.1);
  });
});

describe('KFAC Optimizer', () => {
  it('should create with default parameters', () => {
    const opt = new KFAC();
    assert.equal(opt.lr, 0.01);
    assert.equal(opt.damping, 0.01);
    assert.equal(opt.name, 'kfac');
  });

  it('should fall back to standard gradient without factors', () => {
    const opt = new KFAC(0.1);
    const param = new Matrix(3, 2, new Float64Array([1, 2, 3, 4, 5, 6]));
    const grad = new Matrix(3, 2, new Float64Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]));
    
    const updated = opt.update(param, grad, 'layer1');
    // Should do simple gradient descent
    for (let i = 0; i < 6; i++) {
      assert.ok(Math.abs(updated.data[i] - (param.data[i] - 0.1 * grad.data[i])) < 1e-10);
    }
  });

  it('should update Kronecker factors', () => {
    const opt = new KFAC(0.1, 0.01, 0.9, 1); // updateFreq=1
    const activation = new Matrix(3, 1, new Float64Array([1, 0.5, 0.2]));
    const gradient = new Matrix(2, 1, new Float64Array([0.3, -0.1]));
    
    opt.t = 0;
    opt.updateFactors('layer1', activation, gradient);
    assert.ok(opt.hasFactors('layer1'));
  });

  it('should compute natural gradient with factors', () => {
    const opt = new KFAC(0.1, 0.01, 0.9, 1);
    const activation = new Matrix(3, 1, new Float64Array([1, 0.5, 0.2]));
    const gradient = new Matrix(2, 1, new Float64Array([0.3, -0.1]));
    
    // Update factors multiple times
    for (let i = 0; i < 5; i++) {
      opt.t = i;
      opt.updateFactors('layer1', activation, gradient);
    }

    // Now use K-FAC update
    const param = new Matrix(2, 3, new Float64Array([1, 2, 3, 4, 5, 6])); // 2 out × 3 in
    const grad = new Matrix(2, 3, new Float64Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]));
    const updated = opt.update(param, grad, 'layer1');
    
    assert.equal(updated.rows, 2);
    assert.equal(updated.cols, 3);
    assert.ok(!updated.data.some(isNaN), 'K-FAC update should not produce NaN');
  });

  it('should invert small matrices correctly', () => {
    const opt = new KFAC();
    // Test inversion: 2×2 identity should invert to identity
    const I = new Matrix(2, 2, new Float64Array([1, 0, 0, 1]));
    const Iinv = opt._invert(I);
    assert.ok(Math.abs(Iinv.data[0] - 1) < 1e-10);
    assert.ok(Math.abs(Iinv.data[1]) < 1e-10);
    assert.ok(Math.abs(Iinv.data[2]) < 1e-10);
    assert.ok(Math.abs(Iinv.data[3] - 1) < 1e-10);
  });

  it('should invert non-trivial matrices', () => {
    const opt = new KFAC();
    // [[2, 1], [1, 3]] → inv ≈ [[0.6, -0.2], [-0.2, 0.4]]
    const M = new Matrix(2, 2, new Float64Array([2, 1, 1, 3]));
    const Minv = opt._invert(M);
    // M · Minv should ≈ I
    const product = M.dot(Minv);
    assert.ok(Math.abs(product.data[0] - 1) < 1e-8);
    assert.ok(Math.abs(product.data[1]) < 1e-8);
    assert.ok(Math.abs(product.data[2]) < 1e-8);
    assert.ok(Math.abs(product.data[3] - 1) < 1e-8);
  });
});
