// adaln.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { AdaLN, AdaLNZero, layerNorm } from './adaln.js';
import { Matrix } from './matrix.js';

describe('AdaLN', () => {
  test('layerNorm normalizes to zero mean', () => {
    const x = new Matrix(2, 4);
    x.set(0, 0, 2); x.set(0, 1, 4); x.set(0, 2, 6); x.set(0, 3, 8);
    const normed = layerNorm(x);
    
    let mean = 0;
    for (let j = 0; j < 4; j++) mean += normed.get(0, j);
    mean /= 4;
    assert.ok(Math.abs(mean) < 0.001, `Mean should be ~0, got ${mean}`);
  });

  test('AdaLN output shape matches input', () => {
    const adaln = new AdaLN(8, 4);
    const x = Matrix.random(3, 8);
    const cond = new Float64Array([1, 0, 0, 0]);
    const out = adaln.forward(x, cond);
    assert.equal(out.rows, 3);
    assert.equal(out.cols, 8);
  });

  test('different conditioning produces different outputs', () => {
    const adaln = new AdaLN(8, 4);
    const x = Matrix.random(3, 8);
    const out1 = adaln.forward(x, new Float64Array([1, 0, 0, 0]));
    const out2 = adaln.forward(x, new Float64Array([0, 0, 0, 1]));
    
    let diff = 0;
    for (let i = 0; i < out1.data.length; i++) diff += Math.abs(out1.data[i] - out2.data[i]);
    assert.ok(diff > 0.01, 'Different conditioning should produce different outputs');
  });

  test('AdaLN-Zero alpha starts near zero', () => {
    const adaln = new AdaLNZero(8, 4);
    const x = Matrix.random(3, 8);
    const cond = new Float64Array(4).fill(0); // Zero conditioning
    const { alpha } = adaln.forward(x, cond);
    
    // Alpha should be near zero at initialization
    let sumAlpha = 0;
    for (let i = 0; i < alpha.length; i++) sumAlpha += Math.abs(alpha[i]);
    assert.ok(sumAlpha < 0.1, `Alpha sum should be near 0, got ${sumAlpha}`);
  });

  test('AdaLN-Zero residual with alpha=0 is identity', () => {
    const adaln = new AdaLNZero(4, 2);
    const x = Matrix.random(2, 4);
    const blockOutput = Matrix.random(2, 4);
    const alpha = new Float64Array(4); // All zeros
    
    const result = adaln.applyResidual(x, blockOutput, alpha);
    for (let i = 0; i < x.data.length; i++) {
      assert.ok(Math.abs(result.data[i] - x.data[i]) < 1e-10, 'Alpha=0 residual should be identity');
    }
  });

  test('AdaLN-Zero residual with alpha=1 adds block output', () => {
    const adaln = new AdaLNZero(4, 2);
    const x = Matrix.random(2, 4);
    const blockOutput = Matrix.random(2, 4);
    const alpha = new Float64Array(4).fill(1);
    
    const result = adaln.applyResidual(x, blockOutput, alpha);
    for (let i = 0; i < x.data.length; i++) {
      const expected = x.data[i] + blockOutput.data[i];
      assert.ok(Math.abs(result.data[i] - expected) < 1e-10);
    }
  });
});
