// groupnorm.test.js — Group Normalization tests
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { GroupNorm } from './groupnorm.js';
import { Matrix } from './matrix.js';

describe('GroupNorm', () => {
  test('forward produces correct shape', () => {
    const gn = new GroupNorm(2, 8);
    const x = Matrix.random(4, 8);
    const out = gn.forward(x);
    assert.equal(out.rows, 4);
    assert.equal(out.cols, 8);
  });

  test('each group has zero mean after normalization', () => {
    const gn = new GroupNorm(2, 8, 1e-5, false); // No affine
    const x = Matrix.random(3, 8);
    const out = gn.forward(x);
    
    for (let b = 0; b < 3; b++) {
      for (let g = 0; g < 2; g++) {
        let mean = 0;
        for (let c = g * 4; c < (g + 1) * 4; c++) mean += out.get(b, c);
        mean /= 4;
        assert.ok(Math.abs(mean) < 0.01, `Group ${g} mean should be ~0, got ${mean}`);
      }
    }
  });

  test('each group has unit variance after normalization', () => {
    const gn = new GroupNorm(2, 8, 1e-5, false);
    const x = Matrix.random(10, 8); // More samples for stable variance
    const out = gn.forward(x);
    
    for (let b = 0; b < 10; b++) {
      for (let g = 0; g < 2; g++) {
        let mean = 0, var_ = 0;
        for (let c = g * 4; c < (g + 1) * 4; c++) mean += out.get(b, c);
        mean /= 4;
        for (let c = g * 4; c < (g + 1) * 4; c++) var_ += (out.get(b, c) - mean) ** 2;
        var_ /= 4;
        assert.ok(Math.abs(var_ - 1) < 0.1, `Group ${g} variance should be ~1, got ${var_}`);
      }
    }
  });

  test('backward produces correct shape', () => {
    const gn = new GroupNorm(2, 8);
    const x = Matrix.random(3, 8);
    gn.forward(x);
    
    const dOutput = Matrix.random(3, 8);
    const dInput = gn.backward(dOutput);
    assert.equal(dInput.rows, 3);
    assert.equal(dInput.cols, 8);
  });

  test('affine gamma/beta are learnable', () => {
    const gn = new GroupNorm(2, 4);
    const x = Matrix.random(5, 4);
    gn.forward(x);
    
    const dOutput = Matrix.ones(5, 4);
    gn.backward(dOutput);
    
    // dGamma and dBeta should be non-zero
    let dGammaSum = 0, dBetaSum = 0;
    for (let i = 0; i < 4; i++) {
      dGammaSum += Math.abs(gn.dGamma[i]);
      dBetaSum += Math.abs(gn.dBeta[i]);
    }
    assert.ok(dGammaSum > 0, 'dGamma should be non-zero');
    assert.ok(dBetaSum > 0, 'dBeta should be non-zero');
  });

  test('numGroups=1 behaves like LayerNorm', () => {
    const gn = new GroupNorm(1, 8, 1e-5, false);
    const x = Matrix.random(3, 8);
    const out = gn.forward(x);
    
    // With 1 group, all channels are normalized together
    for (let b = 0; b < 3; b++) {
      let mean = 0;
      for (let c = 0; c < 8; c++) mean += out.get(b, c);
      mean /= 8;
      assert.ok(Math.abs(mean) < 0.01, `LayerNorm-like: mean should be ~0, got ${mean}`);
    }
  });

  test('numChannels must be divisible by numGroups', () => {
    assert.throws(() => new GroupNorm(3, 8), /divisible/);
  });

  test('paramCount is 2*C for affine, 0 otherwise', () => {
    assert.equal(new GroupNorm(2, 8).paramCount(), 16);
    assert.equal(new GroupNorm(2, 8, 1e-5, false).paramCount(), 0);
  });

  test('update modifies gamma/beta', () => {
    const gn = new GroupNorm(2, 4);
    const x = Matrix.random(3, 4);
    gn.forward(x);
    const dOutput = Matrix.ones(3, 4);
    gn.backward(dOutput);
    
    const origGamma = [...gn.gamma];
    gn.update(0.01);
    
    let changed = false;
    for (let i = 0; i < 4; i++) {
      if (gn.gamma[i] !== origGamma[i]) changed = true;
    }
    assert.ok(changed, 'gamma should change after update');
  });
});
