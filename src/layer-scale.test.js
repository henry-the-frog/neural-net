// layer-scale.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { LayerScale, StochasticDepth, linearDropSchedule } from './layer-scale.js';
import { Matrix } from './matrix.js';

describe('Layer Scale & Stochastic Depth', () => {
  test('LayerScale starts near zero', () => {
    const ls = new LayerScale(4, 1e-6);
    const x = Matrix.ones(2, 4);
    const scaled = ls.forward(x);
    for (let i = 0; i < scaled.data.length; i++) {
      assert.ok(Math.abs(scaled.data[i]) < 0.001, `Should be near 0: ${scaled.data[i]}`);
    }
  });

  test('LayerScale applies per-channel scaling', () => {
    const ls = new LayerScale(3, 1.0);
    ls.gamma[0] = 0.5;
    ls.gamma[1] = 1.0;
    ls.gamma[2] = 2.0;
    
    const x = Matrix.ones(1, 3);
    const scaled = ls.forward(x);
    assert.ok(Math.abs(scaled.get(0, 0) - 0.5) < 1e-10);
    assert.ok(Math.abs(scaled.get(0, 1) - 1.0) < 1e-10);
    assert.ok(Math.abs(scaled.get(0, 2) - 2.0) < 1e-10);
  });

  test('StochasticDepth: eval mode always adds residual', () => {
    const sd = new StochasticDepth(0.99); // Very high drop rate
    sd.training = false; // Eval mode
    
    const x = Matrix.ones(2, 4);
    const residual = Matrix.ones(2, 4);
    
    // In eval mode, should always add residual
    for (let i = 0; i < 10; i++) {
      const out = sd.forward(x, residual);
      assert.ok(Math.abs(out.get(0, 0) - 2) < 1e-10, 'Eval: should always be x + residual');
    }
  });

  test('StochasticDepth: training mode sometimes drops', () => {
    const sd = new StochasticDepth(0.5);
    sd.training = true;
    
    const x = Matrix.ones(1, 2);
    const residual = Matrix.ones(1, 2);
    
    let hasIdentity = false;
    let hasResidual = false;
    for (let i = 0; i < 100; i++) {
      const out = sd.forward(x, residual);
      if (Math.abs(out.get(0, 0) - 1) < 0.01) hasIdentity = true;
      if (out.get(0, 0) > 1.5) hasResidual = true;
    }
    assert.ok(hasIdentity, 'Should sometimes return identity');
    assert.ok(hasResidual, 'Should sometimes add residual');
  });

  test('linearDropSchedule: first layer has 0, last has max', () => {
    const schedule = linearDropSchedule(10, 0.2);
    assert.equal(schedule.length, 10);
    assert.ok(Math.abs(schedule[0]) < 1e-10);
    assert.ok(Math.abs(schedule[9] - 0.2) < 1e-10);
  });
});
