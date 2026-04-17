// capsule-stress.test.js — Verify CapsuleNet squash gradient
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { squash, squashBackward, CapsuleLayer } from '../src/capsule.js';

function relErr(a, n) {
  return Math.abs(a - n) / Math.max(Math.abs(a), Math.abs(n), 1e-8);
}

describe('Squash Gradient', () => {
  it('matches numerical gradient', () => {
    const s = [1.5, -0.3, 0.8];
    const dv = [0.2, -0.1, 0.4];
    const ds = squashBackward(s, dv);
    
    // Numerical gradient
    const eps = 1e-6;
    const numDs = [];
    for (let k = 0; k < 3; k++) {
      const sPlus = [...s]; sPlus[k] += eps;
      const sMinus = [...s]; sMinus[k] -= eps;
      const vPlus = squash(sPlus);
      const vMinus = squash(sMinus);
      // loss = sum(v_i * dv_i)
      let lPlus = 0, lMinus = 0;
      for (let i = 0; i < 3; i++) {
        lPlus += vPlus[i] * dv[i];
        lMinus += vMinus[i] * dv[i];
      }
      numDs.push((lPlus - lMinus) / (2 * eps));
    }
    
    let maxErr = 0;
    for (let k = 0; k < 3; k++) {
      const err = relErr(ds[k], numDs[k]);
      maxErr = Math.max(maxErr, err);
    }
    assert.ok(maxErr < 0.01, `Squash gradient error: ${maxErr.toExponential(2)}`);
  });

  it('gradient at origin (small norm)', () => {
    const s = [0.01, 0.01, 0.01];
    const dv = [1, 0, 0];
    const ds = squashBackward(s, dv);
    // At small norm, squash ≈ s, so gradient ≈ identity
    assert.ok(ds.every(Number.isFinite), 'Gradient should be finite at small norm');
  });

  it('gradient at large norm', () => {
    const s = [10, 10, 10];
    const dv = [1, 0, 0];
    const ds = squashBackward(s, dv);
    assert.ok(ds.every(Number.isFinite), 'Gradient should be finite at large norm');
  });
});

describe('CapsuleLayer Training', () => {
  it('loss decreases during training', () => {
    let passed = false;
    for (let attempt = 0; attempt < 3 && !passed; attempt++) {
      const layer = new CapsuleLayer(2, 3, 3, 4, 3);
      // Random inputs
      const inputs = [
        [Math.random(), Math.random(), Math.random(), Math.random()],
        [Math.random(), Math.random(), Math.random(), Math.random()],
        [Math.random(), Math.random(), Math.random(), Math.random()],
      ];
      // Target: capsule 0 should be active (high norm)
      const target = [[1, 0, 0], [0, 0, 0]];
      
      let firstLoss = null;
      for (let step = 0; step < 50; step++) {
        const output = layer.forward(inputs);
        // MSE loss
        let loss = 0;
        const dOutput = output.map((cap, j) => {
          return cap.map((v, d) => {
            const t = target[j] ? (target[j][d] || 0) : 0;
            const diff = v - t;
            loss += diff * diff;
            return 2 * diff;
          });
        });
        if (firstLoss === null) firstLoss = loss;
        layer.backward(dOutput);
        layer.update(0.01);
      }
      
      const finalOutput = layer.forward(inputs);
      let finalLoss = 0;
      finalOutput.forEach((cap, j) => {
        cap.forEach((v, d) => {
          const t = target[j] ? (target[j][d] || 0) : 0;
          finalLoss += (v - t) ** 2;
        });
      });
      
      if (finalLoss < firstLoss) passed = true;
    }
    assert.ok(passed, 'CapsuleLayer training should decrease loss');
  });
});
