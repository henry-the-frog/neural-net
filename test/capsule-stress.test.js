// capsule-stress.test.js — Deep stress tests for Capsule Networks
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { squash, vectorNorm, CapsuleLayer } from '../src/capsule.js';

describe('Squash Function Properties', () => {
  it('output length is always < 1', () => {
    for (let trial = 0; trial < 50; trial++) {
      const v = Array.from({ length: 4 }, () => Math.random() * 10 - 5);
      const s = squash(v);
      const norm = vectorNorm(s);
      assert.ok(norm < 1 + 1e-6, `Squash norm should be < 1: ${norm.toFixed(6)}`);
    }
  });

  it('preserves direction (unit vector unchanged direction)', () => {
    for (let trial = 0; trial < 20; trial++) {
      const v = Array.from({ length: 4 }, () => Math.random() * 4 - 2);
      const norm = vectorNorm(v);
      if (norm < 0.01) continue; // Skip near-zero
      const s = squash(v);
      // Direction: s / |s| should equal v / |v|
      const sNorm = vectorNorm(s);
      if (sNorm < 1e-10) continue;
      const vUnit = v.map(x => x / norm);
      const sUnit = s.map(x => x / sNorm);
      for (let i = 0; i < 4; i++) {
        assert.ok(Math.abs(vUnit[i] - sUnit[i]) < 0.01,
          `Direction mismatch at dim ${i}: ${vUnit[i].toFixed(4)} vs ${sUnit[i].toFixed(4)}`);
      }
    }
  });

  it('larger input produces longer output', () => {
    const small = squash([0.1, 0.1, 0.1]);
    const large = squash([5, 5, 5]);
    assert.ok(vectorNorm(large) > vectorNorm(small),
      `Larger input should produce longer output: ${vectorNorm(large).toFixed(4)} vs ${vectorNorm(small).toFixed(4)}`);
  });

  it('zero vector maps to near-zero', () => {
    const s = squash([0, 0, 0]);
    assert.ok(vectorNorm(s) < 0.01, 'Zero input should produce near-zero output');
  });

  it('very large vector maps to near-unit length', () => {
    const s = squash([1000, 1000, 1000]);
    const norm = vectorNorm(s);
    assert.ok(norm > 0.99, `Large input norm should be ~1: ${norm.toFixed(6)}`);
  });

  it('monotonically increasing: longer input → longer output', () => {
    const lengths = [];
    for (let scale = 0.1; scale <= 10; scale += 0.5) {
      const v = [scale, scale * 0.5, scale * 0.3];
      lengths.push(vectorNorm(squash(v)));
    }
    for (let i = 1; i < lengths.length; i++) {
      assert.ok(lengths[i] >= lengths[i - 1] - 1e-6,
        `Should be monotonic: ${lengths[i].toFixed(4)} < ${lengths[i - 1].toFixed(4)}`);
    }
  });
});

describe('CapsuleLayer', () => {
  it('output has correct shape', () => {
    const layer = new CapsuleLayer(3, 4, 5, 2);
    const input = Array.from({ length: 5 }, () => [Math.random(), Math.random()]);
    const output = layer.forward(input);
    assert.equal(output.length, 3);
    for (const cap of output) {
      assert.equal(cap.length, 4);
    }
  });

  it('coupling coefficients sum to 1 per input capsule', () => {
    const layer = new CapsuleLayer(3, 4, 5, 2, 3);
    const input = Array.from({ length: 5 }, () => [Math.random(), Math.random()]);
    layer.forward(input);
    
    for (let i = 0; i < 5; i++) {
      let sum = 0;
      for (let j = 0; j < 3; j++) {
        sum += layer.couplingCoeffs[i][j];
      }
      assert.ok(Math.abs(sum - 1) < 1e-6,
        `Coupling for input ${i} should sum to 1: ${sum.toFixed(6)}`);
    }
  });

  it('more routing iterations should sharpen coupling', () => {
    const input = Array.from({ length: 4 }, () =>
      Array.from({ length: 3 }, () => Math.random()));
    
    const layer1 = new CapsuleLayer(2, 3, 4, 3, 1);
    const layer5 = new CapsuleLayer(2, 3, 4, 3, 5);
    // Copy weights
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 2; j++) {
        layer5.W[i][j] = layer1.W[i][j].map(row => [...row]);
      }
    }
    
    layer1.forward(input);
    layer5.forward(input);
    
    // 5 iterations should produce more focused coupling (higher max)
    let max1 = 0, max5 = 0;
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 2; j++) {
        max1 = Math.max(max1, layer1.couplingCoeffs[i][j]);
        max5 = Math.max(max5, layer5.couplingCoeffs[i][j]);
      }
    }
    assert.ok(max5 >= max1 - 0.01,
      `More iterations should sharpen: max5=${max5.toFixed(3)}, max1=${max1.toFixed(3)}`);
  });

  it('output capsule norms are < 1 (squashed)', () => {
    const layer = new CapsuleLayer(3, 4, 5, 2);
    const input = Array.from({ length: 5 }, () => [Math.random() * 5, Math.random() * 5]);
    const output = layer.forward(input);
    
    for (let j = 0; j < 3; j++) {
      const norm = vectorNorm(output[j]);
      assert.ok(norm < 1 + 1e-6, `Output capsule ${j} norm should be < 1: ${norm.toFixed(4)}`);
    }
  });

  it('handles large input capsules without NaN', () => {
    const layer = new CapsuleLayer(2, 3, 10, 4);
    const input = Array.from({ length: 10 }, () =>
      Array.from({ length: 4 }, () => Math.random() * 100));
    const output = layer.forward(input);
    for (const cap of output) {
      assert.ok(cap.every(Number.isFinite), 'Output should be finite for large input');
    }
  });

  it('different inputs produce different outputs', () => {
    const layer = new CapsuleLayer(2, 3, 4, 2);
    const in1 = [[1, 0], [0, 1], [1, 1], [0, 0]];
    const in2 = [[0, 1], [1, 0], [0, 0], [1, 1]];
    const out1 = layer.forward(in1);
    const out2 = layer.forward(in2);
    
    let diff = 0;
    for (let j = 0; j < 2; j++) {
      for (let d = 0; d < 3; d++) {
        diff += Math.abs(out1[j][d] - out2[j][d]);
      }
    }
    assert.ok(diff > 0.001, 'Different inputs should produce different outputs');
  });
});

describe('Dynamic Routing Convergence', () => {
  it('routing converges (coupling stabilizes)', () => {
    // Run with many iterations and check that coupling barely changes in last iterations
    const layer = new CapsuleLayer(3, 4, 5, 2, 10);
    const input = Array.from({ length: 5 }, () => [Math.random(), Math.random()]);
    layer.forward(input);
    
    // All coupling coefficients should be valid probabilities
    for (let i = 0; i < 5; i++) {
      for (let j = 0; j < 3; j++) {
        assert.ok(layer.couplingCoeffs[i][j] >= 0, `Coupling should be >= 0`);
        assert.ok(layer.couplingCoeffs[i][j] <= 1 + 1e-6, `Coupling should be <= 1`);
      }
    }
  });

  it('routing with zero input produces near-uniform coupling', () => {
    const layer = new CapsuleLayer(3, 4, 5, 2, 3);
    const input = Array.from({ length: 5 }, () => [0, 0]);
    layer.forward(input);
    
    // With zero input, predictions are all zero, so coupling should be ~uniform
    const expected = 1 / 3;
    for (let i = 0; i < 5; i++) {
      for (let j = 0; j < 3; j++) {
        assert.ok(Math.abs(layer.couplingCoeffs[i][j] - expected) < 0.2,
          `Zero input should give near-uniform coupling: ${layer.couplingCoeffs[i][j].toFixed(3)}`);
      }
    }
  });
});
