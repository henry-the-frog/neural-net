// augmentation-depth.test.js — Data augmentation depth tests

import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { addNoise, randomFlipH } from './augmentation.js';
import { Matrix } from './matrix.js';

describe('addNoise', () => {
  it('preserves shape', () => {
    const input = Matrix.random(3, 5);
    const noisy = addNoise(input, 0.1);
    assert.equal(noisy.rows, 3);
    assert.equal(noisy.cols, 5);
  });

  it('changes values', () => {
    const input = Matrix.ones(1, 20);
    const noisy = addNoise(input, 0.5);
    let changed = false;
    for (let i = 0; i < 20; i++) {
      if (Math.abs(noisy.get(0, i) - 1) > 0.01) {
        changed = true;
        break;
      }
    }
    assert.ok(changed, 'Noise should change at least some values');
  });

  it('zero noise preserves values', () => {
    const input = Matrix.ones(1, 10);
    const noisy = addNoise(input, 0);
    for (let i = 0; i < 10; i++) {
      assert.equal(noisy.get(0, i), 1);
    }
  });
});

describe('randomFlipH', () => {
  it('preserves shape', () => {
    const input = Matrix.random(4, 6); // 4 samples, 2×3×1 images
    const flipped = randomFlipH(input, 3, 2, 1);
    assert.equal(flipped.rows, 4);
    assert.equal(flipped.cols, 6);
  });

  it('preserves values (just reorders)', () => {
    // For a known 2×2 image
    const input = new Matrix(1, 4, new Float64Array([1, 2, 3, 4]));
    const flipped = randomFlipH(input, 2, 2, 1);
    
    // Whether flipped or not, all values should be present
    const vals = new Set();
    for (let i = 0; i < 4; i++) vals.add(flipped.get(0, i));
    assert.ok(vals.has(1) && vals.has(2) && vals.has(3) && vals.has(4),
      'All original values should be present');
  });
});
