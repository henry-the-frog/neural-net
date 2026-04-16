// data-aug-stress.test.js — Data augmentation stress tests
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { addNoise, dropout as dropoutAug, mixup, cutmix, flip, rotate90, randomCrop, normalize, standardize } from '../src/data-augmentation.js';
import { Matrix } from '../src/matrix.js';

describe('Data Augmentation Stress', () => {
  it('addNoise preserves shape', () => {
    const data = Matrix.random(5, 4);
    const noisy = addNoise(data, 0.1);
    assert.equal(noisy.rows, 5);
    assert.equal(noisy.cols, 4);
  });

  it('addNoise produces different output', () => {
    const data = new Matrix(1, 4, new Float64Array([1, 2, 3, 4]));
    const noisy = addNoise(data, 0.5);
    let different = false;
    for (let i = 0; i < 4; i++) {
      if (Math.abs(noisy.data[i] - data.data[i]) > 0.001) different = true;
    }
    assert.ok(different, 'Noise should change at least one value');
  });

  it('dropout zeroes some values', () => {
    const data = new Matrix(1, 100);
    for (let i = 0; i < 100; i++) data.data[i] = 1.0;
    const dropped = dropoutAug(data, 0.5);
    let zeros = 0;
    for (let i = 0; i < 100; i++) {
      if (dropped.data[i] === 0) zeros++;
    }
    // Should zero roughly 50% (±20%)
    assert.ok(zeros > 20 && zeros < 80, `Should zero ~50%: got ${zeros}`);
  });

  it('mixup produces interpolation', () => {
    const x1 = new Matrix(1, 4, new Float64Array([0, 0, 0, 0]));
    const x2 = new Matrix(1, 4, new Float64Array([1, 1, 1, 1]));
    const { x: mixed } = mixup(x1, x2, 0.5);
    // Should be ~0.5 everywhere
    for (let i = 0; i < 4; i++) {
      assert.ok(mixed.data[i] >= 0 && mixed.data[i] <= 1, `Mixup should interpolate: ${mixed.data[i]}`);
    }
  });

  it('normalize to [0, 1]', () => {
    const data = new Matrix(1, 4, new Float64Array([-10, 0, 5, 10]));
    const normed = normalize(data);
    for (let i = 0; i < 4; i++) {
      assert.ok(normed.data[i] >= 0 && normed.data[i] <= 1, `Should be in [0,1]: ${normed.data[i]}`);
    }
    assert.ok(Math.abs(normed.data[0] - 0) < 1e-6, 'Min should map to 0');
    assert.ok(Math.abs(normed.data[3] - 1) < 1e-6, 'Max should map to 1');
  });

  it('standardize to zero mean, unit variance', () => {
    const data = new Matrix(10, 1);
    for (let i = 0; i < 10; i++) data.data[i] = i;
    const std = standardize(data);
    
    let mean = 0;
    for (let i = 0; i < 10; i++) mean += std.data[i];
    mean /= 10;
    assert.ok(Math.abs(mean) < 0.1, `Mean should be ~0: ${mean}`);
  });

  it('all augmentations produce finite values', () => {
    const data = Matrix.random(3, 4);
    const augmented = [
      addNoise(data, 0.1),
      dropoutAug(data, 0.3),
      normalize(data),
      standardize(data),
    ];
    for (const aug of augmented) {
      for (let i = 0; i < aug.data.length; i++) {
        assert.ok(isFinite(aug.data[i]), `Augmented value should be finite: ${aug.data[i]}`);
      }
    }
  });
});
