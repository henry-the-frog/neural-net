import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { addGaussianNoise, mixup, cutmix } from './data-augmentation.js';

describe('Data Augmentation', () => {
  test('gaussian noise changes data', () => {
    const data = [1, 2, 3, 4, 5];
    const noisy = addGaussianNoise(data, 0.1);
    let diff = 0;
    for (let i = 0; i < data.length; i++) diff += Math.abs(data[i] - noisy[i]);
    assert.ok(diff > 0, 'Noise should change data');
  });

  test('mixup produces intermediate values', () => {
    const { x, y } = mixup([0, 0], 0, [10, 10], 1);
    assert.ok(x[0] >= 0 && x[0] <= 10);
    assert.ok(y >= 0 && y <= 1);
  });

  test('cutmix preserves some original data', () => {
    const x1 = [1, 1, 1, 1, 1];
    const x2 = [2, 2, 2, 2, 2];
    const { mixed } = cutmix(x1, x2);
    assert.ok(mixed.some(v => v === 1));
  });
});
