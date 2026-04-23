import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { attentionEntropy, maxEntropy, headRedundancy } from './attention-entropy.js';

describe('Attention Entropy', () => {
  test('uniform attention has max entropy', () => {
    const uniform = [0.25, 0.25, 0.25, 0.25];
    const entropy = attentionEntropy(uniform);
    assert.ok(Math.abs(entropy - 2) < 0.01); // log2(4) = 2
  });

  test('peaked attention has low entropy', () => {
    const peaked = [0.97, 0.01, 0.01, 0.01];
    const entropy = attentionEntropy(peaked);
    assert.ok(entropy < 0.5);
  });

  test('maxEntropy for 8 tokens is 3', () => {
    assert.ok(Math.abs(maxEntropy(8) - 3) < 0.01);
  });

  test('identical heads have high redundancy', () => {
    const head = [[0.5, 0.5], [0.3, 0.7]];
    assert.ok(headRedundancy([head, head]) > 0.99);
  });
});
