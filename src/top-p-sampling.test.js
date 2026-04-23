// top-p-sampling.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { topPSample, topKSample, topKPSample } from './top-p-sampling.js';

describe('Sampling Methods', () => {
  test('topPSample returns valid index', () => {
    const logits = new Float64Array([1, 2, 3, 4, 5]);
    const idx = topPSample(logits, 0.9);
    assert.ok(idx >= 0 && idx < 5);
  });

  test('topPSample with p=1 can sample any token', () => {
    const logits = new Float64Array([0, 0, 0, 0]);
    const seen = new Set();
    for (let i = 0; i < 100; i++) seen.add(topPSample(logits, 1.0));
    assert.ok(seen.size > 1, 'With uniform logits and p=1, should sample diverse tokens');
  });

  test('topPSample with very peaked distribution prefers argmax', () => {
    const logits = new Float64Array([0, 0, 0, 100]); // Token 3 strongly preferred
    const counts = new Float64Array(4);
    for (let i = 0; i < 100; i++) counts[topPSample(logits, 0.9)]++;
    assert.ok(counts[3] > 90, `Token 3 should dominate, got ${counts[3]}/100`);
  });

  test('topKSample returns valid index', () => {
    const logits = new Float64Array([1, 2, 3, 4, 5]);
    const idx = topKSample(logits, 3);
    assert.ok(idx >= 0 && idx < 5);
  });

  test('topKSample with k=1 returns argmax', () => {
    const logits = new Float64Array([1, 5, 2, 3]);
    const counts = new Float64Array(4);
    for (let i = 0; i < 50; i++) counts[topKSample(logits, 1)]++;
    assert.equal(counts[1], 50, 'k=1 should always return argmax');
  });

  test('topKPSample returns valid index', () => {
    const logits = new Float64Array([1, 2, 3, 4, 5]);
    const idx = topKPSample(logits, 3, 0.9);
    assert.ok(idx >= 0 && idx < 5);
  });

  test('temperature=0.01 is nearly deterministic', () => {
    const logits = new Float64Array([1, 5, 2, 3]);
    const counts = new Float64Array(4);
    for (let i = 0; i < 50; i++) counts[topPSample(logits, 0.9, 0.01)]++;
    assert.ok(counts[1] > 45, `Low temp should be nearly deterministic, got ${counts[1]}/50`);
  });

  test('high temperature = more random', () => {
    const logits = new Float64Array([1, 5, 2, 3]);
    const lowTempEntropy = new Set();
    const highTempEntropy = new Set();
    for (let i = 0; i < 100; i++) {
      lowTempEntropy.add(topPSample(logits, 0.99, 0.1));
      highTempEntropy.add(topPSample(logits, 0.99, 10.0));
    }
    assert.ok(highTempEntropy.size >= lowTempEntropy.size,
      `High temp should be more diverse: ${highTempEntropy.size} >= ${lowTempEntropy.size}`);
  });
});
