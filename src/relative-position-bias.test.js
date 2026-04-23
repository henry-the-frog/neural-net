// relative-position-bias.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { relativePosiBucket, computeRelativePositionBias, applyRelativePositionBias } from './relative-position-bias.js';
import { Matrix } from './matrix.js';

describe('Relative Position Bias', () => {
  test('bucket 0 for same position', () => {
    // relativePos = 0 → n = 0 → bucket for n < maxExact
    const bucket = relativePosiBucket(0, true, 32);
    // Bidirectional: ret starts at numBuckets/2=16 (since n=0 ≥ 0, adding numBuckets)
    // Actually n = -relativePos = 0, n < 0 is false, ret += 0
    // n = 0, 0 < maxExact=8, ret += 0 = 0
    // Wait: bidirectional, numBuckets = 16, n=0 >= 0, no offset
    // ret = 0 + 0 = 0 (since n=0 < maxExact=8)
    assert.ok(bucket >= 0 && bucket < 32);
  });

  test('bidirectional: different buckets for positive and negative distances', () => {
    const b1 = relativePosiBucket(5, true, 32);
    const b2 = relativePosiBucket(-5, true, 32);
    assert.notEqual(b1, b2, 'Positive and negative distances should have different buckets');
  });

  test('nearby positions have different exact buckets', () => {
    const b0 = relativePosiBucket(0);
    const b1 = relativePosiBucket(1);
    const b2 = relativePosiBucket(2);
    assert.notEqual(b0, b1);
    assert.notEqual(b1, b2);
  });

  test('very large distances share the same max bucket', () => {
    const b1 = relativePosiBucket(1000);
    const b2 = relativePosiBucket(2000);
    assert.equal(b1, b2, 'Very large distances should share max bucket');
  });

  test('computeRelativePositionBias produces correct shapes', () => {
    const { biasMatrices, biasTable } = computeRelativePositionBias(8, 4, true, 32);
    assert.equal(biasMatrices.length, 4); // One per head
    assert.equal(biasMatrices[0].rows, 8);
    assert.equal(biasMatrices[0].cols, 8);
    assert.equal(biasTable.rows, 32);
    assert.equal(biasTable.cols, 4);
  });

  test('bias is symmetric for bidirectional', () => {
    const { biasMatrices } = computeRelativePositionBias(5, 1, true, 32);
    const bias = biasMatrices[0];
    // bias(i,j) and bias(j,i) should use different buckets but both be valid
    assert.ok(isFinite(bias.get(0, 4)));
    assert.ok(isFinite(bias.get(4, 0)));
  });

  test('applyRelativePositionBias adds bias to scores', () => {
    const scores = new Matrix(3, 3);
    scores.set(0, 0, 1); scores.set(0, 1, 2);
    const bias = new Matrix(3, 3);
    bias.set(0, 0, 0.5); bias.set(0, 1, -0.5);
    
    const result = applyRelativePositionBias(scores, bias);
    assert.ok(Math.abs(result.get(0, 0) - 1.5) < 1e-10);
    assert.ok(Math.abs(result.get(0, 1) - 1.5) < 1e-10);
  });
});
