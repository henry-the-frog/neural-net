// attention-masks.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { causalMask, slidingWindowMask, prefixMask, blockSparseMask, paddingMask, combineMasks } from './attention-masks.js';

describe('Attention Masks', () => {
  test('causal mask: lower triangle is 0, upper is -inf', () => {
    const mask = causalMask(4);
    assert.equal(mask.get(0, 0), 0);
    assert.ok(mask.get(0, 1) < -1e8);
    assert.equal(mask.get(2, 1), 0);
    assert.ok(mask.get(1, 2) < -1e8);
  });

  test('sliding window mask respects window size', () => {
    const mask = slidingWindowMask(6, 2);
    assert.equal(mask.get(3, 3), 0);  // Self
    assert.equal(mask.get(3, 2), 0);  // 1 back
    assert.equal(mask.get(3, 1), 0);  // 2 back
    assert.ok(mask.get(3, 0) < -1e8); // 3 back = out of window
    assert.ok(mask.get(3, 4) < -1e8); // Future
  });

  test('prefix mask: prefix is bidirectional', () => {
    const mask = prefixMask(6, 2);
    // Prefix tokens (0, 1) see each other
    assert.equal(mask.get(0, 1), 0);
    assert.equal(mask.get(1, 0), 0);
    // Non-prefix (2) can see prefix + self
    assert.equal(mask.get(2, 0), 0);
    assert.equal(mask.get(2, 1), 0);
    assert.equal(mask.get(2, 2), 0);
    assert.ok(mask.get(2, 3) < -1e8); // Future
  });

  test('block sparse mask: adjacent blocks are connected', () => {
    const mask = blockSparseMask(8, 4);
    // Block 0 (0-3) and Block 1 (4-7) should connect
    assert.equal(mask.get(0, 4), 0);
    assert.equal(mask.get(3, 7), 0);
    // Within same block
    assert.equal(mask.get(0, 3), 0);
  });

  test('padding mask: pads after actual length', () => {
    const masks = paddingMask([3, 5], 6);
    assert.equal(masks[0][2], 0);   // Valid
    assert.ok(masks[0][3] < -1e8);   // Padded
    assert.equal(masks[1][4], 0);    // Valid
    assert.ok(masks[1][5] < -1e8);   // Padded
  });

  test('combine masks adds biases', () => {
    const causal = causalMask(3);
    const other = causalMask(3); // Same mask
    const combined = combineMasks(causal, other);
    // Where both are 0 → 0, where one is -inf → still clamped
    assert.equal(combined.get(1, 0), 0);
    assert.ok(combined.get(0, 1) < -1e8);
  });
});
