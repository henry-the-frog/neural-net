// activation-functions.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { relu, gelu, silu, mish, elu, selu, sigmoid, hardSwish, softmax, getActivation } from './activation-functions.js';

describe('Activation Functions', () => {
  test('ReLU: negative → 0, positive → identity', () => {
    assert.equal(relu(-5), 0);
    assert.equal(relu(3), 3);
    assert.equal(relu(0), 0);
  });

  test('GELU(0) ≈ 0', () => {
    assert.ok(Math.abs(gelu(0)) < 0.001);
  });

  test('GELU approximates ReLU for large positive x', () => {
    assert.ok(Math.abs(gelu(10) - 10) < 0.01);
  });

  test('SiLU(0) = 0', () => {
    assert.ok(Math.abs(silu(0)) < 0.001);
  });

  test('Mish(0) = 0', () => {
    assert.ok(Math.abs(mish(0)) < 0.001);
  });

  test('ELU: positive → identity, negative → bounded below', () => {
    assert.equal(elu(5), 5);
    assert.ok(elu(-10) > -1.01); // Approaches -α
  });

  test('SELU preserves mean for normal inputs', () => {
    const values = Array.from({ length: 1000 }, () => (Math.random() - 0.5) * 2);
    const activated = values.map(selu);
    const mean = activated.reduce((a, b) => a + b) / activated.length;
    assert.ok(Math.abs(mean) < 0.5, `Mean should be near 0, got ${mean}`);
  });

  test('sigmoid range is (0, 1)', () => {
    assert.ok(sigmoid(-100) >= 0);
    assert.ok(sigmoid(100) <= 1);
    assert.ok(Math.abs(sigmoid(0) - 0.5) < 0.001);
  });

  test('hardSwish approximates SiLU', () => {
    // For x in [-3, 3], hardSwish ≈ silu
    for (const x of [-1, 0, 1, 2]) {
      const diff = Math.abs(hardSwish(x) - silu(x));
      assert.ok(diff < 0.3, `hardSwish(${x})=${hardSwish(x)} vs silu=${silu(x)}`);
    }
  });

  test('softmax sums to 1', () => {
    const result = softmax([1, 2, 3]);
    const sum = result.reduce((a, b) => a + b);
    assert.ok(Math.abs(sum - 1) < 1e-6);
  });

  test('getActivation returns function by name', () => {
    const fn = getActivation('gelu');
    assert.ok(Math.abs(fn(0)) < 0.001);
  });
});
