// mixed-precision.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { toFP16, toBF16, DynamicLossScaler, mixedPrecisionStep } from './mixed-precision.js';

describe('Mixed Precision', () => {
  test('toFP16 preserves normal values', () => {
    assert.ok(Math.abs(toFP16(1.0) - 1.0) < 0.01);
    assert.ok(Math.abs(toFP16(0.5) - 0.5) < 0.01);
  });

  test('toFP16 overflows large values', () => {
    assert.equal(toFP16(70000), Infinity);
    assert.equal(toFP16(-70000), -Infinity);
  });

  test('toFP16 underflows small values', () => {
    assert.equal(toFP16(1e-6), 0);
  });

  test('toBF16 has less precision than FP16', () => {
    const val = 1.234567;
    const fp16 = toFP16(val);
    const bf16 = toBF16(val);
    // BF16 has fewer mantissa bits → more rounding error
    const fp16Err = Math.abs(fp16 - val);
    const bf16Err = Math.abs(bf16 - val);
    assert.ok(bf16Err >= fp16Err, `BF16 error ${bf16Err} should be >= FP16 ${fp16Err}`);
  });

  test('DynamicLossScaler scales and unscales', () => {
    const scaler = new DynamicLossScaler(256);
    const loss = scaler.scaleUp(0.5);
    assert.equal(loss, 128);
    
    const grads = new Float64Array([128, 256]);
    const { gradients } = scaler.unscale(grads);
    assert.equal(gradients[0], 0.5);
    assert.equal(gradients[1], 1.0);
  });

  test('DynamicLossScaler backs off on overflow', () => {
    const scaler = new DynamicLossScaler(256);
    const initialScale = scaler.scale;
    scaler.update(true); // Overflow!
    assert.ok(scaler.scale < initialScale);
    assert.equal(scaler.overflows, 1);
  });

  test('DynamicLossScaler grows after stable steps', () => {
    const scaler = new DynamicLossScaler(256, 2, 0.5, 3);
    scaler.update(false);
    scaler.update(false);
    const before = scaler.scale;
    scaler.update(false); // 3rd good step → grow
    assert.ok(scaler.scale > before);
  });

  test('mixedPrecisionStep updates weights', () => {
    const scaler = new DynamicLossScaler(1);
    const weights = new Float64Array([1.0, 2.0, 3.0]);
    const grads = new Float64Array([0.1, 0.2, 0.3]);
    const { weights: updated, skipped } = mixedPrecisionStep(weights, grads, 0.01, scaler);
    assert.ok(!skipped);
    assert.ok(updated[0] < 1.0);
  });
});
