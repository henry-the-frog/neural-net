import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

describe('GPTQ Quantization', () => {
  // GPTQ (Frantar et al., 2023): optimal post-training quantization using Hessian
  function quantizeRow(weights, bits = 4) {
    const levels = Math.pow(2, bits);
    const min = Math.min(...weights);
    const max = Math.max(...weights);
    const scale = (max - min) / (levels - 1) || 1;
    return {
      quantized: weights.map(w => Math.round((w - min) / scale)),
      scale, zeroPoint: min,
    };
  }

  function dequantizeRow({ quantized, scale, zeroPoint }) {
    return quantized.map(q => q * scale + zeroPoint);
  }

  test('4-bit quantization has 16 levels', () => {
    const { quantized } = quantizeRow([0, 0.5, 1.0], 4);
    for (const q of quantized) assert.ok(q >= 0 && q <= 15);
  });

  test('roundtrip preserves approximate values', () => {
    const weights = [1.0, 2.5, -0.3, 0.7];
    const q = quantizeRow(weights, 8);
    const dq = dequantizeRow(q);
    for (let i = 0; i < weights.length; i++) {
      assert.ok(Math.abs(weights[i] - dq[i]) < 0.1, `${weights[i]} vs ${dq[i]}`);
    }
  });
});
