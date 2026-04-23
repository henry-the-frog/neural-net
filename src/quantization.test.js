// quantization.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { quantizeAbsmax, dequantizeAbsmax, quantizePerChannel, dequantizePerChannel, quantizationError, compressionRatio } from './quantization.js';
import { Matrix } from './matrix.js';

describe('Quantization', () => {
  test('absmax roundtrip preserves shape', () => {
    const w = Matrix.random(4, 8);
    const q = quantizeAbsmax(w);
    const dq = dequantizeAbsmax(q);
    assert.equal(dq.rows, 4);
    assert.equal(dq.cols, 8);
  });

  test('absmax roundtrip has low error', () => {
    const w = Matrix.random(10, 10);
    const q = quantizeAbsmax(w);
    const dq = dequantizeAbsmax(q);
    const err = quantizationError(w, dq);
    assert.ok(err.rmse < 0.01, `RMSE ${err.rmse} should be small`);
  });

  test('per-channel roundtrip preserves shape', () => {
    const w = Matrix.random(4, 8);
    const q = quantizePerChannel(w);
    const dq = dequantizePerChannel(q);
    assert.equal(dq.rows, 4);
    assert.equal(dq.cols, 8);
  });

  test('per-channel is more accurate than absmax for variable scales', () => {
    const w = new Matrix(4, 4);
    // Row 0: small values, Row 3: large values
    for (let j = 0; j < 4; j++) {
      w.set(0, j, 0.001 * (j + 1));
      w.set(1, j, 0.1 * (j + 1));
      w.set(2, j, 1.0 * (j + 1));
      w.set(3, j, 100.0 * (j + 1));
    }
    
    const absErr = quantizationError(w, dequantizeAbsmax(quantizeAbsmax(w)));
    const chanErr = quantizationError(w, dequantizePerChannel(quantizePerChannel(w)));
    
    assert.ok(chanErr.rmse < absErr.rmse, 
      `Per-channel RMSE ${chanErr.rmse} should be < absmax ${absErr.rmse}`);
  });

  test('compression ratio is ~8x for INT8', () => {
    const w = Matrix.random(100, 100);
    const q = quantizeAbsmax(w);
    const ratio = compressionRatio(w, q);
    assert.ok(ratio.ratio.includes('7') || ratio.ratio.includes('8'), 
      `Expected ~8x compression, got ${ratio.ratio}`);
  });

  test('quantized values are in [-127, 127]', () => {
    const w = Matrix.random(10, 10);
    const q = quantizeAbsmax(w);
    for (let i = 0; i < q.quantized.length; i++) {
      assert.ok(q.quantized[i] >= -127 && q.quantized[i] <= 127);
    }
  });

  test('SNR is positive for normal weights', () => {
    const w = Matrix.random(50, 50);
    const q = quantizeAbsmax(w);
    const err = quantizationError(w, dequantizeAbsmax(q));
    assert.ok(err.snr > 30, `SNR ${err.snr.toFixed(1)} dB should be > 30 dB`);
  });
});
