// quantization.test.js — Tests for weight quantization
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  quantizeAbsmaxINT8, dequantizeINT8,
  quantizeGroupINT4, dequantizeINT4,
  quantizationError, compressionRatio
} from './quantization.js';
import { Matrix } from './matrix.js';

describe('Quantization', () => {
  describe('INT8 absmax quantization', () => {
    it('roundtrip preserves approximate values', () => {
      const mat = Matrix.random(4, 4);
      const q = quantizeAbsmaxINT8(mat);
      const deq = dequantizeINT8(q);
      const err = quantizationError(mat, deq);
      assert.ok(err < 0.05, `Error should be small: ${err}`);
    });

    it('quantized values are in INT8 range', () => {
      const mat = Matrix.random(3, 3);
      const q = quantizeAbsmaxINT8(mat);
      for (const v of q.quantized) {
        assert.ok(v >= -127 && v <= 127, `Value ${v} out of range`);
      }
    });

    it('zero matrix quantizes to zeros', () => {
      const mat = Matrix.zeros(2, 2);
      const q = quantizeAbsmaxINT8(mat);
      for (const v of q.quantized) assert.equal(v, 0);
    });

    it('preserves relative magnitudes', () => {
      const mat = new Matrix(1, 4);
      mat.set(0, 0, 0.1); mat.set(0, 1, 0.5); mat.set(0, 2, -0.3); mat.set(0, 3, 1.0);
      const q = quantizeAbsmaxINT8(mat);
      const deq = dequantizeINT8(q);
      assert.ok(deq.get(0, 3) > deq.get(0, 1), 'Largest value should stay largest');
      assert.ok(deq.get(0, 1) > deq.get(0, 0), 'Order preserved');
    });
  });

  describe('INT4 group quantization', () => {
    it('roundtrip produces reasonable approximation', () => {
      const mat = Matrix.random(8, 8);
      const q = quantizeGroupINT4(mat, 16);
      const deq = dequantizeINT4(q);
      const err = quantizationError(mat, deq);
      console.log(`  INT4 error: ${err.toFixed(6)}`);
      assert.ok(err < 0.2, `Error should be manageable: ${err}`);
    });

    it('group size affects accuracy', () => {
      const mat = Matrix.random(16, 16);
      
      const q_small = quantizeGroupINT4(mat, 8);   // smaller groups = more scales = better
      const q_large = quantizeGroupINT4(mat, 128);  // larger groups = fewer scales = worse
      
      const err_small = quantizationError(mat, dequantizeINT4(q_small));
      const err_large = quantizationError(mat, dequantizeINT4(q_large));
      
      console.log(`  Group 8: ${err_small.toFixed(6)}, Group 128: ${err_large.toFixed(6)}`);
      assert.ok(err_small <= err_large + 0.01, 'Smaller groups should be more accurate');
    });

    it('handles odd-sized matrices', () => {
      const mat = Matrix.random(3, 5); // 15 elements, not divisible by 2
      const q = quantizeGroupINT4(mat, 8);
      const deq = dequantizeINT4(q);
      assert.equal(deq.rows, 3);
      assert.equal(deq.cols, 5);
    });
  });

  describe('compression ratio', () => {
    it('INT8 gives ~8x compression vs FP64', () => {
      const mat = Matrix.random(100, 100);
      const ratio = compressionRatio(mat, 8);
      console.log(`  INT8 compression: ${ratio.toFixed(1)}x`);
      assert.ok(ratio > 7, `Expected ~8x, got ${ratio}x`);
    });

    it('INT4 gives ~16x compression vs FP64', () => {
      const mat = Matrix.random(100, 100);
      const ratio = compressionRatio(mat, 4);
      console.log(`  INT4 compression: ${ratio.toFixed(1)}x`);
      assert.ok(ratio > 12, `Expected ~16x, got ${ratio}x`);
    });
  });

  describe('accuracy comparison', () => {
    it('INT8 is more accurate than INT4', () => {
      const mat = Matrix.random(32, 32);
      
      const q8 = quantizeAbsmaxINT8(mat);
      const q4 = quantizeGroupINT4(mat, 32);
      
      const err8 = quantizationError(mat, dequantizeINT8(q8));
      const err4 = quantizationError(mat, dequantizeINT4(q4));
      
      console.log(`  INT8 error: ${err8.toFixed(6)}, INT4 error: ${err4.toFixed(6)}`);
      assert.ok(err8 < err4, 'INT8 should be more accurate than INT4');
    });
  });
});
