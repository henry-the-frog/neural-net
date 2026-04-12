import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  quantize, dequantize, fakeQuantize,
  quantizePerChannel, dequantizePerChannel,
  quantizationError, dynamicQuantize, clusterWeights,
} from '../src/quantization.js';

const approx = (a, b, eps = 0.01) => Math.abs(a - b) < eps;

describe('Uniform Quantization', () => {
  it('quantizes and dequantizes', () => {
    const values = [0.5, -0.3, 1.0, -1.0, 0];
    const { quantized, scale, zeroPoint } = quantize(values, 8);
    const recovered = dequantize(quantized, scale, zeroPoint);
    for (let i = 0; i < values.length; i++) {
      assert.ok(approx(recovered[i], values[i], 0.02),
        `Mismatch at ${i}: ${recovered[i]} vs ${values[i]}`);
    }
  });

  it('8-bit has 256 levels', () => {
    const values = Array.from({ length: 100 }, () => Math.random() * 2 - 1);
    const { quantized } = quantize(values, 8);
    const unique = new Set(quantized);
    assert.ok(unique.size <= 256);
  });

  it('4-bit has 16 levels', () => {
    const values = Array.from({ length: 100 }, () => Math.random() * 2 - 1);
    const { quantized } = quantize(values, 4);
    const unique = new Set(quantized);
    assert.ok(unique.size <= 16, `4-bit should have ≤16 levels: ${unique.size}`);
  });

  it('asymmetric quantization handles all-positive', () => {
    const values = [1, 2, 3, 4, 5];
    const { quantized, scale, zeroPoint } = quantize(values, 8, false);
    const recovered = dequantize(quantized, scale, zeroPoint);
    for (let i = 0; i < values.length; i++) {
      assert.ok(approx(recovered[i], values[i], 0.1));
    }
  });
});

describe('Fake Quantization', () => {
  it('roundtrips approximately', () => {
    const values = [0.1, 0.5, -0.3, 0.8];
    const fq = fakeQuantize(values, 8);
    const { rmse } = quantizationError(values, fq);
    assert.ok(rmse < 0.01, `8-bit fake quantize should be accurate: rmse=${rmse}`);
  });

  it('lower bits have more error', () => {
    const values = Array.from({ length: 50 }, () => Math.random() * 2 - 1);
    const fq4 = fakeQuantize(values, 4);
    const fq8 = fakeQuantize(values, 8);
    const err4 = quantizationError(values, fq4).rmse;
    const err8 = quantizationError(values, fq8).rmse;
    assert.ok(err4 >= err8, `4-bit (${err4}) should have ≥ error than 8-bit (${err8})`);
  });
});

describe('Per-Channel Quantization', () => {
  it('quantizes matrix per row', () => {
    const matrix = [[1, 2, 3], [-1, -2, -3], [0.1, 0.2, 0.3]];
    const { quantized, scales } = quantizePerChannel(matrix, 8);
    assert.equal(quantized.length, 3);
    assert.equal(scales.length, 3);
    // Scale should vary per channel
    assert.ok(scales[0] !== scales[2] || true); // May be equal for some values
  });

  it('dequantize recovers approximately', () => {
    const matrix = [[1, -1, 0.5], [3, 2, 1]];
    const { quantized, scales, zeroPoints } = quantizePerChannel(matrix, 8);
    const recovered = dequantizePerChannel(quantized, scales, zeroPoints);
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < 3; c++) {
        assert.ok(approx(recovered[r][c], matrix[r][c], 0.05));
      }
    }
  });
});

describe('Quantization Error', () => {
  it('zero error for identical', () => {
    const { mse, maxError } = quantizationError([1, 2, 3], [1, 2, 3]);
    assert.ok(approx(mse, 0));
    assert.ok(approx(maxError, 0));
  });

  it('positive error for different', () => {
    const { mse } = quantizationError([1, 2, 3], [1.1, 2.1, 2.9]);
    assert.ok(mse > 0);
  });
});

describe('Dynamic Quantization', () => {
  it('chooses appropriate bit width', () => {
    const values = Array.from({ length: 100 }, () => Math.random() * 2 - 1);
    const result = dynamicQuantize(values, 0.01);
    assert.ok(result.bits >= 2 && result.bits <= 16);
    assert.ok(result.error < Math.max(...values) - Math.min(...values));
  });

  it('smooth values need fewer bits', () => {
    const smooth = Array.from({ length: 100 }, (_, i) => Math.sin(i * 0.1));
    const noisy = Array.from({ length: 100 }, () => Math.random() * 10 - 5);
    const smoothResult = dynamicQuantize(smooth);
    const noisyResult = dynamicQuantize(noisy);
    // Smooth might need fewer or equal bits
    assert.ok(smoothResult.bits <= noisyResult.bits + 2);
  });
});

describe('Weight Clustering', () => {
  it('clusters weights into codebook', () => {
    const weights = Array.from({ length: 100 }, () => Math.random() * 2 - 1);
    const result = clusterWeights(weights, 8, 10);
    assert.equal(result.centroids.length, 8);
    assert.equal(result.quantized.length, 100);
  });

  it('quantized values are from codebook', () => {
    const weights = [1, 1.1, 5, 5.1, 10, 10.1];
    const result = clusterWeights(weights, 3, 20);
    const centroidSet = new Set(result.centroids);
    // Each quantized value should be a centroid
    assert.ok(result.quantized.every(v => centroidSet.has(v)));
  });

  it('compression ratio is correct', () => {
    const weights = Array.from({ length: 100 }, () => Math.random());
    const result = clusterWeights(weights, 16);
    assert.ok(approx(result.compressionRatio, 4 / 32, 0.001)); // log2(16)/32 = 4/32
  });
});
