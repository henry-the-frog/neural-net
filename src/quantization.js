// quantization.js — Neural Network Quantization
// Reduce precision for efficient inference
// Implements: uniform quantization, per-channel, symmetric/asymmetric, fake quantization

// ===== Uniform Quantization =====
// Map floating point values to fixed-point integers

export function quantize(values, bits = 8, symmetric = true) {
  const qMin = symmetric ? -(1 << (bits - 1)) : 0;
  const qMax = symmetric ? (1 << (bits - 1)) - 1 : (1 << bits) - 1;

  const max = Math.max(...values.map(Math.abs));
  const scale = max > 0 ? (symmetric ? max / qMax : max / (qMax - qMin)) : 1;
  const zeroPoint = symmetric ? 0 : Math.round(-Math.min(...values) / scale);

  const quantized = values.map(v => {
    const q = Math.round(v / scale + zeroPoint);
    return Math.max(qMin, Math.min(qMax, q));
  });

  return { quantized, scale, zeroPoint, bits };
}

export function dequantize(quantized, scale, zeroPoint = 0) {
  return quantized.map(q => (q - zeroPoint) * scale);
}

// ===== Fake Quantization (for quantization-aware training) =====
// Simulates quantization during forward, but keeps gradients continuous
export function fakeQuantize(values, bits = 8) {
  const { quantized, scale, zeroPoint } = quantize(values, bits);
  return dequantize(quantized, scale, zeroPoint);
}

// ===== Per-Channel Quantization =====
// Different scale/zero-point per output channel (for weight matrices)
export function quantizePerChannel(matrix, bits = 8) {
  // matrix: 2D array [rows][cols]
  const channels = matrix.length;
  const scales = [];
  const zeroPoints = [];
  const quantized = [];

  for (let c = 0; c < channels; c++) {
    const { quantized: q, scale, zeroPoint } = quantize(matrix[c], bits);
    quantized.push(q);
    scales.push(scale);
    zeroPoints.push(zeroPoint);
  }

  return { quantized, scales, zeroPoints, bits };
}

export function dequantizePerChannel(quantized, scales, zeroPoints) {
  return quantized.map((row, c) => dequantize(row, scales[c], zeroPoints[c]));
}

// ===== Quantization Error Analysis =====
export function quantizationError(original, quantized) {
  let mse = 0, maxErr = 0;
  for (let i = 0; i < original.length; i++) {
    const err = Math.abs(original[i] - quantized[i]);
    mse += err * err;
    maxErr = Math.max(maxErr, err);
  }
  return { mse: mse / original.length, maxError: maxErr, rmse: Math.sqrt(mse / original.length) };
}

// ===== Dynamic Range Quantization =====
// Choose bit width based on value distribution
export function dynamicQuantize(values, targetError = 0.01) {
  for (let bits = 2; bits <= 16; bits++) {
    const fq = fakeQuantize(values, bits);
    const { rmse } = quantizationError(values, fq);
    const range = Math.max(...values) - Math.min(...values);
    if (range > 0 && rmse / range < targetError) {
      return { bits, quantized: fq, error: rmse };
    }
  }
  return { bits: 16, quantized: fakeQuantize(values, 16), error: 0 };
}

// ===== Weight Clustering (K-means quantization) =====
export function clusterWeights(weights, numClusters = 16, iterations = 20) {
  // Initialize centroids
  const sorted = [...weights].sort((a, b) => a - b);
  const centroids = Array.from({ length: numClusters }, (_, i) =>
    sorted[Math.floor(i * sorted.length / numClusters)]
  );

  let assignments = new Array(weights.length).fill(0);

  for (let iter = 0; iter < iterations; iter++) {
    // Assign each weight to nearest centroid
    assignments = weights.map(w => {
      let minDist = Infinity, minIdx = 0;
      for (let c = 0; c < numClusters; c++) {
        const dist = Math.abs(w - centroids[c]);
        if (dist < minDist) { minDist = dist; minIdx = c; }
      }
      return minIdx;
    });

    // Update centroids
    for (let c = 0; c < numClusters; c++) {
      const members = weights.filter((_, i) => assignments[i] === c);
      if (members.length > 0) {
        centroids[c] = members.reduce((a, b) => a + b, 0) / members.length;
      }
    }
  }

  return {
    centroids,
    assignments,
    quantized: assignments.map(a => centroids[a]),
    codebook: centroids,
    compressionRatio: Math.log2(numClusters) / 32, // vs 32-bit float
  };
}

// ===== Mixed-Precision Strategy =====
// Recommend bit widths per layer based on sensitivity
export function analyzeSensitivity(layers, testFn, bits = [2, 4, 8, 16]) {
  const results = [];

  for (let l = 0; l < layers.length; l++) {
    const layerResults = [];
    const originalWeights = layers[l].flat ? [...layers[l]] : layers[l].map(r => [...r]);

    for (const b of bits) {
      const fq = Array.isArray(originalWeights[0])
        ? originalWeights.map(row => fakeQuantize(row, b))
        : fakeQuantize(originalWeights, b);
      const error = testFn(l, fq);
      layerResults.push({ bits: b, error });
    }

    results.push({
      layer: l,
      sensitivity: layerResults,
      recommendedBits: layerResults.find(r => r.error < 0.05)?.bits || 16,
    });
  }

  return results;
}
