// data-augmentation.js — Basic data augmentation for training

export function addGaussianNoise(data, std = 0.1) {
  return data.map(x => x + std * (Math.random() * 2 - 1));
}

// Alias for test compatibility
export const addNoise = addGaussianNoise;

export function dropout(data, rate = 0.1) {
  return data.map(x => Math.random() < rate ? 0 : x);
}

export function mixup(x1, y1, x2, y2, alpha = 0.2) {
  const lambda = alpha > 0 ? betaSample(alpha, alpha) : 1;
  const x = x1.map((v, i) => lambda * v + (1 - lambda) * x2[i]);
  const y = lambda * y1 + (1 - lambda) * y2;
  return { x, y };
}

export function cutmix(x1, x2, alpha = 1.0) {
  const lambda = Math.random();
  const cutLen = Math.floor(x1.length * (1 - lambda));
  const start = Math.floor(Math.random() * (x1.length - cutLen));
  const mixed = [...x1];
  for (let i = start; i < start + cutLen; i++) mixed[i] = x2[i];
  return { mixed, lambda };
}

/**
 * Horizontal flip a 2D array (matrix rows or flat image).
 */
export function flip(data) {
  if (Array.isArray(data[0])) {
    return data.map(row => [...row].reverse());
  }
  return [...data].reverse();
}

/**
 * Rotate 2D square data by 90 degrees CW.
 */
export function rotate90(data) {
  if (!Array.isArray(data[0])) return [...data]; // 1D: no-op
  const n = data.length;
  const m = data[0].length;
  const result = Array.from({ length: m }, () => new Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      result[j][n - 1 - i] = data[i][j];
    }
  }
  return result;
}

/**
 * Random crop from a 1D array.
 */
export function randomCrop(data, cropSize) {
  const start = Math.floor(Math.random() * (data.length - cropSize + 1));
  return data.slice(start, start + cropSize);
}

/**
 * Normalize to [0, 1] range.
 */
export function normalize(data) {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  return data.map(x => (x - min) / range);
}

/**
 * Standardize to zero mean, unit variance.
 */
export function standardize(data) {
  const n = data.length;
  const mean = data.reduce((a, b) => a + b, 0) / n;
  const variance = data.reduce((a, x) => a + (x - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance) || 1;
  return data.map(x => (x - mean) / std);
}

function betaSample(a, b) {
  // Simplified: use uniform approximation for small alpha
  return Math.random();
}
