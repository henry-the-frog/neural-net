// data-augmentation.js — Basic data augmentation for training
import { Matrix } from './matrix.js';

// Helper: check if something is a Matrix
function isMatrix(x) { return x && typeof x.rows === 'number' && x.data; }

// Helper: map over data whether it's an array or Matrix
function mapData(data, fn) {
  if (isMatrix(data)) {
    const result = new Matrix(data.rows, data.cols);
    for (let i = 0; i < data.data.length; i++) {
      result.data[i] = fn(data.data[i], i);
    }
    return result;
  }
  return data.map(fn);
}

export function addGaussianNoise(data, std = 0.1) {
  return mapData(data, x => x + std * (Math.random() * 2 - 1));
}

// Alias for test compatibility
export const addNoise = addGaussianNoise;

export function dropout(data, rate = 0.1) {
  return mapData(data, x => Math.random() < rate ? 0 : x);
}

export function mixup(x1, y1, x2, y2, alpha) {
  // Support both mixup(x1, y1, x2, y2, alpha) and mixup(x1, x2, alpha)
  let actualX1, actualX2, actualAlpha;
  if (isMatrix(y1) || (typeof y1 !== 'number' && !Array.isArray(y1) && y1 !== undefined && y1 !== null)) {
    // mixup(x1, x2, alpha) — no separate y values
    actualX1 = x1;
    actualX2 = y1;
    actualAlpha = x2 || 0.5;
    const lambda = actualAlpha > 0 ? betaSample(actualAlpha, actualAlpha) : 1;
    if (isMatrix(actualX1)) {
      const result = new Matrix(actualX1.rows, actualX1.cols);
      for (let i = 0; i < actualX1.data.length; i++) {
        result.data[i] = lambda * actualX1.data[i] + (1 - lambda) * actualX2.data[i];
      }
      return { x: result, lambda };
    }
    const x = actualX1.map((v, i) => lambda * v + (1 - lambda) * actualX2[i]);
    return { x, lambda };
  }
  
  // mixup(x1, y1, x2, y2, alpha)
  const lambda = (alpha || 0.2) > 0 ? betaSample(alpha || 0.2, alpha || 0.2) : 1;
  if (isMatrix(x1)) {
    const result = new Matrix(x1.rows, x1.cols);
    for (let i = 0; i < x1.data.length; i++) {
      result.data[i] = lambda * x1.data[i] + (1 - lambda) * x2.data[i];
    }
    const y = lambda * y1 + (1 - lambda) * y2;
    return { x: result, y };
  }
  const x = x1.map((v, i) => lambda * v + (1 - lambda) * x2[i]);
  const y = lambda * y1 + (1 - lambda) * y2;
  return { x, y };
}

export function cutmix(x1, x2, alpha = 1.0) {
  const d1 = isMatrix(x1) ? x1.data : x1;
  const d2 = isMatrix(x2) ? x2.data : x2;
  const lambda = Math.random();
  const cutLen = Math.floor(d1.length * (1 - lambda));
  const start = Math.floor(Math.random() * (d1.length - cutLen));
  const mixed = new Float64Array(d1);
  for (let i = start; i < start + cutLen; i++) mixed[i] = d2[i];
  if (isMatrix(x1)) {
    return { mixed: new Matrix(x1.rows, x1.cols, mixed), lambda };
  }
  return { mixed: Array.from(mixed), lambda };
}

/**
 * Horizontal flip a 2D array (matrix rows or flat image).
 */
export function flip(data) {
  if (isMatrix(data)) {
    const result = new Matrix(data.rows, data.cols);
    for (let r = 0; r < data.rows; r++) {
      for (let c = 0; c < data.cols; c++) {
        result.set(r, data.cols - 1 - c, data.get(r, c));
      }
    }
    return result;
  }
  if (Array.isArray(data[0])) {
    return data.map(row => [...row].reverse());
  }
  return [...data].reverse();
}

/**
 * Rotate 2D square data by 90 degrees CW.
 */
export function rotate90(data) {
  if (isMatrix(data)) {
    const result = new Matrix(data.cols, data.rows);
    for (let i = 0; i < data.rows; i++) {
      for (let j = 0; j < data.cols; j++) {
        result.set(j, data.rows - 1 - i, data.get(i, j));
      }
    }
    return result;
  }
  if (!Array.isArray(data[0])) return [...data];
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
 * Random crop from a 1D array or Matrix.
 */
export function randomCrop(data, cropSize) {
  if (isMatrix(data)) {
    const totalLen = data.data.length;
    const start = Math.floor(Math.random() * (totalLen - cropSize + 1));
    const cropped = new Float64Array(cropSize);
    for (let i = 0; i < cropSize; i++) cropped[i] = data.data[start + i];
    return new Matrix(1, cropSize, cropped);
  }
  const start = Math.floor(Math.random() * (data.length - cropSize + 1));
  return data.slice(start, start + cropSize);
}

/**
 * Normalize to [0, 1] range.
 */
export function normalize(data) {
  const arr = isMatrix(data) ? data.data : data;
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] < min) min = arr[i];
    if (arr[i] > max) max = arr[i];
  }
  const range = max - min || 1;
  if (isMatrix(data)) {
    const result = new Matrix(data.rows, data.cols);
    for (let i = 0; i < arr.length; i++) {
      result.data[i] = (arr[i] - min) / range;
    }
    return result;
  }
  return data.map(x => (x - min) / range);
}

/**
 * Standardize to zero mean, unit variance.
 */
export function standardize(data) {
  const arr = isMatrix(data) ? data.data : data;
  const n = arr.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += arr[i];
  const mean = sum / n;
  let varSum = 0;
  for (let i = 0; i < n; i++) varSum += (arr[i] - mean) ** 2;
  const std = Math.sqrt(varSum / n) || 1;
  if (isMatrix(data)) {
    const result = new Matrix(data.rows, data.cols);
    for (let i = 0; i < n; i++) result.data[i] = (arr[i] - mean) / std;
    return result;
  }
  return data.map(x => (x - mean) / std);
}

function betaSample(a, b) {
  // Simplified: use uniform approximation for small alpha
  return Math.random();
}
