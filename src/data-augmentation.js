// Data augmentation module - bridges to augmentation.js and adds extras
import { Matrix } from './matrix.js';
export { addNoise, randomCrop } from './augmentation.js';

export function mixup(x1, x2, alpha = 0.2) {
  const lambda = Math.random() * alpha + (1 - alpha) * 0.5;
  const result = new Matrix(x1.rows, x1.cols);
  for (let i = 0; i < result.data.length; i++) {
    result.data[i] = lambda * x1.data[i] + (1 - lambda) * x2.data[i];
  }
  return { x: result, lambda };
}

export function dropout(data, rate = 0.5) {
  if (data instanceof Matrix) {
    const result = new Matrix(data.rows, data.cols);
    for (let i = 0; i < data.data.length; i++) {
      result.data[i] = Math.random() > rate ? data.data[i] / (1 - rate) : 0;
    }
    return result;
  }
  return data.map(v => Math.random() > rate ? v / (1 - rate) : 0);
}

export function cutmix(input1, input2, target1, target2, alpha = 1.0) {
  const lambda = Math.random() * alpha;
  const rows = input1.rows, cols = input1.cols;
  const cutRow = Math.floor(Math.random() * rows);
  const cutCol = Math.floor(Math.random() * cols);
  const cutH = Math.floor(rows * Math.sqrt(1 - lambda));
  const cutW = Math.floor(cols * Math.sqrt(1 - lambda));
  
  const result = new Matrix(rows, cols);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const inCut = r >= cutRow && r < cutRow + cutH && c >= cutCol && c < cutCol + cutW;
      result.set(r, c, inCut ? input2.get(r % input2.rows, c % input2.cols) : input1.get(r, c));
    }
  }
  
  const area = cutH * cutW;
  const total = rows * cols;
  const lam = 1 - area / total;
  const mixedTarget = target1 instanceof Matrix 
    ? target1.map(v => v * lam).add(target2.map(v => v * (1 - lam)))
    : target1 * lam + target2 * (1 - lam);
  
  return { x: result, y: mixedTarget, lambda: lam };
}

export function flip(data, axis = 'horizontal') {
  if (!(data instanceof Matrix)) return data;
  const result = new Matrix(data.rows, data.cols);
  for (let r = 0; r < data.rows; r++) {
    for (let c = 0; c < data.cols; c++) {
      if (axis === 'horizontal') {
        result.set(r, data.cols - 1 - c, data.get(r, c));
      } else {
        result.set(data.rows - 1 - r, c, data.get(r, c));
      }
    }
  }
  return result;
}

export function rotate90(data, times = 1) {
  if (!(data instanceof Matrix)) return data;
  let result = data;
  for (let t = 0; t < (times % 4); t++) {
    const rotated = new Matrix(result.cols, result.rows);
    for (let r = 0; r < result.rows; r++) {
      for (let c = 0; c < result.cols; c++) {
        rotated.set(c, result.rows - 1 - r, result.get(r, c));
      }
    }
    result = rotated;
  }
  return result;
}

export function normalize(data, min = 0, max = 1) {
  if (!(data instanceof Matrix)) return data;
  const dataMin = Math.min(...data.data);
  const dataMax = Math.max(...data.data);
  const range = dataMax - dataMin || 1;
  const result = new Matrix(data.rows, data.cols);
  for (let i = 0; i < data.data.length; i++) {
    result.data[i] = ((data.data[i] - dataMin) / range) * (max - min) + min;
  }
  return result;
}

export function standardize(data) {
  if (!(data instanceof Matrix)) return data;
  const n = data.data.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += data.data[i];
  const mean = sum / n;
  let sumSq = 0;
  for (let i = 0; i < n; i++) sumSq += (data.data[i] - mean) ** 2;
  const std = Math.sqrt(sumSq / n) || 1;
  const result = new Matrix(data.rows, data.cols);
  for (let i = 0; i < n; i++) result.data[i] = (data.data[i] - mean) / std;
  return result;
}
