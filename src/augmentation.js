// augmentation.js — Data augmentation for training

import { Matrix } from './matrix.js';

/**
 * Add Gaussian noise to data
 */
export function addNoise(data, stddev = 0.1) {
  return data.map(v => v + (Math.random() * 2 - 1) * stddev);
}

/**
 * Randomly flip horizontal (for image-like data arranged as rows of pixels)
 * Assumes data is [batch, width * height * channels]
 */
export function randomFlipH(data, width, height, channels = 1) {
  const result = new Matrix(data.rows, data.cols);
  for (let b = 0; b < data.rows; b++) {
    const flip = Math.random() < 0.5;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const srcX = flip ? (width - 1 - x) : x;
        for (let c = 0; c < channels; c++) {
          const srcIdx = (y * width + srcX) * channels + c;
          const dstIdx = (y * width + x) * channels + c;
          result.set(b, dstIdx, data.get(b, srcIdx));
        }
      }
    }
  }
  return result;
}

/**
 * Random crop from padded image
 */
export function randomCrop(data, width, height, channels, padSize = 2) {
  const paddedW = width + 2 * padSize;
  const paddedH = height + 2 * padSize;
  const result = new Matrix(data.rows, data.cols);
  
  for (let b = 0; b < data.rows; b++) {
    const offX = Math.floor(Math.random() * (2 * padSize + 1));
    const offY = Math.floor(Math.random() * (2 * padSize + 1));
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const srcY = y + offY - padSize;
        const srcX = x + offX - padSize;
        for (let c = 0; c < channels; c++) {
          const dstIdx = (y * width + x) * channels + c;
          if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width) {
            const srcIdx = (srcY * width + srcX) * channels + c;
            result.set(b, dstIdx, data.get(b, srcIdx));
          } else {
            result.set(b, dstIdx, 0); // Zero padding
          }
        }
      }
    }
  }
  return result;
}

/**
 * Mixup augmentation (Zhang et al. 2017)
 * Linearly interpolates between pairs of examples
 */
export function mixup(inputs, targets, alpha = 0.2) {
  const n = inputs.rows;
  const newInputs = new Matrix(n, inputs.cols);
  const newTargets = new Matrix(n, targets.cols);
  
  for (let i = 0; i < n; i++) {
    // Beta distribution approximated by clipping
    const lambda = Math.max(alpha, Math.min(1 - alpha, 
      alpha > 0 ? betaSample(alpha, alpha) : 1));
    const j = Math.floor(Math.random() * n);
    
    for (let k = 0; k < inputs.cols; k++) {
      newInputs.set(i, k, lambda * inputs.get(i, k) + (1 - lambda) * inputs.get(j, k));
    }
    for (let k = 0; k < targets.cols; k++) {
      newTargets.set(i, k, lambda * targets.get(i, k) + (1 - lambda) * targets.get(j, k));
    }
  }
  
  return { inputs: newInputs, targets: newTargets };
}

// Simple beta distribution sampling (approximation)
function betaSample(a, b) {
  // Use inverse transform of uniform
  const u = Math.random();
  const v = Math.random();
  const x = Math.pow(u, 1/a);
  const y = Math.pow(v, 1/b);
  return x / (x + y);
}

/**
 * Cutout augmentation — randomly zero out a square region
 */
export function cutout(data, width, height, channels, cutSize = 4) {
  const result = new Matrix(data.rows, data.cols);
  // Copy data
  for (let i = 0; i < data.rows; i++)
    for (let j = 0; j < data.cols; j++)
      result.set(i, j, data.get(i, j));
  
  for (let b = 0; b < data.rows; b++) {
    const cx = Math.floor(Math.random() * width);
    const cy = Math.floor(Math.random() * height);
    const half = Math.floor(cutSize / 2);
    
    for (let y = Math.max(0, cy - half); y < Math.min(height, cy + half); y++) {
      for (let x = Math.max(0, cx - half); x < Math.min(width, cx + half); x++) {
        for (let c = 0; c < channels; c++) {
          result.set(b, (y * width + x) * channels + c, 0);
        }
      }
    }
  }
  return result;
}

/**
 * Random brightness/contrast adjustment for image data
 */
export function randomBrightnessContrast(data, brightnessRange = 0.1, contrastRange = 0.1) {
  const result = new Matrix(data.rows, data.cols);
  for (let i = 0; i < data.rows; i++) {
    const brightness = (Math.random() * 2 - 1) * brightnessRange;
    const contrast = 1 + (Math.random() * 2 - 1) * contrastRange;
    for (let j = 0; j < data.cols; j++) {
      const v = data.get(i, j);
      result.set(i, j, Math.max(0, Math.min(1, (v - 0.5) * contrast + 0.5 + brightness)));
    }
  }
  return result;
}

/**
 * Compose multiple augmentations
 */
export function compose(...augFns) {
  return (data, ...args) => {
    let result = data;
    for (const fn of augFns) {
      result = fn(result, ...args);
    }
    return result;
  };
}
