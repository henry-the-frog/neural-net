// lr-finder.js — Learning Rate Finder (Leslie Smith, 2015)
// "Cyclical Learning Rates for Training Neural Networks"
//
// Exponentially increases learning rate from minLR to maxLR over one pass
// of the training data, recording loss at each step. The optimal LR is
// where loss decreases fastest (steepest negative gradient), typically
// 1/10th of the LR where loss starts increasing.

import { Matrix } from './matrix.js';

/**
 * Find optimal learning rate by sweeping from minLR to maxLR.
 * 
 * @param {Network} network - The network to test
 * @param {Object} data - { inputs: Matrix, targets: Matrix }
 * @param {Object} options
 * @param {number} options.minLR - Starting LR (default 1e-7)
 * @param {number} options.maxLR - Ending LR (default 1)
 * @param {number} options.steps - Number of LR steps (default 100)
 * @param {number} options.batchSize - Batch size (default 32)
 * @param {number} options.smoothing - Exponential smoothing factor (default 0.05)
 * @param {number} options.divergeThreshold - Stop if loss > best * this (default 4)
 * @returns {Object} { lrs, losses, smoothedLosses, suggestedLR, bestLR }
 */
export function findLR(network, data, {
  minLR = 1e-7,
  maxLR = 1,
  steps = 100,
  batchSize = 32,
  smoothing = 0.05,
  divergeThreshold = 4,
} = {}) {
  const { inputs, targets } = data;
  const n = inputs.rows;

  // Save model state
  const savedState = network.toJSON();

  const lrMult = (maxLR / minLR) ** (1 / steps);
  const lrs = [];
  const losses = [];
  const smoothedLosses = [];
  let bestLoss = Infinity;
  let smoothedLoss = 0;
  let lr = minLR;

  // Create shuffled indices
  const indices = Array.from({ length: n }, (_, i) => i);

  for (let step = 0; step < steps; step++) {
    // Get a batch
    const batchStart = (step * batchSize) % n;
    // Wrap around and reshuffle if needed
    if (batchStart === 0 && step > 0) {
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
    }

    const batchIndices = [];
    for (let i = 0; i < batchSize && i < n; i++) {
      batchIndices.push(indices[(batchStart + i) % n]);
    }

    const batchInputs = new Matrix(batchIndices.length, inputs.cols);
    const batchTargets = new Matrix(batchIndices.length, targets.cols);
    for (let i = 0; i < batchIndices.length; i++) {
      const idx = batchIndices[i];
      for (let j = 0; j < inputs.cols; j++) batchInputs.set(i, j, inputs.get(idx, j));
      for (let j = 0; j < targets.cols; j++) batchTargets.set(i, j, targets.get(idx, j));
    }

    // Train one batch at current LR
    const loss = network.trainBatch(batchInputs, batchTargets, lr);

    // Record
    lrs.push(lr);
    losses.push(loss);

    // Exponential smoothing
    if (step === 0) {
      smoothedLoss = loss;
    } else {
      smoothedLoss = smoothing * loss + (1 - smoothing) * smoothedLoss;
    }
    // Bias correction (like Adam)
    const corrected = smoothedLoss / (1 - (1 - smoothing) ** (step + 1));
    smoothedLosses.push(corrected);

    if (corrected < bestLoss) bestLoss = corrected;

    // Stop if loss diverges
    if (step > 10 && corrected > bestLoss * divergeThreshold) break;

    // Increase LR
    lr *= lrMult;
  }

  // Restore model state
  const NetworkClass = network.constructor;
  const restored = NetworkClass.fromJSON(savedState);
  // Copy weights back
  for (let i = 0; i < network.layers.length; i++) {
    if (restored.layers[i]) {
      const src = restored.layers[i];
      const dst = network.layers[i];
      if (src.weights && dst.weights) {
        dst.weights = src.weights;
        dst.biases = src.biases;
      }
    }
  }

  // Find suggested LR: steepest descent in smoothed loss
  const suggestedLR = findSteepestDescent(lrs, smoothedLosses);
  // bestLR: the LR where minimum smoothed loss occurred
  const bestIdx = smoothedLosses.indexOf(Math.min(...smoothedLosses));
  const bestLR = lrs[bestIdx];

  return {
    lrs,
    losses,
    smoothedLosses,
    suggestedLR,
    bestLR,
    steps: lrs.length,
  };
}

/**
 * Find the LR with steepest descent (most negative gradient) in smoothed loss.
 * Returns the LR at ~1/10th of the divergence point (practical rule of thumb).
 */
function findSteepestDescent(lrs, smoothedLosses) {
  if (lrs.length < 3) return lrs[0];

  let bestGrad = 0;
  let bestIdx = 0;

  for (let i = 1; i < smoothedLosses.length - 1; i++) {
    // Gradient in log-space
    const grad = (smoothedLosses[i + 1] - smoothedLosses[i - 1]) / (Math.log(lrs[i + 1]) - Math.log(lrs[i - 1]));
    if (grad < bestGrad) {
      bestGrad = grad;
      bestIdx = i;
    }
  }

  return lrs[bestIdx];
}

/**
 * Format LR finder results as a text chart.
 */
export function formatLRFinderResults(results, { width = 60, height = 15 } = {}) {
  const { lrs, smoothedLosses, suggestedLR, bestLR } = results;
  const lines = [];
  
  lines.push('Learning Rate Finder Results');
  lines.push('─'.repeat(width));
  lines.push(`Steps: ${results.steps}`);
  lines.push(`Suggested LR (steepest descent): ${suggestedLR.toExponential(2)}`);
  lines.push(`Best LR (min loss): ${bestLR.toExponential(2)}`);
  lines.push(`LR range: ${lrs[0].toExponential(2)} → ${lrs[lrs.length - 1].toExponential(2)}`);
  lines.push(`Loss range: ${Math.min(...smoothedLosses).toFixed(4)} → ${Math.max(...smoothedLosses).toFixed(4)}`);
  lines.push('─'.repeat(width));

  // Simple text chart
  const minLoss = Math.min(...smoothedLosses);
  const maxLoss = Math.max(...smoothedLosses);
  const lossRange = maxLoss - minLoss || 1;

  const bucketSize = Math.ceil(lrs.length / width);
  const bucketedLoss = [];
  for (let i = 0; i < lrs.length; i += bucketSize) {
    const slice = smoothedLosses.slice(i, i + bucketSize);
    bucketedLoss.push(slice.reduce((a, b) => a + b) / slice.length);
  }

  for (let row = 0; row < height; row++) {
    const threshold = maxLoss - (row / (height - 1)) * lossRange;
    let line = '';
    for (let col = 0; col < bucketedLoss.length; col++) {
      line += bucketedLoss[col] >= threshold ? '█' : ' ';
    }
    const lossLabel = threshold.toFixed(3).padStart(8);
    lines.push(`${lossLabel} |${line}`);
  }

  return lines.join('\n');
}
