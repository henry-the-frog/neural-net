// loss.js — Loss functions

import { Matrix } from './matrix.js';

// Mean Squared Error
export const mse = {
  name: 'mse',
  compute(predicted, target) {
    const diff = predicted.sub(target);
    return diff.mul(diff).sum() / (2 * predicted.rows);
  },
  gradient(predicted, target) {
    return predicted.sub(target);
  }
};

// Cross-Entropy (for softmax output)
export const crossEntropy = {
  name: 'cross_entropy',
  compute(predicted, target) {
    let loss = 0;
    const eps = 1e-15;
    for (let i = 0; i < predicted.rows; i++) {
      for (let j = 0; j < predicted.cols; j++) {
        const p = Math.max(eps, Math.min(1 - eps, predicted.get(i, j)));
        loss -= target.get(i, j) * Math.log(p);
      }
    }
    return loss / predicted.rows;
  },
  gradient(predicted, target) {
    // For softmax + cross-entropy: gradient = predicted - target
    return predicted.sub(target);
  }
};

export function getLoss(name) {
  const losses = { mse, cross_entropy: crossEntropy };
  return losses[name] || mse;
}
