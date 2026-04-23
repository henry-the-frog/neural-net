// loss-functions.js — Comprehensive Loss Functions
// All major loss functions used in deep learning.

/**
 * Mean Squared Error (MSE): regression
 */
export function mse(predicted, target) {
  let sum = 0;
  for (let i = 0; i < predicted.length; i++) {
    sum += (predicted[i] - target[i]) ** 2;
  }
  return sum / predicted.length;
}

/**
 * Mean Absolute Error (MAE / L1 Loss)
 */
export function mae(predicted, target) {
  let sum = 0;
  for (let i = 0; i < predicted.length; i++) sum += Math.abs(predicted[i] - target[i]);
  return sum / predicted.length;
}

/**
 * Huber Loss: MSE when error is small, MAE when error is large.
 * Robust to outliers.
 */
export function huber(predicted, target, delta = 1.0) {
  let sum = 0;
  for (let i = 0; i < predicted.length; i++) {
    const err = Math.abs(predicted[i] - target[i]);
    sum += err <= delta ? 0.5 * err * err : delta * (err - 0.5 * delta);
  }
  return sum / predicted.length;
}

/**
 * Binary Cross-Entropy Loss
 */
export function binaryCrossEntropy(predicted, target) {
  let sum = 0;
  for (let i = 0; i < predicted.length; i++) {
    const p = Math.max(1e-10, Math.min(1 - 1e-10, predicted[i]));
    sum -= target[i] * Math.log(p) + (1 - target[i]) * Math.log(1 - p);
  }
  return sum / predicted.length;
}

/**
 * Categorical Cross-Entropy with logits
 */
export function crossEntropy(logits, targetIdx) {
  const max = Math.max(...logits);
  let sumExp = 0;
  for (const l of logits) sumExp += Math.exp(l - max);
  const logSumExp = Math.log(sumExp) + max;
  return -(logits[targetIdx] - logSumExp);
}

/**
 * Focal Loss (Lin et al., 2017): addresses class imbalance.
 * FL = -α(1-p)^γ * log(p)
 */
export function focalLoss(predicted, target, gamma = 2.0, alpha = 0.25) {
  let sum = 0;
  for (let i = 0; i < predicted.length; i++) {
    const p = Math.max(1e-10, Math.min(1 - 1e-10, predicted[i]));
    const pt = target[i] === 1 ? p : 1 - p;
    const alphat = target[i] === 1 ? alpha : 1 - alpha;
    sum -= alphat * Math.pow(1 - pt, gamma) * Math.log(pt);
  }
  return sum / predicted.length;
}

/**
 * Dice Loss: overlap-based, used in segmentation.
 * 1 - 2*|A∩B| / (|A| + |B|)
 */
export function diceLoss(predicted, target) {
  let intersection = 0, sumP = 0, sumT = 0;
  for (let i = 0; i < predicted.length; i++) {
    intersection += predicted[i] * target[i];
    sumP += predicted[i];
    sumT += target[i];
  }
  return 1 - (2 * intersection + 1e-6) / (sumP + sumT + 1e-6);
}

/**
 * Hinge Loss: used in SVMs and some classifiers.
 * L = max(0, 1 - y * ŷ) where y ∈ {-1, +1}
 */
export function hingeLoss(predicted, target) {
  let sum = 0;
  for (let i = 0; i < predicted.length; i++) {
    sum += Math.max(0, 1 - target[i] * predicted[i]);
  }
  return sum / predicted.length;
}
