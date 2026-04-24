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

// Binary Cross-Entropy (for sigmoid output, binary classification)
export const binaryCrossEntropy = {
  name: 'binary_cross_entropy',
  compute(predicted, target) {
    let loss = 0;
    const eps = 1e-15;
    for (let i = 0; i < predicted.rows; i++) {
      for (let j = 0; j < predicted.cols; j++) {
        const p = Math.max(eps, Math.min(1 - eps, predicted.get(i, j)));
        const t = target.get(i, j);
        loss -= t * Math.log(p) + (1 - t) * Math.log(1 - p);
      }
    }
    return loss / predicted.rows;
  },
  gradient(predicted, target) {
    const eps = 1e-15;
    const grad = new Matrix(predicted.rows, predicted.cols);
    for (let i = 0; i < predicted.rows; i++) {
      for (let j = 0; j < predicted.cols; j++) {
        const p = Math.max(eps, Math.min(1 - eps, predicted.get(i, j)));
        const t = target.get(i, j);
        grad.set(i, j, (p - t) / (p * (1 - p) + eps));
      }
    }
    return grad;
  }
};

// Cosine Similarity Loss
// Minimizes 1 - cos(predicted, target) for each sample
export const cosineSimilarityLoss = {
  name: 'cosine_similarity',
  compute(predicted, target) {
    let totalLoss = 0;
    for (let i = 0; i < predicted.rows; i++) {
      let dot = 0, normP = 0, normT = 0;
      for (let j = 0; j < predicted.cols; j++) {
        const p = predicted.get(i, j);
        const t = target.get(i, j);
        dot += p * t;
        normP += p * p;
        normT += t * t;
      }
      const sim = dot / (Math.sqrt(normP) * Math.sqrt(normT) + 1e-8);
      totalLoss += 1 - sim;
    }
    return totalLoss / predicted.rows;
  },
  gradient(predicted, target) {
    const grad = new Matrix(predicted.rows, predicted.cols);
    for (let i = 0; i < predicted.rows; i++) {
      let dot = 0, normP = 0, normT = 0;
      for (let j = 0; j < predicted.cols; j++) {
        const p = predicted.get(i, j);
        const t = target.get(i, j);
        dot += p * t;
        normP += p * p;
        normT += t * t;
      }
      const normPsqrt = Math.sqrt(normP) + 1e-8;
      const normTsqrt = Math.sqrt(normT) + 1e-8;
      const denom = normPsqrt * normTsqrt;
      for (let j = 0; j < predicted.cols; j++) {
        const p = predicted.get(i, j);
        const t = target.get(i, j);
        // d(1 - cos)/dp = -(t/denom - p*dot/(normP*denom))
        const g = -(t / denom - p * dot / (normP * denom + 1e-8));
        grad.set(i, j, g / predicted.rows);
      }
    }
    return grad;
  }
};

// Hinge Loss (for SVM-style classification, targets should be -1 or +1)
export const hingeLoss = {
  name: 'hinge',
  compute(predicted, target) {
    let loss = 0;
    for (let i = 0; i < predicted.rows; i++) {
      for (let j = 0; j < predicted.cols; j++) {
        loss += Math.max(0, 1 - target.get(i, j) * predicted.get(i, j));
      }
    }
    return loss / predicted.rows;
  },
  gradient(predicted, target) {
    const grad = new Matrix(predicted.rows, predicted.cols);
    for (let i = 0; i < predicted.rows; i++) {
      for (let j = 0; j < predicted.cols; j++) {
        const margin = target.get(i, j) * predicted.get(i, j);
        grad.set(i, j, margin < 1 ? -target.get(i, j) / predicted.rows : 0);
      }
    }
    return grad;
  }
};

// Huber Loss (smooth L1, less sensitive to outliers than MSE)
export const huberLoss = {
  name: 'huber',
  delta: 1.0,
  compute(predicted, target) {
    let loss = 0;
    const delta = this.delta;
    for (let i = 0; i < predicted.rows; i++) {
      for (let j = 0; j < predicted.cols; j++) {
        const diff = predicted.get(i, j) - target.get(i, j);
        if (Math.abs(diff) <= delta) {
          loss += 0.5 * diff * diff;
        } else {
          loss += delta * (Math.abs(diff) - 0.5 * delta);
        }
      }
    }
    return loss / predicted.rows;
  },
  gradient(predicted, target) {
    const grad = new Matrix(predicted.rows, predicted.cols);
    const delta = this.delta;
    for (let i = 0; i < predicted.rows; i++) {
      for (let j = 0; j < predicted.cols; j++) {
        const diff = predicted.get(i, j) - target.get(i, j);
        if (Math.abs(diff) <= delta) {
          grad.set(i, j, diff / predicted.rows);
        } else {
          grad.set(i, j, delta * Math.sign(diff) / predicted.rows);
        }
      }
    }
    return grad;
  }
};

// Triplet Loss (for metric learning)
// Takes three matrices: anchor, positive, negative
// Loss = max(0, ||anchor - positive||² - ||anchor - negative||² + margin)
export function tripletLoss(anchor, positive, negative, margin = 1.0) {
  let loss = 0;
  for (let i = 0; i < anchor.rows; i++) {
    let distPos = 0, distNeg = 0;
    for (let j = 0; j < anchor.cols; j++) {
      const diffP = anchor.get(i, j) - positive.get(i, j);
      const diffN = anchor.get(i, j) - negative.get(i, j);
      distPos += diffP * diffP;
      distNeg += diffN * diffN;
    }
    loss += Math.max(0, distPos - distNeg + margin);
  }
  return loss / anchor.rows;
}

// Triplet loss gradients (returns gradients for anchor, positive, negative)
export function tripletLossGradient(anchor, positive, negative, margin = 1.0) {
  const gradA = new Matrix(anchor.rows, anchor.cols);
  const gradP = new Matrix(anchor.rows, anchor.cols);
  const gradN = new Matrix(anchor.rows, anchor.cols);

  for (let i = 0; i < anchor.rows; i++) {
    let distPos = 0, distNeg = 0;
    for (let j = 0; j < anchor.cols; j++) {
      const diffP = anchor.get(i, j) - positive.get(i, j);
      const diffN = anchor.get(i, j) - negative.get(i, j);
      distPos += diffP * diffP;
      distNeg += diffN * diffN;
    }

    if (distPos - distNeg + margin > 0) {
      for (let j = 0; j < anchor.cols; j++) {
        const diffP = anchor.get(i, j) - positive.get(i, j);
        const diffN = anchor.get(i, j) - negative.get(i, j);
        gradA.set(i, j, 2 * (diffP - diffN) / anchor.rows);
        gradP.set(i, j, -2 * diffP / anchor.rows);
        gradN.set(i, j, 2 * diffN / anchor.rows);
      }
    }
  }

  return { gradAnchor: gradA, gradPositive: gradP, gradNegative: gradN };
}

export function getLoss(name) {
  const losses = {
    mse,
    cross_entropy: crossEntropy,
    crossEntropy,
    crossentropy: crossEntropy,
    binary_cross_entropy: binaryCrossEntropy,
    bce: binaryCrossEntropy,
    cosine: cosineSimilarityLoss,
    cosine_similarity: cosineSimilarityLoss,
    hinge: hingeLoss,
    huber: huberLoss,
  };
  return losses[name] || mse;
}
