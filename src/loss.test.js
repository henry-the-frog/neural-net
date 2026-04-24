import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import {
  mse, crossEntropy, binaryCrossEntropy,
  cosineSimilarityLoss, hingeLoss, huberLoss,
  tripletLoss, tripletLossGradient, getLoss
} from './loss.js';
import { Matrix } from './matrix.js';

function mat(data) {
  const rows = data.length;
  const cols = data[0].length;
  const m = new Matrix(rows, cols);
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++)
      m.set(i, j, data[i][j]);
  return m;
}

describe('MSE', () => {
  test('zero loss for perfect prediction', () => {
    const p = mat([[1, 0], [0, 1]]);
    assert.equal(mse.compute(p, p), 0);
  });

  test('positive loss for imperfect prediction', () => {
    const p = mat([[0.5, 0.5]]);
    const t = mat([[1, 0]]);
    assert.ok(mse.compute(p, t) > 0);
  });

  test('gradient is predicted - target', () => {
    const p = mat([[0.5]]);
    const t = mat([[1]]);
    const g = mse.gradient(p, t);
    assert.ok(g.get(0, 0) < 0); // 0.5 - 1 = -0.5
  });
});

describe('Cross-Entropy', () => {
  test('zero loss for perfect prediction', () => {
    const p = mat([[0.999, 0.001]]);
    const t = mat([[1, 0]]);
    assert.ok(crossEntropy.compute(p, t) < 0.01);
  });

  test('high loss for wrong prediction', () => {
    const p = mat([[0.001, 0.999]]);
    const t = mat([[1, 0]]);
    assert.ok(crossEntropy.compute(p, t) > 5);
  });
});

describe('Binary Cross-Entropy', () => {
  test('low loss for correct prediction', () => {
    const p = mat([[0.99]]);
    const t = mat([[1]]);
    assert.ok(binaryCrossEntropy.compute(p, t) < 0.02);
  });

  test('high loss for wrong prediction', () => {
    const p = mat([[0.01]]);
    const t = mat([[1]]);
    assert.ok(binaryCrossEntropy.compute(p, t) > 4);
  });

  test('gradient pushes toward target', () => {
    const p = mat([[0.3]]);
    const t = mat([[1]]);
    const g = binaryCrossEntropy.gradient(p, t);
    assert.ok(g.get(0, 0) < 0); // Should decrease (move toward 1)
  });
});

describe('Cosine Similarity Loss', () => {
  test('zero loss for identical vectors', () => {
    const v = mat([[1, 2, 3]]);
    assert.ok(cosineSimilarityLoss.compute(v, v) < 1e-6);
  });

  test('high loss for opposite vectors', () => {
    const p = mat([[1, 0, 0]]);
    const t = mat([[-1, 0, 0]]);
    assert.ok(cosineSimilarityLoss.compute(p, t) > 1.5);
  });

  test('moderate loss for orthogonal vectors', () => {
    const p = mat([[1, 0]]);
    const t = mat([[0, 1]]);
    assert.ok(Math.abs(cosineSimilarityLoss.compute(p, t) - 1.0) < 0.01);
  });

  test('gradient has correct shape', () => {
    const p = mat([[1, 2], [3, 4]]);
    const t = mat([[5, 6], [7, 8]]);
    const g = cosineSimilarityLoss.gradient(p, t);
    assert.equal(g.rows, 2);
    assert.equal(g.cols, 2);
  });
});

describe('Hinge Loss', () => {
  test('zero loss for correct margin', () => {
    const p = mat([[2]]);   // predicted > 0 for positive class
    const t = mat([[1]]);   // target = +1
    assert.equal(hingeLoss.compute(p, t), 0);
  });

  test('positive loss for wrong prediction', () => {
    const p = mat([[-0.5]]);
    const t = mat([[1]]);
    assert.ok(hingeLoss.compute(p, t) > 0);
  });

  test('gradient for violated margin', () => {
    const p = mat([[0.5]]);
    const t = mat([[1]]);
    const g = hingeLoss.gradient(p, t);
    assert.ok(g.get(0, 0) < 0); // margin violated, grad = -target
  });

  test('zero gradient for satisfied margin', () => {
    const p = mat([[2]]);
    const t = mat([[1]]);
    const g = hingeLoss.gradient(p, t);
    assert.equal(g.get(0, 0), 0);
  });
});

describe('Huber Loss', () => {
  test('behaves like MSE for small errors', () => {
    const p = mat([[0.1]]);
    const t = mat([[0]]);
    const mseVal = mse.compute(p, t);
    const huberVal = huberLoss.compute(p, t);
    // Huber should be close to 0.5 * 0.1^2 = 0.005
    assert.ok(Math.abs(huberVal - 0.005) < 0.001);
  });

  test('grows linearly for large errors', () => {
    const small = huberLoss.compute(mat([[5]]), mat([[0]]));
    const large = huberLoss.compute(mat([[10]]), mat([[0]]));
    // Linear growth: large ≈ 2 * small (approximately, for values >> delta)
    assert.ok(large / small < 2.5 && large / small > 1.5);
  });
});

describe('Triplet Loss', () => {
  test('zero loss when negative is far', () => {
    const anchor = mat([[1, 0]]);
    const positive = mat([[1, 0.1]]);
    const negative = mat([[10, 10]]);
    assert.equal(tripletLoss(anchor, positive, negative, 0.5), 0);
  });

  test('positive loss when negative is close', () => {
    const anchor = mat([[1, 0]]);
    const positive = mat([[5, 5]]);    // far
    const negative = mat([[1, 0.1]]);  // close
    assert.ok(tripletLoss(anchor, positive, negative, 1.0) > 0);
  });

  test('gradient shapes are correct', () => {
    const anchor = mat([[1, 2], [3, 4]]);
    const positive = mat([[1.1, 2.1], [3.1, 4.1]]);
    const negative = mat([[5, 6], [7, 8]]);
    const { gradAnchor, gradPositive, gradNegative } = tripletLossGradient(anchor, positive, negative);
    assert.equal(gradAnchor.rows, 2);
    assert.equal(gradAnchor.cols, 2);
    assert.equal(gradPositive.rows, 2);
    assert.equal(gradNegative.rows, 2);
  });
});

describe('getLoss', () => {
  test('retrieves all named losses', () => {
    assert.equal(getLoss('mse').name, 'mse');
    assert.equal(getLoss('cross_entropy').name, 'cross_entropy');
    assert.equal(getLoss('bce').name, 'binary_cross_entropy');
    assert.equal(getLoss('cosine').name, 'cosine_similarity');
    assert.equal(getLoss('hinge').name, 'hinge');
    assert.equal(getLoss('huber').name, 'huber');
  });

  test('defaults to mse for unknown', () => {
    assert.equal(getLoss('unknown').name, 'mse');
  });
});
