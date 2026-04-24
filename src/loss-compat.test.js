// loss-compat.test.js — Loss Function Compatibility Matrix
// Verifies every loss function works with forward pass, gradient computation,
// and a few training steps without NaN/Infinity/crashes.

import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { Network, Dense, Matrix } from './index.js';
import {
  mse, crossEntropy, binaryCrossEntropy,
  cosineSimilarityLoss, hingeLoss, huberLoss,
  tripletLoss, tripletLossGradient, getLoss
} from './loss.js';

function mat(data) {
  const rows = data.length;
  const cols = data[0].length;
  const m = new Matrix(rows, cols);
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++)
      m.set(i, j, data[i][j]);
  return m;
}

function isFiniteMatrix(m) {
  for (let i = 0; i < m.rows; i++)
    for (let j = 0; j < m.cols; j++)
      if (!isFinite(m.get(i, j))) return false;
  return true;
}

const standardLosses = [
  { name: 'mse', loss: mse },
  { name: 'cross_entropy', loss: crossEntropy },
  { name: 'binary_cross_entropy', loss: binaryCrossEntropy },
  { name: 'cosine_similarity', loss: cosineSimilarityLoss },
  { name: 'hinge', loss: hingeLoss },
  { name: 'huber', loss: huberLoss },
];

describe('Loss Compatibility: compute + gradient', () => {
  for (const { name, loss } of standardLosses) {
    test(`${name}: compute returns finite number`, () => {
      const pred = mat([[0.5, 0.3], [0.7, 0.2]]);
      const target = mat([[1, 0], [0, 1]]);
      const val = loss.compute(pred, target);
      assert.ok(isFinite(val), `${name} returned ${val}`);
    });

    test(`${name}: gradient returns finite matrix`, () => {
      const pred = mat([[0.5, 0.3], [0.7, 0.2]]);
      const target = mat([[1, 0], [0, 1]]);
      const grad = loss.gradient(pred, target);
      assert.ok(grad instanceof Matrix);
      assert.equal(grad.rows, 2);
      assert.equal(grad.cols, 2);
      assert.ok(isFiniteMatrix(grad), `${name} gradient has non-finite values`);
    });
  }
});

describe('Loss Compatibility: edge cases', () => {
  for (const { name, loss } of standardLosses) {
    test(`${name}: handles single sample`, () => {
      const pred = mat([[0.5]]);
      const target = mat([[1]]);
      const val = loss.compute(pred, target);
      assert.ok(isFinite(val));
    });

    test(`${name}: handles near-zero predictions`, () => {
      const pred = mat([[0.001, 0.999]]);
      const target = mat([[0, 1]]);
      const val = loss.compute(pred, target);
      assert.ok(isFinite(val), `${name} with near-zero: ${val}`);
    });

    test(`${name}: handles near-one predictions`, () => {
      const pred = mat([[0.999, 0.001]]);
      const target = mat([[1, 0]]);
      const val = loss.compute(pred, target);
      assert.ok(isFinite(val), `${name} with near-one: ${val}`);
    });
  }
});

describe('Loss Compatibility: training integration', () => {
  const lossNames = ['mse', 'cross_entropy', 'bce', 'cosine', 'hinge', 'huber'];

  for (const lossName of lossNames) {
    test(`${lossName}: 10 training steps without crash`, () => {
      const net = new Network();
      net.add(new Dense(2, 4, 'relu'));
      net.add(new Dense(4, 1, 'sigmoid'));
      net.loss(lossName);

      const inputs = new Matrix(10, 2);
      const targets = new Matrix(10, 1);
      for (let i = 0; i < 10; i++) {
        inputs.set(i, 0, Math.random());
        inputs.set(i, 1, Math.random());
        targets.set(i, 0, Math.random() > 0.5 ? 0.9 : 0.1);  // Avoid 0/1 exactly
      }

      const history = net.train({ inputs, targets }, { epochs: 10, learningRate: 0.01 });
      assert.ok(history.length === 10);
      // All losses should be finite
      for (const l of history) {
        assert.ok(isFinite(l), `${lossName} produced non-finite loss: ${l}`);
      }
    });
  }
});

describe('Loss Compatibility: getLoss registry', () => {
  const names = ['mse', 'cross_entropy', 'crossEntropy', 'crossentropy',
    'binary_cross_entropy', 'bce', 'cosine', 'cosine_similarity',
    'hinge', 'huber'];

  for (const name of names) {
    test(`getLoss('${name}') returns valid loss`, () => {
      const loss = getLoss(name);
      assert.ok(loss);
      assert.ok(typeof loss.compute === 'function');
      assert.ok(typeof loss.gradient === 'function');
    });
  }
});

describe('Triplet Loss Compatibility', () => {
  test('compute returns finite number', () => {
    const anchor = mat([[1, 2], [3, 4]]);
    const positive = mat([[1.1, 2.1], [3.1, 4.1]]);
    const negative = mat([[5, 6], [7, 8]]);
    const loss = tripletLoss(anchor, positive, negative, 1.0);
    assert.ok(isFinite(loss));
  });

  test('gradients are finite', () => {
    const anchor = mat([[1, 2]]);
    const positive = mat([[1.5, 2.5]]);
    const negative = mat([[5, 6]]);
    const { gradAnchor, gradPositive, gradNegative } = tripletLossGradient(anchor, positive, negative);
    assert.ok(isFiniteMatrix(gradAnchor));
    assert.ok(isFiniteMatrix(gradPositive));
    assert.ok(isFiniteMatrix(gradNegative));
  });

  test('zero loss when negative is very far', () => {
    const anchor = mat([[0, 0]]);
    const positive = mat([[0.1, 0]]);
    const negative = mat([[100, 100]]);
    assert.equal(tripletLoss(anchor, positive, negative, 1.0), 0);
  });

  test('margin affects loss', () => {
    const anchor = mat([[0, 0]]);
    const positive = mat([[1, 0]]);
    const negative = mat([[2, 0]]);
    const l1 = tripletLoss(anchor, positive, negative, 0.5);
    const l2 = tripletLoss(anchor, positive, negative, 5.0);
    assert.ok(l2 >= l1);
  });
});
