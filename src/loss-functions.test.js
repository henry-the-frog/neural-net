// loss-functions.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { mse, mae, huber, binaryCrossEntropy, crossEntropy, focalLoss, diceLoss, hingeLoss } from './loss-functions.js';

describe('Loss Functions', () => {
  test('MSE is 0 for perfect prediction', () => {
    assert.equal(mse([1, 2, 3], [1, 2, 3]), 0);
  });

  test('MSE is positive for imperfect prediction', () => {
    assert.ok(mse([1, 2, 3], [2, 3, 4]) > 0);
  });

  test('MAE is sum of absolute differences / n', () => {
    assert.equal(mae([0, 0], [1, -1]), 1);
  });

  test('Huber: small errors → MSE-like', () => {
    const h = huber([0], [0.1], 1.0);
    const m = mse([0], [0.1]);
    assert.ok(Math.abs(h - m / 2) < 0.01); // Huber = 0.5 * err² for small err
  });

  test('Huber: large errors → MAE-like', () => {
    const h = huber([0], [10], 1.0);
    // For err=10, delta=1: delta * (err - 0.5*delta) = 1 * 9.5 = 9.5
    assert.ok(Math.abs(h - 9.5) < 0.01);
  });

  test('BCE is 0 when prediction matches target', () => {
    const loss = binaryCrossEntropy([0.999], [1]);
    assert.ok(loss < 0.01);
  });

  test('BCE is large when prediction is wrong', () => {
    const loss = binaryCrossEntropy([0.001], [1]);
    assert.ok(loss > 5);
  });

  test('cross-entropy picks correct class', () => {
    const loss1 = crossEntropy([10, 0, 0], 0); // Correct class has high logit
    const loss2 = crossEntropy([0, 10, 0], 0); // Wrong class has high logit
    assert.ok(loss1 < loss2);
  });

  test('focal loss reduces penalty for easy examples', () => {
    const easy = focalLoss([0.95], [1], 2); // Easy: high confidence, correct
    const hard = focalLoss([0.55], [1], 2); // Hard: low confidence
    assert.ok(easy < hard, 'Easy examples should have lower focal loss');
  });

  test('dice loss: perfect overlap → 0', () => {
    const loss = diceLoss([1, 0, 1], [1, 0, 1]);
    assert.ok(loss < 0.01);
  });

  test('hinge loss: correct margin → 0', () => {
    // Target 1, prediction 2 → margin satisfied
    assert.equal(hingeLoss([2], [1]), 0);
    // Target 1, prediction -1 → margin violated
    assert.ok(hingeLoss([-1], [1]) > 0);
  });
});
