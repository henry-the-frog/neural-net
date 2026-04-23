import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { multiTokenPredictionLoss } from './multi-token-prediction.js';

describe('Multi-Token Prediction', () => {
  test('loss is finite', () => {
    const heads = [
      new Float64Array([1, 5, 2]), // Head 1: predict token at idx 1
      new Float64Array([2, 1, 5]), // Head 2: predict token at idx 2
    ];
    const targets = [0, 1, 2, 0]; // Tokens
    const loss = multiTokenPredictionLoss(heads, targets, 0);
    assert.ok(isFinite(loss));
  });

  test('correct predictions give lower loss', () => {
    const good = [new Float64Array([0, 10, 0])]; // Strongly predicts token 1
    const bad = [new Float64Array([10, 0, 0])];  // Predicts token 0 instead
    const targets = [0, 1];
    const goodLoss = multiTokenPredictionLoss(good, targets, 0);
    const badLoss = multiTokenPredictionLoss(bad, targets, 0);
    assert.ok(goodLoss < badLoss);
  });
});
