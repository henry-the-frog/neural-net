import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { accuracy, confusionMatrix, precisionRecallF1, topKAccuracy } from './metrics.js';

describe('Metrics', () => {
  test('accuracy: perfect', () => assert.equal(accuracy([1,2,3], [1,2,3]), 1));
  test('accuracy: 50%', () => assert.equal(accuracy([1,2], [1,3]), 0.5));
  
  test('confusion matrix shape', () => {
    const cm = confusionMatrix([0,1,0,1], [0,0,1,1], 2);
    assert.equal(cm.length, 2);
    assert.equal(cm[0].length, 2);
  });

  test('precision/recall/f1', () => {
    const { precision, recall, f1 } = precisionRecallF1([1,1,0,0], [1,0,0,1], 1);
    assert.ok(Math.abs(precision - 0.5) < 0.01);
    assert.ok(Math.abs(recall - 0.5) < 0.01);
  });

  test('topK accuracy', () => {
    const logits = [[0.1, 0.9, 0.5]]; // Top-1: class 1, Top-2: 1,2
    assert.equal(topKAccuracy(logits, [1], 1), 1);
    assert.equal(topKAccuracy(logits, [2], 1), 0);
    assert.equal(topKAccuracy(logits, [2], 2), 1);
  });
});
