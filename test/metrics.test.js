// metrics.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  confusionMatrix, precision, recall, f1Score, accuracy,
  classificationReport, printConfusionMatrix,
} from '../src/metrics.js';

describe('Confusion Matrix', () => {
  it('perfect predictions', () => {
    const cm = confusionMatrix([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2], 3);
    assert.equal(cm[0][0], 2); // class 0: 2 correct
    assert.equal(cm[1][1], 2);
    assert.equal(cm[2][2], 2);
    assert.equal(accuracy(cm), 1.0);
  });

  it('all wrong predictions', () => {
    const cm = confusionMatrix([1, 2, 0], [0, 1, 2], 3);
    assert.equal(accuracy(cm), 0);
  });

  it('binary classification metrics', () => {
    // 3 true positives, 1 false positive, 1 false negative, 2 true negatives
    const pred   = [1, 1, 1, 1, 0, 0, 0];
    const actual = [1, 1, 1, 0, 0, 0, 1];
    const cm = confusionMatrix(pred, actual, 2);
    
    // class 1: TP=3, FP=1, FN=1
    const p = precision(cm);
    const r = recall(cm);
    assert.ok(Math.abs(p[1] - 0.75) < 0.01); // 3/(3+1)
    assert.ok(Math.abs(r[1] - 0.75) < 0.01); // 3/(3+1)
  });

  it('F1 score', () => {
    const cm = [[5, 2], [1, 8]]; // class0: TP=5, FP=1, FN=2; class1: TP=8, FP=2, FN=1
    const f1 = f1Score(cm);
    // class0: P=5/6, R=5/7, F1 = 2*(5/6)*(5/7)/((5/6)+(5/7))
    assert.ok(f1[0] > 0.7 && f1[0] < 0.8);
    assert.ok(f1[1] > 0.8 && f1[1] < 0.9);
  });
});

describe('Classification Report', () => {
  it('returns per-class metrics', () => {
    const pred = [0, 1, 0, 1, 0, 1];
    const actual = [0, 1, 0, 0, 1, 1];
    const report = classificationReport(pred, actual);
    
    assert.ok(Array.isArray(report));
    assert.equal(report.length, 2);
    assert.ok(report[0].precision >= 0);
    assert.ok(report[0].recall >= 0);
    assert.ok(report[0].f1 >= 0);
    assert.ok(report[0].support > 0);
  });
});

describe('Print', () => {
  it('printConfusionMatrix returns string', () => {
    const cm = [[5, 2], [1, 8]];
    const str = printConfusionMatrix(cm);
    assert.ok(str.includes('5'));
    assert.ok(str.includes('8'));
  });
});
