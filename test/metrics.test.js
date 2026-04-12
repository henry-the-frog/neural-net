// metrics.test.js — Tests for evaluation metrics

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  accuracy, confusionMatrix, classificationReport,
  macroF1, weightedF1, mse, mae, r2Score, rmse
} from '../src/metrics.js';

describe('Classification Metrics', () => {
  describe('accuracy', () => {
    it('should return 1.0 for perfect predictions', () => {
      assert.equal(accuracy([0, 1, 2, 0, 1], [0, 1, 2, 0, 1]), 1.0);
    });

    it('should return 0.0 for completely wrong predictions', () => {
      assert.equal(accuracy([1, 0], [0, 1]), 0.0);
    });

    it('should handle mixed results', () => {
      assert.ok(Math.abs(accuracy([0, 1, 1, 0], [0, 1, 0, 0]) - 0.75) < 0.01);
    });
  });

  describe('confusionMatrix', () => {
    it('should be diagonal for perfect predictions', () => {
      const cm = confusionMatrix([0, 1, 2, 0], [0, 1, 2, 0]);
      assert.equal(cm[0][0], 2);
      assert.equal(cm[1][1], 1);
      assert.equal(cm[2][2], 1);
      assert.equal(cm[0][1], 0);
    });

    it('should track misclassifications', () => {
      const cm = confusionMatrix([1, 0], [0, 1]); // Both wrong
      assert.equal(cm[0][1], 1); // Actual 0, predicted 1
      assert.equal(cm[1][0], 1); // Actual 1, predicted 0
    });
  });

  describe('classificationReport', () => {
    it('should compute per-class metrics', () => {
      // Perfect 2-class classification
      const report = classificationReport([0, 0, 1, 1], [0, 0, 1, 1]);
      assert.equal(report[0].precision, 1.0);
      assert.equal(report[0].recall, 1.0);
      assert.equal(report[0].f1, 1.0);
    });

    it('should handle zero-precision cases', () => {
      // Predict everything as class 0
      const report = classificationReport([0, 0, 0, 0], [0, 0, 1, 1]);
      assert.equal(report[1].precision, 0); // No true positives for class 1
      assert.equal(report[1].recall, 0);    // All class-1 missed
    });

    it('should report correct support', () => {
      const report = classificationReport([0, 0, 1, 1, 1], [0, 1, 1, 1, 0]);
      // Actual class 0: 2 instances, actual class 1: 3 instances
      assert.equal(report[0].support, 2);
      assert.equal(report[1].support, 3);
    });
  });

  describe('macroF1', () => {
    it('should average F1 across classes', () => {
      const f1 = macroF1([0, 0, 1, 1], [0, 0, 1, 1]);
      assert.equal(f1, 1.0);
    });
  });

  describe('weightedF1', () => {
    it('should weight by class support', () => {
      const f1 = weightedF1([0, 0, 1, 1, 1], [0, 0, 1, 1, 1]);
      assert.equal(f1, 1.0);
    });
  });
});

describe('Regression Metrics', () => {
  describe('mse', () => {
    it('should return 0 for perfect predictions', () => {
      assert.equal(mse([1, 2, 3], [1, 2, 3]), 0);
    });

    it('should compute correct MSE', () => {
      // (1-2)² + (3-3)² = 1
      assert.ok(Math.abs(mse([2, 3], [1, 3]) - 0.5) < 0.001);
    });
  });

  describe('mae', () => {
    it('should return 0 for perfect predictions', () => {
      assert.equal(mae([1, 2, 3], [1, 2, 3]), 0);
    });

    it('should compute correct MAE', () => {
      assert.ok(Math.abs(mae([2, 5], [1, 3]) - 1.5) < 0.001);
    });
  });

  describe('r2Score', () => {
    it('should return 1.0 for perfect predictions', () => {
      assert.equal(r2Score([1, 2, 3], [1, 2, 3]), 1.0);
    });

    it('should return 0 for mean prediction', () => {
      // Predicting the mean of [1, 2, 3] = 2 for everything
      const r2 = r2Score([2, 2, 2], [1, 2, 3]);
      assert.ok(Math.abs(r2) < 0.01);
    });

    it('should be negative for worse than mean', () => {
      const r2 = r2Score([10, 20, 30], [1, 2, 3]);
      assert.ok(r2 < 0, `R² should be negative: ${r2}`);
    });
  });

  describe('rmse', () => {
    it('should return 0 for perfect predictions', () => {
      assert.equal(rmse([1, 2, 3], [1, 2, 3]), 0);
    });

    it('should be sqrt of MSE', () => {
      const mseVal = mse([2, 4], [1, 3]);
      assert.ok(Math.abs(rmse([2, 4], [1, 3]) - Math.sqrt(mseVal)) < 1e-10);
    });
  });
});
