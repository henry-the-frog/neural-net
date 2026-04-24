import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import {
  accuracy, confusionMatrix, precisionRecallF1, topKAccuracy,
  classificationReport, macroAverage, weightedAverage, microAverage,
  rocAuc, mae, mse, rmse, r2Score, matthewsCorrelation, cohensKappa
} from './metrics.js';

describe('Metrics — Classification', () => {
  test('accuracy: perfect', () => assert.equal(accuracy([1,2,3], [1,2,3]), 1));
  test('accuracy: 50%', () => assert.equal(accuracy([1,2], [1,3]), 0.5));
  test('accuracy: 0%', () => assert.equal(accuracy([0,0,0], [1,1,1]), 0));
  
  test('confusion matrix shape', () => {
    const cm = confusionMatrix([0,1,0,1], [0,0,1,1], 2);
    assert.equal(cm.length, 2);
    assert.equal(cm[0].length, 2);
  });

  test('confusion matrix values', () => {
    const cm = confusionMatrix([0,0,1,1], [0,1,0,1], 2);
    assert.equal(cm[0][0], 1); // TN
    assert.equal(cm[0][1], 1); // FP (true=0, pred=1)
    assert.equal(cm[1][0], 1); // FN (true=1, pred=0)
    assert.equal(cm[1][1], 1); // TP
  });

  test('precision/recall/f1', () => {
    const { precision, recall, f1 } = precisionRecallF1([1,1,0,0], [1,0,0,1], 1);
    assert.ok(Math.abs(precision - 0.5) < 0.01);
    assert.ok(Math.abs(recall - 0.5) < 0.01);
  });

  test('precision/recall/f1 perfect', () => {
    const { precision, recall, f1 } = precisionRecallF1([1,0,1,0], [1,0,1,0], 1);
    assert.ok(precision > 0.99);
    assert.ok(recall > 0.99);
    assert.ok(f1 > 0.99);
  });

  test('topK accuracy', () => {
    const logits = [[0.1, 0.9, 0.5]]; // Top-1: class 1, Top-2: 1,2
    assert.equal(topKAccuracy(logits, [1], 1), 1);
    assert.equal(topKAccuracy(logits, [2], 1), 0);
    assert.equal(topKAccuracy(logits, [2], 2), 1);
  });
});

describe('Metrics — Multi-class', () => {
  const preds   = [0, 0, 1, 1, 2, 2, 0, 1, 2];
  const targets = [0, 1, 1, 1, 2, 0, 0, 2, 2];

  test('classificationReport returns per-class metrics', () => {
    const report = classificationReport(preds, targets);
    assert.equal(report.length, 3); // 3 classes
    assert.equal(report[0].class, 0);
    assert.equal(report[1].class, 1);
    assert.equal(report[2].class, 2);
    // Class 0: TP=2, FP=1, FN=1 → precision=2/3, recall=2/3
    assert.ok(Math.abs(report[0].precision - 2/3) < 0.01);
    assert.ok(Math.abs(report[0].recall - 2/3) < 0.01);
  });

  test('macroAverage computes unweighted mean', () => {
    const avg = macroAverage(preds, targets);
    assert.equal(avg.type, 'macro');
    assert.ok(avg.precision > 0 && avg.precision < 1);
    assert.ok(avg.recall > 0 && avg.recall < 1);
  });

  test('weightedAverage accounts for support', () => {
    const avg = weightedAverage(preds, targets);
    assert.equal(avg.type, 'weighted');
    assert.ok(avg.precision > 0 && avg.precision < 1);
  });

  test('microAverage aggregates globally', () => {
    const avg = microAverage(preds, targets);
    assert.equal(avg.type, 'micro');
    // Micro precision = micro recall = accuracy for multi-class
    const acc = accuracy(preds, targets);
    assert.ok(Math.abs(avg.precision - acc) < 0.01);
  });

  test('perfect classification report', () => {
    const report = classificationReport([0,1,2], [0,1,2]);
    for (const r of report) {
      assert.equal(r.precision, 1);
      assert.equal(r.recall, 1);
      assert.equal(r.f1, 1);
    }
  });
});

describe('Metrics — ROC AUC', () => {
  test('perfect separation gives AUC ≈ 1', () => {
    const scores  = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1];
    const targets = [1,   1,   1,   0,   0,   0];
    assert.ok(rocAuc(scores, targets) > 0.99);
  });

  test('random gives AUC ≈ 0.5', () => {
    const scores  = [0.5, 0.5, 0.5, 0.5];
    const targets = [1,   0,   1,   0];
    assert.ok(Math.abs(rocAuc(scores, targets) - 0.5) < 0.1);
  });

  test('inverted gives AUC ≈ 0', () => {
    const scores  = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9];
    const targets = [1,   1,   1,   0,   0,   0];
    assert.ok(rocAuc(scores, targets) < 0.05);
  });

  test('all same class returns 0.5', () => {
    assert.equal(rocAuc([0.5, 0.6], [1, 1]), 0.5);
  });
});

describe('Metrics — Regression', () => {
  test('MAE: perfect', () => assert.equal(mae([1,2,3], [1,2,3]), 0));
  test('MAE: off by 1', () => assert.equal(mae([2,3,4], [1,2,3]), 1));
  
  test('MSE: perfect', () => assert.equal(mse([1,2,3], [1,2,3]), 0));
  test('MSE: off by 1', () => assert.equal(mse([2,3,4], [1,2,3]), 1));
  test('MSE: mixed errors', () => {
    const result = mse([1,3], [2,2]);  // errors: -1, 1 → squared: 1, 1 → mean: 1
    assert.equal(result, 1);
  });
  
  test('RMSE: off by 1', () => assert.equal(rmse([2,3,4], [1,2,3]), 1));
  test('RMSE: mixed', () => {
    const result = rmse([1,5], [3,3]);  // errors: -2, 2 → MSE=4 → RMSE=2
    assert.equal(result, 2);
  });
  
  test('R²: perfect', () => assert.equal(r2Score([1,2,3], [1,2,3]), 1));
  test('R²: mean predictor gives 0', () => {
    // predict mean(1,2,3) = 2 for everything
    const result = r2Score([2,2,2], [1,2,3]);
    assert.ok(Math.abs(result) < 0.01);
  });
  test('R²: worse than mean gives negative', () => {
    const result = r2Score([10,10,10], [1,2,3]);
    assert.ok(result < 0);
  });
});

describe('Metrics — Advanced Classification', () => {
  test('Matthews Correlation: perfect', () => {
    const mcc = matthewsCorrelation([1,0,1,0], [1,0,1,0]);
    assert.ok(Math.abs(mcc - 1) < 0.01);
  });

  test('Matthews Correlation: random', () => {
    const mcc = matthewsCorrelation([1,0,1,0], [0,1,0,1]);
    assert.ok(Math.abs(mcc - (-1)) < 0.01);
  });

  test('Matthews Correlation: 50/50', () => {
    const mcc = matthewsCorrelation([1,1,0,0], [1,0,1,0]);
    assert.ok(Math.abs(mcc) < 0.01);
  });

  test("Cohen's Kappa: perfect agreement", () => {
    const kappa = cohensKappa([0,1,2,0,1], [0,1,2,0,1]);
    assert.ok(Math.abs(kappa - 1) < 0.01);
  });

  test("Cohen's Kappa: no better than chance", () => {
    // With balanced classes and random predictions, kappa ≈ 0
    const kappa = cohensKappa([0,1,0,1], [0,0,1,1]);
    assert.ok(Math.abs(kappa) < 0.01);
  });
});
