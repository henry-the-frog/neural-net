// history.test.js — Tests for training history tracking

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { TrainingHistory } from '../src/history.js';

describe('TrainingHistory', () => {
  it('should record epochs', () => {
    const hist = new TrainingHistory();
    hist.record(0, { loss: 1.0, lr: 0.1 });
    hist.record(1, { loss: 0.5, lr: 0.09 });
    assert.equal(hist.length, 2);
    assert.equal(hist.losses[0], 1.0);
    assert.equal(hist.losses[1], 0.5);
  });

  it('should find best epoch', () => {
    const hist = new TrainingHistory();
    hist.record(0, { loss: 1.0 });
    hist.record(1, { loss: 0.3 });
    hist.record(2, { loss: 0.5 });
    assert.equal(hist.best().loss, 0.3);
    assert.equal(hist.best().epoch, 1);
  });

  it('should provide summary', () => {
    const hist = new TrainingHistory();
    for (let i = 0; i < 100; i++) {
      hist.record(i, { loss: 1.0 - i * 0.009 });
    }
    const s = hist.summary();
    assert.equal(s.epochs, 100);
    assert.equal(s.initialLoss, 1.0);
    assert.ok(s.finalLoss < 0.2);
    assert.ok(s.improvement.includes('%'));
  });

  it('should generate sparkline', () => {
    const hist = new TrainingHistory();
    for (let i = 0; i < 50; i++) {
      hist.record(i, { loss: 1.0 / (i + 1) });
    }
    const spark = hist.sparkline();
    assert.ok(spark.length > 0);
    // Should start high and end low
    assert.ok(spark[0] >= spark[spark.length - 1]);
  });

  it('should generate loss plot', () => {
    const hist = new TrainingHistory();
    for (let i = 0; i < 50; i++) {
      hist.record(i, { loss: 1.0 * Math.exp(-i * 0.05) });
    }
    const plot = hist.plotLoss(40, 10);
    assert.ok(plot.includes('Loss:'));
    assert.ok(plot.includes('Epoch'));
  });

  it('should handle empty history', () => {
    const hist = new TrainingHistory();
    assert.equal(hist.length, 0);
    assert.equal(hist.plotLoss(), '(no data)');
    assert.equal(hist.sparkline(), '');
  });

  it('should track time', () => {
    const hist = new TrainingHistory();
    hist.record(0, { loss: 1.0 });
    assert.ok(hist.last().time >= 0);
  });
});
