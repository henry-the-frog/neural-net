// early-stopping.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { EarlyStopping, trainWithEarlyStopping } from '../src/early-stopping.js';
import { ModelZoo } from '../src/model-zoo.js';
import { Datasets } from '../src/datasets.js';

describe('EarlyStopping', () => {
  it('stops after patience exceeded', () => {
    const stopper = new EarlyStopping({ patience: 3 });
    stopper.step(1.0);  // Best
    stopper.step(0.9);  // Better
    stopper.step(0.95); // Worse (1)
    stopper.step(0.96); // Worse (2)
    const shouldStop = stopper.step(0.97); // Worse (3) → stop
    assert.ok(shouldStop);
    assert.ok(stopper.stopped);
  });

  it('does not stop while improving', () => {
    const stopper = new EarlyStopping({ patience: 5 });
    for (let i = 10; i > 0; i--) {
      assert.ok(!stopper.step(i), `Should not stop at value ${i}`);
    }
  });

  it('tracks best value and epoch', () => {
    const stopper = new EarlyStopping({ patience: 10 });
    stopper.step(1.0);
    stopper.step(0.5);
    stopper.step(0.8);
    assert.equal(stopper.bestValue, 0.5);
    assert.equal(stopper.bestEpoch, 2);
  });

  it('respects minDelta', () => {
    const stopper = new EarlyStopping({ patience: 2, minDelta: 0.1 });
    stopper.step(1.0);   // Best
    stopper.step(0.95);  // Not enough improvement (delta=0.05 < 0.1)
    const shouldStop = stopper.step(0.91); // Still not enough (2 waits)
    assert.ok(shouldStop);
  });

  it('mode=max works for accuracy', () => {
    const stopper = new EarlyStopping({ patience: 3, mode: 'max' });
    stopper.step(0.5);  // Best
    stopper.step(0.8);  // Better
    stopper.step(0.75); // Worse (1)
    stopper.step(0.76); // Worse (2)
    const stop = stopper.step(0.77); // Worse (3)
    assert.ok(stop);
    assert.equal(stopper.bestValue, 0.8);
  });

  it('summary returns correct info', () => {
    const stopper = new EarlyStopping({ patience: 2 });
    stopper.step(1.0);
    stopper.step(0.5);
    stopper.step(0.6);
    stopper.step(0.7);
    const s = stopper.summary();
    assert.ok(s.stopped);
    assert.equal(s.bestValue, 0.5);
    assert.equal(s.bestEpoch, 2);
    assert.equal(s.totalEpochs, 4);
  });
});

describe('trainWithEarlyStopping', () => {
  it('stops before maxEpochs', () => {
    let passed = false;
    for (let attempt = 0; attempt < 3 && !passed; attempt++) {
      const model = ModelZoo.tiny();
      const { inputs, targets } = Datasets.xor();
      const result = trainWithEarlyStopping(model, inputs, targets, inputs, targets, {
        maxEpochs: 1000, lr: 0.5, patience: 50,
      });
      if (result.totalEpochs < 1000) passed = true;
    }
    assert.ok(passed, 'Should stop early in 1 of 3 attempts');
  });

  it('returns best model', () => {
    const model = ModelZoo.tiny();
    const { inputs, targets } = Datasets.xor();
    const result = trainWithEarlyStopping(model, inputs, targets, inputs, targets, {
      maxEpochs: 200, lr: 0.5, patience: 20,
    });
    assert.ok(result.bestValue < 0.5, `Best loss should be reasonable: ${result.bestValue.toFixed(4)}`);
  });
});
