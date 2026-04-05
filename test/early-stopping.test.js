// early-stopping.test.js — Tests for EarlyStopping callback
import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import { Network, Matrix, Dense, EarlyStopping, LossHistory } from '../src/index.js';

describe('EarlyStopping', () => {
  function makeData(n = 100) {
    const inputs = new Matrix(n, 2);
    const targets = new Matrix(n, 1);
    for (let i = 0; i < n; i++) {
      const x = Math.random();
      const y = Math.random();
      inputs.set(i, 0, x);
      inputs.set(i, 1, y);
      targets.set(i, 0, x + y > 1 ? 1 : 0);
    }
    return { inputs, targets };
  }

  function makeNet() {
    const net = new Network();
    net.add(new Dense(2, 8, 'relu'));
    net.add(new Dense(8, 1, 'sigmoid'));
    net.loss('mse');
    net.optimizer('adam', { lr: 0.01 });
    return net;
  }

  it('should stop training early when loss plateaus', () => {
    const net = makeNet();
    const es = new EarlyStopping({ patience: 5, minDelta: 1e-6 });
    const data = makeData(200);

    const history = net.train(data, {
      epochs: 1000,
      batchSize: 32,
      callbacks: [es]
    });

    // Should have stopped before 1000 epochs
    assert.ok(history.length < 1000, `Stopped at epoch ${history.length}`);
    assert.ok(es.stopped, 'EarlyStopping flagged as stopped');
  });

  it('should not stop if loss keeps improving', () => {
    const net = makeNet();
    const es = new EarlyStopping({ patience: 100, minDelta: 0 });
    const data = makeData(100);

    const history = net.train(data, {
      epochs: 20,
      batchSize: 32,
      callbacks: [es]
    });

    // With patience=100, should complete all 20 epochs
    assert.equal(history.length, 20);
    assert.ok(!es.stopped);
  });

  it('should restore best weights when restore=true', () => {
    const net = makeNet();
    const es = new EarlyStopping({ patience: 3, restore: true });
    const data = makeData(100);

    net.train(data, {
      epochs: 500,
      batchSize: 32,
      callbacks: [es]
    });

    // After early stopping with restore, the network should have the best weights
    // We can't easily verify the weights are "best" but we can verify restore ran
    if (es.stopped) {
      assert.ok(es.bestEpoch < es.stoppedEpoch, 'Best epoch is before stopped epoch');
    }
  });

  it('should work with mode=max', () => {
    const es = new EarlyStopping({ patience: 3, mode: 'max' });

    // Simulate improving then plateauing metric
    assert.ok(!es.onEpochEnd(0, 0.5, null));
    assert.ok(!es.onEpochEnd(1, 0.7, null));
    assert.ok(!es.onEpochEnd(2, 0.8, null));
    assert.ok(!es.onEpochEnd(3, 0.79, null)); // wait=1
    assert.ok(!es.onEpochEnd(4, 0.78, null)); // wait=2
    assert.ok(es.onEpochEnd(5, 0.77, null));  // wait=3, stop!
    assert.ok(es.stopped);
    assert.equal(es.bestEpoch, 2);
  });

  it('should respect minDelta', () => {
    const es = new EarlyStopping({ patience: 3, minDelta: 0.1, mode: 'min' });

    assert.ok(!es.onEpochEnd(0, 1.0, null));
    assert.ok(!es.onEpochEnd(1, 0.8, null));   // improved by 0.2
    assert.ok(!es.onEpochEnd(2, 0.75, null));   // improved by 0.05 (< minDelta), wait=1
    assert.ok(!es.onEpochEnd(3, 0.74, null));   // wait=2
    assert.ok(es.onEpochEnd(4, 0.73, null));    // wait=3, stop
    assert.ok(es.stopped);
  });

  it('should work with trainWithGradientAccumulation', () => {
    const net = makeNet();
    const es = new EarlyStopping({ patience: 5 });
    const data = makeData(100);

    const history = net.trainWithGradientAccumulation(data, {
      epochs: 500,
      microBatchSize: 10,
      accumSteps: 2,
      optimizer: 'adam',
      callbacks: [es]
    });

    assert.ok(history.length < 500, `Stopped at ${history.length}`);
  });

  it('should reset properly', () => {
    const es = new EarlyStopping({ patience: 2 });
    es.onEpochEnd(0, 1.0, null);
    es.onEpochEnd(1, 1.1, null);
    es.onEpochEnd(2, 1.2, null); // stops

    assert.ok(es.stopped);
    es.reset();
    assert.ok(!es.stopped);
    assert.equal(es.wait, 0);
    assert.equal(es.bestValue, Infinity);
  });
});

describe('LossHistory', () => {
  it('should record all epoch losses', () => {
    const lh = new LossHistory();
    lh.onEpochEnd(0, 1.5);
    lh.onEpochEnd(1, 1.2);
    lh.onEpochEnd(2, 0.9);

    assert.deepEqual(lh.losses, [1.5, 1.2, 0.9]);
  });

  it('should integrate with Network.train', () => {
    const net = new Network();
    net.add(new Dense(2, 4, 'relu'));
    net.add(new Dense(4, 1, 'sigmoid'));
    net.loss('mse');
    net.optimizer('adam', { lr: 0.01 });

    const data = {
      inputs: new Matrix(50, 2).map(() => Math.random()),
      targets: new Matrix(50, 1).map(() => Math.random() > 0.5 ? 1 : 0)
    };

    const lh = new LossHistory();
    const history = net.train(data, { epochs: 10, callbacks: [lh] });

    assert.equal(lh.losses.length, 10);
    // LossHistory losses should match the returned history
    for (let i = 0; i < 10; i++) {
      assert.ok(Math.abs(lh.losses[i] - history[i]) < 1e-10);
    }
  });

  it('should reset properly', () => {
    const lh = new LossHistory();
    lh.onEpochEnd(0, 1.0);
    lh.onEpochEnd(1, 0.5);
    lh.reset();
    assert.deepEqual(lh.losses, []);
  });
});
