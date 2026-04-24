import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { ModelCheckpoint, TrainingState, ReduceLROnPlateau } from './model-checkpoint.js';
import { Network, Dense, Matrix } from './index.js';

describe('ModelCheckpoint', () => {
  function makeNet() {
    const net = new Network();
    net.add(new Dense(2, 3, 'relu'));
    net.add(new Dense(3, 1, 'sigmoid'));
    return net;
  }

  test('saves checkpoints on improving metric', () => {
    const ckpt = new ModelCheckpoint({ mode: 'min', maxCheckpoints: 3 });
    const net = makeNet();

    ckpt.onEpochEnd(0, 1.0, net);
    ckpt.onEpochEnd(1, 0.8, net);
    ckpt.onEpochEnd(2, 0.5, net);

    assert.equal(ckpt.getCheckpoints().length, 3);
    assert.equal(ckpt.getBestMetric(), 0.5);
    assert.equal(ckpt.getBestEpoch(), 2);
  });

  test('replaces worst checkpoint when full', () => {
    const ckpt = new ModelCheckpoint({ mode: 'min', maxCheckpoints: 2 });
    const net = makeNet();

    ckpt.onEpochEnd(0, 1.0, net);
    ckpt.onEpochEnd(1, 0.8, net);
    ckpt.onEpochEnd(2, 0.3, net);  // should replace 1.0

    const checkpoints = ckpt.getCheckpoints();
    assert.equal(checkpoints.length, 2);
    assert.equal(checkpoints[0].metric, 0.3);
    assert.equal(checkpoints[1].metric, 0.8);
  });

  test('getBestModel returns valid JSON', () => {
    const ckpt = new ModelCheckpoint();
    const net = makeNet();

    ckpt.onEpochEnd(0, 0.5, net);
    const model = ckpt.getBestModel();
    assert.ok(model);
    assert.ok(model.layers);
    assert.equal(model.layers.length, 2);
  });

  test('getHistory tracks all epochs', () => {
    const ckpt = new ModelCheckpoint();
    const net = makeNet();

    ckpt.onEpochEnd(0, 1.0, net);
    ckpt.onEpochEnd(1, 0.9, net);
    ckpt.onEpochEnd(2, 0.95, net);  // worse but still tracked

    const history = ckpt.getHistory();
    assert.equal(history.length, 3);
    assert.equal(history[2].metric, 0.95);
  });

  test('mode max works for accuracy', () => {
    const ckpt = new ModelCheckpoint({ mode: 'max', maxCheckpoints: 2 });
    const net = makeNet();

    ckpt.onEpochEnd(0, 0.5, net);
    ckpt.onEpochEnd(1, 0.8, net);
    ckpt.onEpochEnd(2, 0.6, net);

    assert.equal(ckpt.getBestMetric(), 0.8);
    assert.equal(ckpt.getBestEpoch(), 1);
  });

  test('reset clears state', () => {
    const ckpt = new ModelCheckpoint();
    const net = makeNet();
    ckpt.onEpochEnd(0, 0.5, net);
    ckpt.reset();
    assert.equal(ckpt.getBestModel(), null);
    assert.equal(ckpt.getHistory().length, 0);
  });

  test('never returns true (never stops training)', () => {
    const ckpt = new ModelCheckpoint();
    const net = makeNet();
    for (let i = 0; i < 100; i++) {
      assert.equal(ckpt.onEpochEnd(i, Math.random(), net), false);
    }
  });
});

describe('TrainingState', () => {
  test('capture and resume training', () => {
    const net = new Network();
    net.add(new Dense(2, 4, 'relu'));
    net.add(new Dense(4, 1, 'sigmoid'));
    net.loss('mse');

    // Create small training data
    const inputs = new Matrix(20, 2);
    const targets = new Matrix(20, 1);
    for (let i = 0; i < 20; i++) {
      inputs.set(i, 0, Math.random());
      inputs.set(i, 1, Math.random());
      targets.set(i, 0, Math.random() > 0.5 ? 1 : 0);
    }

    // Train for 5 epochs
    const history1 = net.train({ inputs, targets }, { epochs: 5, learningRate: 0.01 });

    // Capture state
    const state = TrainingState.capture(net, {
      epoch: 5,
      history: history1,
      config: { epochs: 10, learningRate: 0.01 },
    });

    assert.equal(state.version, 1);
    assert.equal(state.epoch, 5);
    assert.equal(state.history.length, 5);
    assert.ok(state.model);
    assert.ok(state.timestamp);

    // Resume training
    const { network: resumed, history, totalEpochs } = TrainingState.resume(
      Network, state, { inputs, targets }
    );

    assert.ok(resumed instanceof Network);
    assert.equal(history.length, 10);  // 5 + 5
    assert.equal(totalEpochs, 10);
  });

  test('resume with no remaining epochs returns network as-is', () => {
    const net = new Network();
    net.add(new Dense(2, 1, 'sigmoid'));

    const state = TrainingState.capture(net, {
      epoch: 10,
      history: [1, 0.5, 0.3],
      config: { epochs: 10, learningRate: 0.01 },
    });

    const { network, history } = TrainingState.resume(
      Network, state, { inputs: new Matrix(1, 2), targets: new Matrix(1, 1) }
    );

    assert.ok(network instanceof Network);
    assert.deepEqual(history, [1, 0.5, 0.3]);
  });
});

describe('ReduceLROnPlateau', () => {
  test('does not reduce when improving', () => {
    const scheduler = new ReduceLROnPlateau({ patience: 3, factor: 0.5 });

    scheduler.onEpochEnd(0, 1.0, null);
    scheduler.onEpochEnd(1, 0.9, null);
    scheduler.onEpochEnd(2, 0.8, null);

    assert.equal(scheduler.getLRMultiplier(), 1.0);
    assert.equal(scheduler.reductions, 0);
  });

  test('reduces after patience epochs without improvement', () => {
    const scheduler = new ReduceLROnPlateau({ patience: 2, factor: 0.5 });

    scheduler.onEpochEnd(0, 0.5, null);
    scheduler.onEpochEnd(1, 0.6, null);  // no improve, wait=1
    scheduler.onEpochEnd(2, 0.7, null);  // no improve, wait=2 → reduce

    assert.equal(scheduler.reductions, 1);
    assert.equal(scheduler.getLRMultiplier(), 0.5);
  });

  test('multiple reductions compound', () => {
    const scheduler = new ReduceLROnPlateau({ patience: 1, factor: 0.5 });

    scheduler.onEpochEnd(0, 0.5, null);
    scheduler.onEpochEnd(1, 0.6, null);  // wait=1 → reduce
    scheduler.onEpochEnd(2, 0.7, null);  // wait=1 → reduce

    assert.equal(scheduler.reductions, 2);
    assert.equal(scheduler.getLRMultiplier(), 0.25);
  });

  test('reset clears state', () => {
    const scheduler = new ReduceLROnPlateau({ patience: 1, factor: 0.5 });
    scheduler.onEpochEnd(0, 0.5, null);
    scheduler.onEpochEnd(1, 0.6, null);
    scheduler.reset();
    assert.equal(scheduler.reductions, 0);
    assert.equal(scheduler.getLRMultiplier(), 1.0);
  });

  test('mode max for accuracy', () => {
    const scheduler = new ReduceLROnPlateau({ patience: 2, factor: 0.5, mode: 'max' });

    scheduler.onEpochEnd(0, 0.9, null);
    scheduler.onEpochEnd(1, 0.85, null);  // no improve
    scheduler.onEpochEnd(2, 0.8, null);   // no improve → reduce

    assert.equal(scheduler.reductions, 1);
  });
});
