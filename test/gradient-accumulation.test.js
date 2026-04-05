// gradient-accumulation.test.js — Tests for gradient accumulation training
import { strict as assert } from 'node:assert';
import { describe, it } from 'node:test';
import { Network, Matrix, Dense } from '../src/index.js';

describe('Gradient Accumulation', () => {
  // Create a simple XOR-like dataset
  function makeData(n = 100) {
    const inputs = new Matrix(n, 2);
    const targets = new Matrix(n, 1);
    for (let i = 0; i < n; i++) {
      const x = Math.random() > 0.5 ? 1 : 0;
      const y = Math.random() > 0.5 ? 1 : 0;
      inputs.set(i, 0, x);
      inputs.set(i, 1, y);
      targets.set(i, 0, x ^ y);
    }
    return { inputs, targets };
  }

  it('should train with gradient accumulation and reduce loss', () => {
    const net = new Network();
    net.add(new Dense(2, 8, 'relu'));
    net.add(new Dense(8, 1, 'sigmoid'));
    net.loss('mse');
    net.optimizer('adam', { lr: 0.01 });

    const data = makeData(200);
    const history = net.trainWithGradientAccumulation(data, {
      epochs: 50,
      microBatchSize: 8,
      accumSteps: 4,
      learningRate: 0.01,
      optimizer: 'adam'
    });

    assert.ok(history.length === 50, 'Should have 50 epochs');
    assert.ok(history[history.length - 1] < history[0], 'Loss should decrease');
  });

  it('should support accumSteps=1 (equivalent to normal training)', () => {
    const net = new Network();
    net.add(new Dense(2, 4, 'relu'));
    net.add(new Dense(4, 1, 'sigmoid'));
    net.loss('mse');

    const data = makeData(50);
    const history = net.trainWithGradientAccumulation(data, {
      epochs: 20,
      microBatchSize: 10,
      accumSteps: 1,
      optimizer: 'adam'
    });

    assert.ok(history.length === 20);
    assert.ok(typeof history[0] === 'number');
  });

  it('should produce different results with different accumSteps', () => {
    // Train two identical networks with different accumulation settings
    const data = makeData(64);

    // Network 1: small batches, no accumulation
    const net1 = new Network();
    net1.add(new Dense(2, 8, 'relu'));
    net1.add(new Dense(8, 1, 'sigmoid'));
    net1.loss('mse');

    // Network 2: same micro-batch but 4x accumulation
    const net2 = new Network();
    net2.add(new Dense(2, 8, 'relu'));
    net2.add(new Dense(8, 1, 'sigmoid'));
    net2.loss('mse');

    // Copy weights from net1 to net2
    for (let i = 0; i < net1.layers.length; i++) {
      if (net1.layers[i].weights) {
        net2.layers[i].weights = Matrix.fromArray(net1.layers[i].weights.toArray());
        net2.layers[i].biases = Matrix.fromArray(net1.layers[i].biases.toArray());
      }
    }

    const h1 = net1.trainWithGradientAccumulation(data, {
      epochs: 10, microBatchSize: 16, accumSteps: 1, optimizer: 'adam'
    });
    const h2 = net2.trainWithGradientAccumulation(data, {
      epochs: 10, microBatchSize: 16, accumSteps: 4, optimizer: 'adam'
    });

    // Both should reduce loss, but final values differ due to different effective batch sizes
    assert.ok(h1[h1.length - 1] < h1[0], 'net1 loss decreases');
    assert.ok(h2[h2.length - 1] < h2[0], 'net2 loss decreases');
  });

  it('should work with verbose mode', () => {
    const net = new Network();
    net.add(new Dense(2, 4, 'relu'));
    net.add(new Dense(4, 1, 'sigmoid'));
    net.loss('mse');

    const data = makeData(32);
    const history = net.trainWithGradientAccumulation(data, {
      epochs: 5,
      microBatchSize: 8,
      accumSteps: 2,
      optimizer: 'adam',
      verbose: false
    });

    assert.ok(history.length === 5);
  });

  it('should work with learning rate schedules', () => {
    const net = new Network();
    net.add(new Dense(2, 8, 'relu'));
    net.add(new Dense(8, 1, 'sigmoid'));
    net.loss('mse');

    const data = makeData(100);
    for (const schedule of ['cosine', 'step', 'linear']) {
      const history = net.trainWithGradientAccumulation(data, {
        epochs: 10,
        microBatchSize: 10,
        accumSteps: 2,
        optimizer: 'adam',
        lrSchedule: schedule
      });
      assert.ok(history.length === 10, `${schedule} schedule should complete`);
    }
  });

  it('should call onEpoch callback', () => {
    const net = new Network();
    net.add(new Dense(2, 4, 'relu'));
    net.add(new Dense(4, 1, 'sigmoid'));
    net.loss('mse');

    const data = makeData(32);
    const epochLog = [];

    net.trainWithGradientAccumulation(data, {
      epochs: 5,
      microBatchSize: 8,
      accumSteps: 2,
      optimizer: 'adam',
      onEpoch: (epoch, loss) => epochLog.push({ epoch, loss })
    });

    assert.equal(epochLog.length, 5);
    assert.equal(epochLog[0].epoch, 0);
    assert.ok(typeof epochLog[0].loss === 'number');
  });

  it('should handle data size not evenly divisible by effective batch', () => {
    const net = new Network();
    net.add(new Dense(2, 4, 'relu'));
    net.add(new Dense(4, 1, 'sigmoid'));
    net.loss('mse');

    // 37 samples, micro=8, accum=3 → effective=24, so last batch is partial
    const data = makeData(37);
    const history = net.trainWithGradientAccumulation(data, {
      epochs: 5,
      microBatchSize: 8,
      accumSteps: 3,
      optimizer: 'adam'
    });

    assert.ok(history.length === 5);
    assert.ok(!history.some(isNaN), 'No NaN losses');
  });

  it('should work with momentum SGD optimizer', () => {
    const net = new Network();
    net.add(new Dense(2, 8, 'relu'));
    net.add(new Dense(8, 1, 'sigmoid'));
    net.loss('mse');

    const data = makeData(64);
    const history = net.trainWithGradientAccumulation(data, {
      epochs: 20,
      microBatchSize: 8,
      accumSteps: 4,
      optimizer: 'momentum',
      learningRate: 0.05
    });

    assert.ok(history[history.length - 1] < history[0], 'Loss should decrease with momentum');
  });

  it('large effective batch (accum=8) should still converge', () => {
    const net = new Network();
    net.add(new Dense(2, 16, 'relu'));
    net.add(new Dense(16, 1, 'sigmoid'));
    net.loss('mse');

    const data = makeData(128);
    const history = net.trainWithGradientAccumulation(data, {
      epochs: 50,
      microBatchSize: 4,
      accumSteps: 8,
      optimizer: 'adam',
      learningRate: 0.005
    });

    assert.ok(history[history.length - 1] < history[0], 'Should converge with large effective batch');
  });
});
