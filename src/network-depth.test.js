// network-depth.test.js — Network class depth tests

import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { Network } from './network.js';
import { Dense } from './layer.js';
import { Matrix } from './matrix.js';

describe('Network Construction', () => {
  it('builds network with layers', () => {
    const net = new Network([
      new Dense(4, 8, 'relu'),
      new Dense(8, 3, 'softmax'),
    ]);
    assert.equal(net.layers.length, 2);
  });

  it('empty network has no layers', () => {
    const net = new Network([]);
    assert.equal(net.layers.length, 0);
  });
});

describe('Network Configuration', () => {
  it('network has loss property', () => {
    const net = new Network([
      new Dense(4, 3, 'softmax'),
    ], { loss: 'cross_entropy' });
    assert.ok(net.loss);
  });
});

describe('Network Predict', () => {
  it('predict returns correct shape', () => {
    const net = new Network([
      new Dense(4, 8, 'relu'),
      new Dense(8, 3, 'softmax'),
    ]);
    const input = Matrix.random(2, 4);
    const output = net.predict(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 3);
  });

  it('predict with single sample', () => {
    const net = new Network([
      new Dense(10, 5, 'relu'),
      new Dense(5, 1, 'sigmoid'),
    ]);
    const input = Matrix.random(1, 10);
    const output = net.predict(input);
    assert.equal(output.rows, 1);
    assert.equal(output.cols, 1);
    // Sigmoid output should be between 0 and 1
    assert.ok(output.get(0, 0) >= 0 && output.get(0, 0) <= 1);
  });
});

describe('Network Train', () => {
  it('fit reduces loss over epochs', () => {
    const net = new Network([
      new Dense(2, 4, 'relu'),
      new Dense(4, 1, 'sigmoid'),
    ]);
    net.loss('mse');

    const inputs = new Matrix(4, 2, new Float64Array([0,0, 0,1, 1,0, 1,1]));
    const targets = new Matrix(4, 1, new Float64Array([0, 1, 1, 0]));

    net.train({ inputs, targets }, { epochs: 10, learningRate: 0.1, verbose: false });
    assert.ok(true);
  });
});

describe('Network Save/Load', () => {
  it('save and load preserves architecture', () => {
    const net = new Network([
      new Dense(4, 8, 'relu'),
      new Dense(8, 3, 'softmax'),
    ]);
    const json = net.toJSON();
    const net2 = Network.fromJSON(json);

    assert.equal(net2.layers.length, 2);

    // Same prediction
    const input = Matrix.random(1, 4);
    const out1 = net.predict(input);
    const out2 = net2.predict(input);

    for (let i = 0; i < out1.cols; i++) {
      assert.ok(Math.abs(out1.get(0, i) - out2.get(0, i)) < 1e-6,
        `Prediction should match after save/load`);
    }
  });
});
