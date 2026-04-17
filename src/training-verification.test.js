// training-verification.test.js — End-to-end training verification
// Verify that networks can actually LEARN basic functions: XOR, AND, OR, simple regression.

import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { Network } from './network.js';
import { Dense } from './layer.js';
import { Matrix } from './matrix.js';

describe('XOR Learning', () => {
  it('network learns XOR gate', () => {
    const net = new Network();
    net.add(new Dense(2, 8, 'relu'));
    net.add(new Dense(8, 1, 'sigmoid'));
    net.loss('mse');
    net.optimizer('adam', { learningRate: 0.01 });

    // XOR dataset
    const inputs = [
      new Matrix(1, 2, new Float64Array([0, 0])),
      new Matrix(1, 2, new Float64Array([0, 1])),
      new Matrix(1, 2, new Float64Array([1, 0])),
      new Matrix(1, 2, new Float64Array([1, 1])),
    ];
    const targets = [
      new Matrix(1, 1, new Float64Array([0])),
      new Matrix(1, 1, new Float64Array([1])),
      new Matrix(1, 1, new Float64Array([1])),
      new Matrix(1, 1, new Float64Array([0])),
    ];

    // Train for 2000 epochs
    for (let epoch = 0; epoch < 2000; epoch++) {
      net.optimizerInstance.step();
      for (let i = 0; i < 4; i++) {
        net.trainBatch(inputs[i], targets[i]);
      }
    }

    // Verify predictions
    const predictions = inputs.map(inp => {
      const out = net.forward(inp);
      return Math.round(out.get(0, 0));
    });

    assert.deepEqual(predictions, [0, 1, 1, 0], 'Network should learn XOR');
  });

  it('XOR learning with different architecture', () => {
    const net = new Network();
    net.add(new Dense(2, 4, 'tanh'));
    net.add(new Dense(4, 4, 'tanh'));
    net.add(new Dense(4, 1, 'sigmoid'));
    net.loss('mse');
    net.optimizer('adam', { learningRate: 0.01 });

    const data = [
      { in: [0, 0], out: 0 },
      { in: [0, 1], out: 1 },
      { in: [1, 0], out: 1 },
      { in: [1, 1], out: 0 },
    ];

    for (let epoch = 0; epoch < 3000; epoch++) {
      net.optimizerInstance.step();
      for (const d of data) {
        net.trainBatch(
          new Matrix(1, 2, new Float64Array(d.in)),
          new Matrix(1, 1, new Float64Array([d.out]))
        );
      }
    }

    for (const d of data) {
      const out = net.forward(new Matrix(1, 2, new Float64Array(d.in)));
      const pred = Math.round(out.get(0, 0));
      assert.equal(pred, d.out, `XOR(${d.in}) should be ${d.out}, got ${out.get(0, 0).toFixed(3)}`);
    }
  });
});

describe('AND Gate Learning', () => {
  it('network learns AND gate quickly', () => {
    const net = new Network();
    net.add(new Dense(2, 4, 'relu'));
    net.add(new Dense(4, 1, 'sigmoid'));
    net.loss('mse');
    net.optimizer('sgd', { learningRate: 0.5 });

    const data = [
      { in: [0, 0], out: 0 },
      { in: [0, 1], out: 0 },
      { in: [1, 0], out: 0 },
      { in: [1, 1], out: 1 },
    ];

    for (let epoch = 0; epoch < 500; epoch++) {
      for (const d of data) {
        net.trainBatch(
          new Matrix(1, 2, new Float64Array(d.in)),
          new Matrix(1, 1, new Float64Array([d.out]))
        );
      }
    }

    let correct = 0;
    for (const d of data) {
      const out = net.forward(new Matrix(1, 2, new Float64Array(d.in)));
      if (Math.round(out.get(0, 0)) === d.out) correct++;
    }
    assert.equal(correct, 4, 'AND gate should be learned perfectly');
  });
});

describe('OR Gate Learning', () => {
  it('network learns OR gate', () => {
    const net = new Network();
    net.add(new Dense(2, 4, 'relu'));
    net.add(new Dense(4, 1, 'sigmoid'));
    net.loss('mse');
    net.optimizer('sgd', { learningRate: 0.5 });

    const data = [
      { in: [0, 0], out: 0 },
      { in: [0, 1], out: 1 },
      { in: [1, 0], out: 1 },
      { in: [1, 1], out: 1 },
    ];

    for (let epoch = 0; epoch < 500; epoch++) {
      for (const d of data) {
        net.trainBatch(
          new Matrix(1, 2, new Float64Array(d.in)),
          new Matrix(1, 1, new Float64Array([d.out]))
        );
      }
    }

    let correct = 0;
    for (const d of data) {
      const out = net.forward(new Matrix(1, 2, new Float64Array(d.in)));
      if (Math.round(out.get(0, 0)) === d.out) correct++;
    }
    assert.equal(correct, 4, 'OR gate should be learned perfectly');
  });
});

describe('Simple Regression', () => {
  it('network learns linear function y = 2x + 1', () => {
    const net = new Network();
    net.add(new Dense(1, 8, 'relu'));
    net.add(new Dense(8, 1, 'linear'));
    net.loss('mse');
    net.optimizer('adam', { learningRate: 0.01 });

    // Training data: y = 2x + 1 for x in [0, 1]
    for (let epoch = 0; epoch < 1000; epoch++) {
      net.optimizerInstance.step();
      for (let i = 0; i < 10; i++) {
        const x = Math.random();
        const y = 2 * x + 1;
        net.trainBatch(
          new Matrix(1, 1, new Float64Array([x])),
          new Matrix(1, 1, new Float64Array([y]))
        );
      }
    }

    // Test on new points
    const testPoints = [0.1, 0.3, 0.5, 0.7, 0.9];
    let maxError = 0;
    for (const x of testPoints) {
      const pred = net.forward(new Matrix(1, 1, new Float64Array([x]))).get(0, 0);
      const expected = 2 * x + 1;
      maxError = Math.max(maxError, Math.abs(pred - expected));
    }
    assert.ok(maxError < 0.3, `Max error should be < 0.3, got ${maxError.toFixed(3)}`);
  });
});

describe('Training Robustness', () => {
  it('loss decreases over training', () => {
    const net = new Network();
    net.add(new Dense(2, 4, 'relu'));
    net.add(new Dense(4, 1, 'sigmoid'));
    net.loss('mse');
    net.optimizer('adam', { learningRate: 0.01 });

    const input = new Matrix(1, 2, new Float64Array([0.5, 0.3]));
    const target = new Matrix(1, 1, new Float64Array([0.8]));

    const initialOutput = net.forward(input);
    const initialLoss = net.lossFunction.compute(initialOutput, target);

    // Train
    for (let i = 0; i < 500; i++) {
      net.optimizerInstance.step();
      net.trainBatch(input, target);
    }

    const finalOutput = net.forward(input);
    const finalLoss = net.lossFunction.compute(finalOutput, target);

    assert.ok(finalLoss < initialLoss, 
      `Loss should decrease: initial=${initialLoss.toFixed(4)}, final=${finalLoss.toFixed(4)}`);
    assert.ok(finalLoss < 0.01, `Final loss should be small: ${finalLoss.toFixed(4)}`);
  });

  it('network does not diverge with reasonable learning rate', () => {
    const net = new Network();
    net.add(new Dense(3, 10, 'tanh'));
    net.add(new Dense(10, 2, 'sigmoid'));
    net.loss('mse');
    net.optimizer('adam', { learningRate: 0.001 });

    const input = Matrix.random(1, 3);
    const target = new Matrix(1, 2, new Float64Array([0.5, 0.5]));

    for (let i = 0; i < 100; i++) {
      net.optimizerInstance.step();
      net.trainBatch(input, target);
    }

    const output = net.forward(input);
    assert.ok(isFinite(output.get(0, 0)), 'Output should be finite after training');
    assert.ok(!isNaN(output.get(0, 0)), 'Output should not be NaN after training');
  });
});
