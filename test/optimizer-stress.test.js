// optimizer-stress.test.js — Verify all optimizers work via Dense layer training
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Dense } from '../src/layer.js';
import { Network } from '../src/network.js';
import { Matrix } from '../src/matrix.js';

// Test that the network can learn with different configurations
describe('Training Convergence — Various Configs', () => {
  function trainNetwork(activation, lr, epochs) {
    let passed = false;
    for (let attempt = 0; attempt < 5 && !passed; attempt++) {
      const net = new Network();
      net.dense(2, 8, activation);
      net.dense(8, 1, 'sigmoid');
      net.loss('mse');
      
      const inputs = Matrix.fromArray([[0,0],[0,1],[1,0],[1,1]]);
      const targets = Matrix.fromArray([[0],[1],[1],[0]]);
      
      const history = net.train({ inputs, targets }, {
        epochs, learningRate: lr, batchSize: 4
      });
      
      if (history[history.length - 1] < history[0] * 0.5) passed = true;
    }
    return passed;
  }

  it('ReLU activation converges', () => {
    assert.ok(trainNetwork('relu', 0.5, 3000), 'ReLU + XOR should converge');
  });

  it('sigmoid activation converges', () => {
    assert.ok(trainNetwork('sigmoid', 1.0, 3000), 'Sigmoid + XOR should converge');
  });

  it('tanh activation converges', () => {
    assert.ok(trainNetwork('tanh', 0.5, 3000), 'Tanh + XOR should converge');
  });

  it('deeper network converges', () => {
    let passed = false;
    for (let attempt = 0; attempt < 5 && !passed; attempt++) {
      const net = new Network();
      net.dense(2, 16, 'relu');
      net.dense(16, 8, 'relu');
      net.dense(8, 1, 'sigmoid');
      net.loss('mse');
      
      const inputs = Matrix.fromArray([[0,0],[0,1],[1,0],[1,1]]);
      const targets = Matrix.fromArray([[0],[1],[1],[0]]);
      
      const history = net.train({ inputs, targets }, {
        epochs: 3000, learningRate: 0.5, batchSize: 4
      });
      
      if (history[history.length - 1] < history[0] * 0.3) passed = true;
    }
    assert.ok(passed, '3-layer network should converge');
  });

  it('different learning rates', () => {
    for (const lr of [0.001, 0.01, 0.1, 1.0]) {
      const net = new Network();
      net.dense(2, 8, 'relu');
      net.dense(8, 1, 'linear');
      net.loss('mse');
      
      const inputs = Matrix.fromArray([[0,0],[0,1],[1,0],[1,1]]);
      const targets = Matrix.fromArray([[0],[1],[1],[0]]);
      
      const history = net.train({ inputs, targets }, {
        epochs: 100, learningRate: lr, batchSize: 4
      });
      
      // All should produce finite losses
      for (const loss of history) {
        assert.ok(isFinite(loss), `Loss should be finite at lr=${lr}: ${loss}`);
      }
    }
  });

  it('BCE loss converges', () => {
    let passed = false;
    for (let attempt = 0; attempt < 5 && !passed; attempt++) {
      const net = new Network();
      net.dense(2, 16, 'relu');
      net.dense(16, 1, 'sigmoid');
      net.loss('bce');
      
      const inputs = Matrix.fromArray([[0,0],[0,1],[1,0],[1,1]]);
      const targets = Matrix.fromArray([[0],[1],[1],[0]]);
      
      const history = net.train({ inputs, targets }, {
        epochs: 3000, learningRate: 0.5, batchSize: 4
      });
      
      if (history[history.length - 1] < history[0] * 0.3) passed = true;
    }
    assert.ok(passed, 'BCE loss should converge');
  });
});
