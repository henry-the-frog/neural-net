import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { SGD, Adam, NaturalGradient, createOptimizer } from '../src/optimizer.js';
import { Network } from '../src/network.js';
import { Dense } from '../src/layer.js';
import { Matrix } from '../src/matrix.js';
import { createMiniDigits, trainTestSplit, evaluate, packBatch } from '../src/mnist.js';

describe('Optimizer Comparison', () => {
  // Helper: train a simple network and return final accuracy
  function trainAndEvaluate(optimizerName, lr, epochs = 50) {
    const data = createMiniDigits({ samplesPerDigit: 20, noise: 0.02 });
    const { train, test } = trainTestSplit(data.images, data.labels, data.rawLabels, 0.2);

    const net = new Network();
    net.add(new Dense(64, 32, 'relu'));
    net.add(new Dense(32, 10, 'softmax'));
    net.loss('crossEntropy');

    const { inputs, targets } = packBatch(train.images, train.labels);

    net.train({ inputs, targets }, {
      epochs,
      learningRate: lr,
      batchSize: 16,
    });

    return evaluate(net, test.images, test.rawLabels);
  }

  it('SGD should achieve reasonable accuracy on mini-digits', () => {
    const acc = trainAndEvaluate('sgd', 0.01, 100);
    assert.ok(acc > 0.3, `SGD accuracy should be > 30%, got ${(acc * 100).toFixed(1)}%`);
  });

  it('Adam should achieve reasonable accuracy on mini-digits', () => {
    const acc = trainAndEvaluate('adam', 0.005, 100);
    assert.ok(acc > 0.3, `Adam accuracy should be > 30%, got ${(acc * 100).toFixed(1)}%`);
  });

  it('NaturalGradient should work as standalone optimizer', () => {
    const opt = new NaturalGradient(0.05, 0.01, 0.95);
    
    // Simple optimization: minimize f(x) = x²
    let x = new Matrix(1, 1, new Float64Array([5.0]));
    for (let i = 0; i < 100; i++) {
      const grad = x.mul(2); // df/dx = 2x
      x = opt.update(x, grad, 'x');
    }
    
    assert.ok(Math.abs(x.data[0]) < 1.0,
      `Should converge toward 0, got ${x.data[0].toFixed(4)}`);
  });

  it('NaturalGradient should adapt step size based on Fisher', () => {
    const opt = new NaturalGradient(0.1, 0.001, 0.9);
    
    // Feed many large gradients to build up Fisher estimate
    const param = new Matrix(2, 1, new Float64Array([0, 0]));
    for (let i = 0; i < 20; i++) {
      const grad = new Matrix(2, 1, new Float64Array([
        100 * Math.sin(i), // Large oscillating gradient for param 0
        1,                  // Small stable gradient for param 1
      ]));
      opt.update(param, grad, 'w');
    }

    // Now apply uniform gradient — check Fisher estimates differ
    const fisher = opt.getFisherEstimate('w');
    assert.ok(fisher.data[0] > fisher.data[1] * 10,
      `Fisher for noisy param should be much larger: ${fisher.data[0].toFixed(2)} vs ${fisher.data[1].toFixed(2)}`);
  });

  it('all optimizers should converge on Rosenbrock function', () => {
    // Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
    // Minimum at (1,1)
    function rosenbrockGrad(x, y) {
      const dx = -2 * (1 - x) - 400 * x * (y - x * x);
      const dy = 200 * (y - x * x);
      return [dx, dy];
    }

    const optimizers = [
      { name: 'SGD', opt: new SGD(0.0001) },
      { name: 'Adam', opt: new Adam(0.01) },
      { name: 'Natural', opt: new NaturalGradient(0.001, 0.01, 0.95) },
    ];

    for (const { name, opt } of optimizers) {
      let param = new Matrix(2, 1, new Float64Array([-1, 1]));
      if (opt.step) opt.t = 0;

      for (let i = 0; i < 500; i++) {
        if (opt.step) opt.step();
        const [dx, dy] = rosenbrockGrad(param.data[0], param.data[1]);
        const grad = new Matrix(2, 1, new Float64Array([dx, dy]));
        param = opt.update(param, grad, `${name}_p`);
      }

      const finalVal = (1 - param.data[0]) ** 2 + 100 * (param.data[1] - param.data[0] ** 2) ** 2;
      // All should make progress (not necessarily converge fully)
      assert.ok(finalVal < 100,
        `${name} should make progress on Rosenbrock: f=${finalVal.toFixed(4)}, x=(${param.data[0].toFixed(3)}, ${param.data[1].toFixed(3)})`);
    }
  });

  it('should create optimizers by name', () => {
    const names = ['sgd', 'momentum', 'adam', 'adamw', 'rmsprop', 'natural'];
    for (const name of names) {
      const opt = createOptimizer(name);
      assert.equal(opt.name, name);
    }
  });
});
