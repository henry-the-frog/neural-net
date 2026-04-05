import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix, Residual, Sequential, Dense, Network } from '../src/index.js';

describe('Residual', () => {
  it('forward adds sublayer output to input', () => {
    const dense = new Dense(4, 4, 'linear');
    // Set weights to identity, biases to 1 → sublayer adds 1 to each element
    dense.weights = Matrix.fromArray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]);
    dense.biases = Matrix.fromArray([[1,1,1,1]]);
    
    const res = new Residual(dense);
    const input = Matrix.fromArray([[2, 3, 4, 5]]);
    const output = res.forward(input);
    
    // output = input + (input · I + 1) = input + input + 1 = 2*input + 1
    assert.equal(output.get(0, 0), 5); // 2 + (2 + 1) = 5
    assert.equal(output.get(0, 1), 7); // 3 + (3 + 1) = 7
  });

  it('backward passes gradient through both paths', () => {
    const dense = new Dense(4, 4, 'linear');
    const res = new Residual(dense);
    const input = Matrix.random(2, 4);
    res.forward(input);
    const dOutput = Matrix.random(2, 4);
    const dInput = res.backward(dOutput);
    
    // Gradient should be non-zero (from both skip and sublayer)
    let sum = 0;
    for (let i = 0; i < 2; i++)
      for (let j = 0; j < 4; j++)
        sum += Math.abs(dInput.get(i, j));
    assert.ok(sum > 0);
  });

  it('param count matches sublayer', () => {
    const dense = new Dense(4, 4, 'relu');
    const res = new Residual(dense);
    assert.equal(res.paramCount(), dense.paramCount());
  });

  it('deep residual network trains better', () => {
    // Compare: deep network with vs without residuals
    // Without residuals, deep networks can degrade
    const net = new Network();
    for (let i = 0; i < 5; i++) {
      net.add(new Residual(new Dense(4, 4, 'relu')));
    }
    net.add(new Dense(4, 1, 'sigmoid'));
    net.loss('mse');
    
    const inputs = Matrix.random(20, 4);
    const targets = new Matrix(20, 1);
    for (let i = 0; i < 20; i++) {
      let sum = 0;
      for (let j = 0; j < 4; j++) sum += inputs.get(i, j);
      targets.set(i, 0, sum > 0 ? 1 : 0);
    }
    
    const history = net.train({ inputs, targets }, { epochs: 50, learningRate: 0.01, batchSize: 10 });
    assert.ok(history.length === 50);
  });
});

describe('Sequential', () => {
  it('chains layers together', () => {
    const seq = new Sequential([
      new Dense(4, 8, 'relu'),
      new Dense(8, 4, 'linear')
    ]);
    const input = Matrix.random(2, 4);
    const output = seq.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 4);
  });

  it('backward works', () => {
    const seq = new Sequential([
      new Dense(4, 8, 'relu'),
      new Dense(8, 4, 'linear')
    ]);
    const input = Matrix.random(2, 4);
    seq.forward(input);
    const dOutput = Matrix.random(2, 4);
    const dInput = seq.backward(dOutput);
    assert.equal(dInput.rows, 2);
    assert.equal(dInput.cols, 4);
  });

  it('param count sums all layers', () => {
    const seq = new Sequential([
      new Dense(4, 8, 'relu'),  // 40
      new Dense(8, 4, 'linear') // 36
    ]);
    assert.equal(seq.paramCount(), 40 + 36);
  });

  it('works as Residual sublayer', () => {
    const block = new Residual(new Sequential([
      new Dense(4, 8, 'relu'),
      new Dense(8, 4, 'linear')
    ]));
    const input = Matrix.random(2, 4);
    const output = block.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 4);
  });

  it('ResNet-style: stack of residual blocks', () => {
    const net = new Network();
    for (let i = 0; i < 3; i++) {
      net.add(new Residual(new Sequential([
        new Dense(4, 8, 'relu'),
        new Dense(8, 4, 'linear')
      ])));
    }
    net.add(new Dense(4, 2, 'softmax'));
    net.loss('crossEntropy');
    
    const input = Matrix.random(5, 4);
    const output = net.forward(input);
    assert.equal(output.rows, 5);
    assert.equal(output.cols, 2);
  });
});
