import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix } from '../src/matrix.js';
import { BatchNorm } from '../src/batchnorm.js';

function mat(rows, cols, values) {
  const m = new Matrix(rows, cols);
  for (let i = 0; i < values.length; i++) m.data[i] = values[i];
  return m;
}

describe('BatchNorm', () => {
  it('normalizes to zero mean, unit variance', () => {
    const bn = new BatchNorm(3);
    const input = mat(4, 3, [
      1, 10, 100,
      2, 20, 200,
      3, 30, 300,
      4, 40, 400,
    ]);
    
    const output = bn.forward(input);
    
    // Check each feature has ~zero mean
    for (let j = 0; j < 3; j++) {
      let sum = 0;
      for (let i = 0; i < 4; i++) sum += output.get(i, j);
      assert.ok(Math.abs(sum / 4) < 0.01, `Mean of feature ${j}: ${sum / 4}`);
    }
    
    // Check each feature has ~unit variance
    for (let j = 0; j < 3; j++) {
      let sumSq = 0;
      for (let i = 0; i < 4; i++) sumSq += output.get(i, j) ** 2;
      const variance = sumSq / 4;
      assert.ok(Math.abs(variance - 1) < 0.1, `Variance of feature ${j}: ${variance}`);
    }
  });

  it('applies learnable gamma and beta', () => {
    const bn = new BatchNorm(2);
    bn.gamma = mat(1, 2, [2, 3]); // Scale
    bn.beta = mat(1, 2, [10, 20]); // Shift
    
    const input = mat(4, 2, [1, 2, 3, 4, 5, 6, 7, 8]);
    const output = bn.forward(input);
    
    // Mean of output should be approximately beta (since mean of normalized is 0)
    let sum0 = 0, sum1 = 0;
    for (let i = 0; i < 4; i++) {
      sum0 += output.get(i, 0);
      sum1 += output.get(i, 1);
    }
    assert.ok(Math.abs(sum0 / 4 - 10) < 0.01, `Mean should be ~10, got ${sum0 / 4}`);
    assert.ok(Math.abs(sum1 / 4 - 20) < 0.01, `Mean should be ~20, got ${sum1 / 4}`);
  });

  it('updates running statistics', () => {
    const bn = new BatchNorm(2, { momentum: 0.1 });
    const input = mat(4, 2, [1, 10, 2, 20, 3, 30, 4, 40]);
    
    bn.forward(input);
    
    // Running mean should be ~momentum * batch_mean
    assert.ok(bn.runningMean.get(0, 0) > 0);
    assert.ok(bn.runningVar.get(0, 0) > 0);
  });

  it('uses running stats in eval mode', () => {
    const bn = new BatchNorm(2, { momentum: 1.0 }); // momentum=1 → running = batch stats
    
    // Train to establish running stats
    bn.training = true;
    const trainData = mat(4, 2, [0, 0, 2, 2, 4, 4, 6, 6]); // mean=3, var=5
    bn.forward(trainData);
    
    // Switch to eval
    bn.training = false;
    const testData = mat(1, 2, [3, 3]); // mean of training data
    const output = bn.forward(testData);
    
    // Should normalize using running stats → (3 - 3) / sqrt(5 + eps) ≈ 0
    assert.ok(Math.abs(output.get(0, 0)) < 0.1, `Should be ~0, got ${output.get(0, 0)}`);
  });

  it('backward pass produces correct gradient shapes', () => {
    const bn = new BatchNorm(3);
    const input = mat(4, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    
    bn.forward(input);
    
    const dOutput = mat(4, 3, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]);
    const dInput = bn.backward(dOutput);
    
    assert.equal(dInput.rows, 4);
    assert.equal(dInput.cols, 3);
    assert.equal(bn.dGamma.cols, 3);
    assert.equal(bn.dBeta.cols, 3);
  });

  it('gradient flows correctly', () => {
    const bn = new BatchNorm(2);
    const input = mat(4, 2, [1, 10, 2, 20, 3, 30, 4, 40]);
    
    bn.forward(input);
    
    // Uniform gradient
    const dOutput = mat(4, 2, [1, 1, 1, 1, 1, 1, 1, 1]);
    const dInput = bn.backward(dOutput);
    
    // With uniform gradient flowing back through normalization,
    // dInput should sum to approximately 0 (property of batch norm)
    let sum = 0;
    for (let i = 0; i < dInput.data.length; i++) sum += dInput.data[i];
    assert.ok(Math.abs(sum) < 0.01, `Sum of gradients should be ~0, got ${sum}`);
  });

  it('update changes parameters', () => {
    const bn = new BatchNorm(2);
    const gammaBefore = bn.gamma.get(0, 0);
    
    const input = mat(4, 2, [1, 2, 3, 4, 5, 6, 7, 8]);
    bn.forward(input);
    
    const dOutput = mat(4, 2, [1, 1, 1, 1, 1, 1, 1, 1]);
    bn.backward(dOutput);
    bn.update(0.01);
    
    // Parameters should have changed
    // (dGamma won't be zero because xNorm values differ)
    assert.ok(bn.gamma.get(0, 0) !== gammaBefore || bn.beta.get(0, 0) !== 0);
  });

  it('paramCount returns correct count', () => {
    const bn = new BatchNorm(10);
    assert.equal(bn.paramCount(), 20); // 10 gamma + 10 beta
  });
});
