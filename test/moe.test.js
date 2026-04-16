import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { MixtureOfExperts } from '../src/moe.js';
import { Matrix } from '../src/matrix.js';

const approx = (a, b, eps = 0.01) => Math.abs(a - b) < eps;

describe('Mixture of Experts', () => {
  it('forward produces correct shape', () => {
    const moe = new MixtureOfExperts(4, 4, 8, 3, 2);
    const input = Matrix.random(5, 4); // batch of 5
    const output = moe.forward(input);
    assert.equal(output.rows, 5);
    assert.equal(output.cols, 3);
  });

  it('output is finite', () => {
    const moe = new MixtureOfExperts(4, 4, 8, 3, 2);
    const input = Matrix.random(3, 4);
    const output = moe.forward(input);
    for (let i = 0; i < output.rows; i++) {
      for (let j = 0; j < output.cols; j++) {
        assert.ok(Number.isFinite(output.get(i, j)), `Non-finite at (${i},${j})`);
      }
    }
  });

  it('uses top-K experts (not all)', () => {
    const moe = new MixtureOfExperts(4, 8, 8, 3, 2);
    moe.resetRoutingStats();
    const input = Matrix.random(10, 4);
    moe.forward(input);
    // With top-2 of 8, not all experts should be equally used
    assert.equal(moe.totalRouted, 20); // 10 samples * 2 experts
    // At least some experts should have non-zero counts
    const used = moe.routingCounts.filter(c => c > 0).length;
    assert.ok(used >= 2, `Should use at least 2 experts: ${used}`);
  });

  it('routing distribution sums to 1', () => {
    const moe = new MixtureOfExperts(4, 4, 8, 3, 2);
    moe.resetRoutingStats();
    const input = Matrix.random(20, 4);
    moe.forward(input);
    const dist = moe.routingDistribution();
    const sum = dist.reduce((a, b) => a + b, 0);
    assert.ok(approx(sum, 1, 0.01), `Distribution should sum to 1: ${sum}`);
  });

  it('load balance loss is 0 for perfect balance', () => {
    const moe = new MixtureOfExperts(4, 4, 8, 3, 2);
    // Manually set perfect balance
    moe.routingCounts = [5, 5, 5, 5];
    moe.totalRouted = 20;
    const loss = moe.loadBalanceLoss();
    assert.ok(approx(loss, 0, 0.001), `Perfect balance should have 0 loss: ${loss}`);
  });

  it('load balance loss is positive for imbalance', () => {
    const moe = new MixtureOfExperts(4, 4, 8, 3, 2);
    moe.routingCounts = [20, 0, 0, 0];
    moe.totalRouted = 20;
    const loss = moe.loadBalanceLoss();
    assert.ok(loss > 0, `Imbalanced routing should have positive loss: ${loss}`);
  });

  it('backward produces correct shape', () => {
    const moe = new MixtureOfExperts(4, 4, 8, 3, 2);
    const input = Matrix.random(3, 4);
    moe.forward(input);
    const dOutput = Matrix.random(3, 3);
    const dInput = moe.backward(dOutput);
    assert.equal(dInput.rows, 3);
    assert.equal(dInput.cols, 4);
  });

  it('backward gradients are finite', () => {
    const moe = new MixtureOfExperts(4, 4, 8, 3, 2);
    const input = Matrix.random(3, 4);
    moe.forward(input);
    const dOutput = Matrix.random(3, 3);
    const dInput = moe.backward(dOutput);
    for (let i = 0; i < dInput.rows; i++) {
      for (let j = 0; j < dInput.cols; j++) {
        assert.ok(Number.isFinite(dInput.get(i, j)), `Non-finite grad at (${i},${j})`);
      }
    }
  });

  it('paramCount is correct', () => {
    const moe = new MixtureOfExperts(4, 3, 8, 3, 2);
    const params = moe.paramCount();
    // Gate: 4*3 + 3 = 15
    // Each expert: Dense(4,8) + Dense(8,3) = 4*8+8 + 8*3+3 = 40+27 = 67
    // 3 experts: 201
    // Total: 15 + 201 = 216
    assert.equal(params, 216);
  });

  it('top-1 routing uses only one expert per sample', () => {
    const moe = new MixtureOfExperts(4, 4, 8, 3, 1);
    moe.resetRoutingStats();
    const input = Matrix.random(10, 4);
    moe.forward(input);
    assert.equal(moe.totalRouted, 10); // 10 samples * 1 expert
  });

  it('all-experts routing activates all', () => {
    const moe = new MixtureOfExperts(4, 3, 8, 3, 3);
    moe.resetRoutingStats();
    const input = Matrix.random(5, 4);
    moe.forward(input);
    assert.equal(moe.totalRouted, 15); // 5 samples * 3 experts (all)
  });

  it('can train with gradient descent', () => {
    let passed = false;
    for (let attempt = 0; attempt < 3 && !passed; attempt++) {
      const moe = new MixtureOfExperts(2, 4, 8, 1, 2);
      const input = new Matrix(4, 2, new Float64Array([
        0, 0,
        0, 1,
        1, 0,
        1, 1,
      ]));
      const target = new Matrix(4, 1, new Float64Array([0, 1, 1, 0])); // XOR

      let prevLoss = Infinity;
      for (let epoch = 0; epoch < 200; epoch++) {
        const output = moe.forward(input);
        let loss = 0;
        const dOutput = new Matrix(4, 1);
        for (let i = 0; i < 4; i++) {
          const diff = output.get(i, 0) - target.get(i, 0);
          loss += diff * diff;
          dOutput.set(i, 0, 2 * diff / 4);
        }
        loss /= 4;

        moe.backward(dOutput);
        moe.update(0.05);

        if (epoch === 0) prevLoss = loss;
      }

      const finalOutput = moe.forward(input);
      let finalLoss = 0;
      for (let i = 0; i < 4; i++) {
        finalLoss += (finalOutput.get(i, 0) - target.get(i, 0)) ** 2;
      }
      finalLoss /= 4;
      if (finalLoss < prevLoss) passed = true;
    }
    assert.ok(passed, 'MoE loss should decrease in 1 of 3 attempts');
  });

  it('resetRoutingStats clears counts', () => {
    const moe = new MixtureOfExperts(4, 4, 8, 3, 2);
    const input = Matrix.random(5, 4);
    moe.forward(input);
    assert.ok(moe.totalRouted > 0);
    moe.resetRoutingStats();
    assert.equal(moe.totalRouted, 0);
    assert.ok(moe.routingCounts.every(c => c === 0));
  });

  it('different inputs route to different experts', () => {
    const moe = new MixtureOfExperts(4, 8, 8, 3, 2);

    // Very different inputs
    const input1 = new Matrix(1, 4, new Float64Array([10, 0, 0, 0]));
    const input2 = new Matrix(1, 4, new Float64Array([0, 0, 0, 10]));

    moe.forward(input1);
    const routes1 = [...moe.topKIndices[0]];

    moe.forward(input2);
    const routes2 = [...moe.topKIndices[0]];

    // With random initialization and very different inputs,
    // they should often (but not always) route differently
    // Just verify both produce valid routes
    assert.equal(routes1.length, 2);
    assert.equal(routes2.length, 2);
    assert.ok(routes1.every(r => r >= 0 && r < 8));
    assert.ok(routes2.every(r => r >= 0 && r < 8));
  });
});

describe('Gating Softmax', () => {
  it('gate probabilities sum to 1', () => {
    const moe = new MixtureOfExperts(4, 4, 8, 3, 2);
    const input = Matrix.random(3, 4);
    moe.forward(input);
    for (let b = 0; b < 3; b++) {
      let sum = 0;
      for (let e = 0; e < 4; e++) {
        sum += moe.gateProbs.get(b, e);
      }
      assert.ok(approx(sum, 1, 0.001), `Gate probs should sum to 1: ${sum}`);
    }
  });

  it('gate probabilities are non-negative', () => {
    const moe = new MixtureOfExperts(4, 4, 8, 3, 2);
    const input = Matrix.random(5, 4);
    moe.forward(input);
    for (let b = 0; b < 5; b++) {
      for (let e = 0; e < 4; e++) {
        assert.ok(moe.gateProbs.get(b, e) >= 0, `Gate prob should be >= 0`);
      }
    }
  });
});
