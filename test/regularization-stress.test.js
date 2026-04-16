// regularization-stress.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { l1Regularization, l2Regularization, elasticNet, weightDecay, maxNormConstraint, spectralNorm, gradientClipping } from '../src/regularization.js';
import { Matrix } from '../src/matrix.js';

describe('Regularization Stress', () => {
  it('L1 regularization: zero weights have zero penalty', () => {
    const weights = new Matrix(1, 4, new Float64Array([0, 0, 0, 0]));
    const { penalty, gradient } = l1Regularization(weights, 0.01);
    assert.equal(penalty, 0, 'Zero weights = zero L1 penalty');
  });

  it('L1 regularization: penalty proportional to weight magnitude', () => {
    const w1 = new Matrix(1, 4, new Float64Array([1, 0, 0, 0]));
    const w2 = new Matrix(1, 4, new Float64Array([2, 0, 0, 0]));
    const p1 = l1Regularization(w1, 0.01).penalty;
    const p2 = l1Regularization(w2, 0.01).penalty;
    assert.ok(p2 > p1, `Larger weight = larger penalty: ${p1} vs ${p2}`);
  });

  it('L2 regularization: penalty proportional to weight squared', () => {
    const w1 = new Matrix(1, 4, new Float64Array([1, 0, 0, 0]));
    const w2 = new Matrix(1, 4, new Float64Array([2, 0, 0, 0]));
    const p1 = l2Regularization(w1, 0.01).penalty;
    const p2 = l2Regularization(w2, 0.01).penalty;
    assert.ok(p2 > p1 * 3, `L2 penalty should scale quadratically: ${p1} vs ${p2}`);
  });

  it('elastic net combines L1 and L2', () => {
    const weights = new Matrix(1, 4, new Float64Array([1, -1, 2, -2]));
    const result = elasticNet(weights, 0.01, 0.5); // alpha=0.5
    assert.ok(isFinite(result.penalty), 'Elastic net penalty should be finite');
    assert.ok(result.penalty > 0, 'Non-zero weights should have positive penalty');
  });

  it('weight decay modifies weights', () => {
    const weights = new Matrix(1, 4, new Float64Array([1, 2, 3, 4]));
    const decayed = weightDecay(weights, 0.01);
    for (let i = 0; i < 4; i++) {
      assert.ok(decayed.data[i] < weights.data[i], `Weight should decrease: ${weights.data[i]} → ${decayed.data[i]}`);
    }
  });

  it('gradient clipping limits magnitude', () => {
    const grad = new Matrix(1, 4, new Float64Array([10, -20, 30, -40]));
    const clipped = gradientClipping(grad, 5.0);
    // Norm of clipped should be <= 5.0
    let norm = 0;
    for (let i = 0; i < 4; i++) norm += clipped.data[i] ** 2;
    norm = Math.sqrt(norm);
    assert.ok(norm <= 5.1, `Clipped norm should be <= 5.0: ${norm}`);
  });

  it('gradient clipping preserves small gradients', () => {
    const grad = new Matrix(1, 4, new Float64Array([0.1, 0.2, 0.1, 0.2]));
    const clipped = gradientClipping(grad, 5.0);
    for (let i = 0; i < 4; i++) {
      assert.ok(Math.abs(clipped.data[i] - grad.data[i]) < 1e-6, 'Small gradients should not be clipped');
    }
  });

  it('all regularizations produce finite values', () => {
    const w = Matrix.random(3, 4);
    const results = [
      l1Regularization(w, 0.01),
      l2Regularization(w, 0.01),
      elasticNet(w, 0.01, 0.5),
    ];
    for (const r of results) {
      assert.ok(isFinite(r.penalty), 'Penalty should be finite');
      for (let i = 0; i < r.gradient.data.length; i++) {
        assert.ok(isFinite(r.gradient.data[i]), 'Gradient should be finite');
      }
    }
  });
});
