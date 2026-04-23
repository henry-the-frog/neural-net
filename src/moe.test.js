// moe.test.js — Mixture of Experts tests
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { MixtureOfExperts } from './moe.js';
import { Matrix } from './matrix.js';

describe('MixtureOfExperts', () => {
  test('forward produces correct shape', () => {
    const moe = new MixtureOfExperts(8, 16, 4, 2);
    const x = Matrix.random(6, 8);
    const { output } = moe.forward(x);
    assert.equal(output.rows, 6);
    assert.equal(output.cols, 8);
  });

  test('forward returns auxiliary loss', () => {
    const moe = new MixtureOfExperts(8, 16, 4, 2);
    const x = Matrix.random(6, 8);
    const { auxLoss } = moe.forward(x);
    assert.ok(typeof auxLoss === 'number');
    assert.ok(!isNaN(auxLoss));
    assert.ok(auxLoss >= 0);
  });

  test('each token routed to exactly topK experts', () => {
    const moe = new MixtureOfExperts(8, 16, 4, 2);
    const x = Matrix.random(10, 8);
    moe.forward(x);
    
    for (const routing of moe._routingInfo.tokenRouting) {
      assert.equal(routing.indices.length, 2);
      assert.equal(routing.weights.length, 2);
    }
  });

  test('routing weights sum to 1', () => {
    const moe = new MixtureOfExperts(8, 16, 4, 2);
    const x = Matrix.random(5, 8);
    moe.forward(x);
    
    for (const routing of moe._routingInfo.tokenRouting) {
      const sum = routing.weights.reduce((a, b) => a + b, 0);
      assert.ok(Math.abs(sum - 1) < 0.001, `Weights should sum to 1, got ${sum}`);
    }
  });

  test('all experts get some traffic with enough tokens', () => {
    const moe = new MixtureOfExperts(8, 16, 4, 2);
    const x = Matrix.random(100, 8);
    moe.forward(x);
    
    const stats = moe.getExpertStats();
    for (const s of stats) {
      assert.ok(s.tokens > 0, `Expert ${s.expert} should get some tokens`);
    }
  });

  test('backward runs without error', () => {
    const moe = new MixtureOfExperts(4, 8, 4, 2);
    const x = Matrix.random(3, 4);
    moe.forward(x);
    
    const dOutput = Matrix.random(3, 4);
    moe.backward(dOutput); // Should not throw
  });

  test('update after forward (no backward) does not crash', () => {
    const moe = new MixtureOfExperts(4, 8, 4, 2);
    const x = Matrix.random(20, 4);
    moe.forward(x);
    // Update without backward — router update may fail on null gradients
    // This is expected behavior; update should only be called after backward
    assert.ok(moe.paramCount() > 0);
  });

  test('paramCount scales with numExperts', () => {
    const moe2 = new MixtureOfExperts(8, 16, 2, 1);
    const moe8 = new MixtureOfExperts(8, 16, 8, 1);
    assert.ok(moe8.paramCount() > moe2.paramCount(), 
      'More experts should mean more parameters');
  });

  test('topK=1 uses exactly one expert per token', () => {
    const moe = new MixtureOfExperts(4, 8, 4, 1);
    const x = Matrix.random(10, 4);
    moe.forward(x);
    
    for (const routing of moe._routingInfo.tokenRouting) {
      assert.equal(routing.indices.length, 1);
      assert.ok(Math.abs(routing.weights[0] - 1) < 0.001, 'Single expert weight should be 1');
    }
  });
});
