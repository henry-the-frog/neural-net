// regression-bugs-apr17.test.js — Regression tests for bugs found April 17, 2026
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix } from '../src/matrix.js';
import { Network } from '../src/network.js';

describe('Bug Regression Tests (Apr 17, 2026)', () => {

  it('Bug #1: KANLayer backward produces finite gradients', async () => {
    const { KANLayer } = await import('../src/kan.js');
    const layer = new KANLayer(2, 3);
    const input = Matrix.fromArray([[0.5, 0.5]]);
    layer.forward(input);
    const dOut = Matrix.fromArray([[1, 1, 1]]);
    const dInput = layer.backward(dOut, 0.01);
    assert.ok(dInput.data.every(v => isFinite(v)), 'Gradients should be finite');
  });

  it('Bug #2: MoE produces different outputs for different inputs', async () => {
    const { MixtureOfExperts } = await import('../src/moe.js');
    const moe = new MixtureOfExperts(2, 3, 2, 2);
    const out1 = moe.forward(Matrix.fromArray([[1, 0]]));
    const out2 = moe.forward(Matrix.fromArray([[0, 1]]));
    let allSame = true;
    for (let i = 0; i < out1.data.length; i++) {
      if (Math.abs(out1.data[i] - out2.data[i]) > 1e-10) { allSame = false; break; }
    }
    assert.ok(!allSame, 'MoE should produce different outputs for different inputs');
  });

  it('Bug #3: MoE batch produces finite outputs', async () => {
    const { MixtureOfExperts } = await import('../src/moe.js');
    const moe = new MixtureOfExperts(2, 3, 2, 2);
    const batch = Matrix.fromArray([[1, 0], [0, 1], [0.5, 0.5]]);
    const out = moe.forward(batch);
    assert.ok(out.data.every(v => isFinite(v)), 'All outputs should be finite');
    assert.strictEqual(out.rows, 3, 'Should have 3 output rows');
  });

  it('Bug #4: CapsuleLayer produces finite output', async () => {
    const { CapsuleLayer } = await import('../src/capsule.js');
    const layer = new CapsuleLayer(3, 4, 2, 4, 3);
    const input = [[0.5, -0.3, 0.8, 0.1], [-0.2, 0.7, 0.4, -0.5]];
    const output = layer.forward(input);
    assert.ok(Array.isArray(output), 'Should return array of capsules');
    assert.strictEqual(output.length, 3, 'Should have 3 output capsules');
    assert.ok(output.every(c => c.every(v => isFinite(v))), 'All values should be finite');
  });

  it('Bug #5: NeuralODE backward produces correct-shaped gradients', async () => {
    const { NeuralODELayer } = await import('../src/neural-ode.js');
    const layer = new NeuralODELayer(4, 8);
    const input = Matrix.random(1, 4);
    layer.forward(input);
    const dOut = Matrix.random(1, 4);
    const dInput = layer.backward(dOut, 0.01);
    assert.ok(dInput.data.every(v => isFinite(v)), 'Gradients should be finite');
    assert.strictEqual(dInput.cols, 4);
  });

  it('Bug #6: Autograd MSE loss is finite', async () => {
    const { mseLoss } = await import('../src/autograd.js');
    // Bug: mseLoss([1,2],[1.5,2.5]) returned NaN because plain numbers weren't wrapped in Variables
    const loss = mseLoss([1, 2], [1.5, 2.5]);
    assert.ok(isFinite(loss.value), 'MSE should be finite, got ' + loss.value);
    assert.ok(Math.abs(loss.value - 0.25) < 1e-10, 'MSE should be 0.25');
  });

  it('Bug #7: Cutmix does not crash', async () => {
    const { cutmix } = await import('../src/data-augmentation.js');
    const d1 = Matrix.fromArray([[1, 2, 3, 4]]);
    const d2 = Matrix.fromArray([[5, 6, 7, 8]]);
    const l1 = Matrix.fromArray([[1, 0]]);
    const l2 = Matrix.fromArray([[0, 1]]);
    const result = cutmix(d1, d2, l1, l2, 0.5);
    assert.ok(result, 'Should return a result');
  });

  it('Bug #8-9: Pruning returns proper Matrix', async () => {
    const { magnitudePrune, countSparsity } = await import('../src/pruning.js');
    const weights = Matrix.random(4, 4);
    const pruned = magnitudePrune(weights, 0.5);
    assert.ok(pruned instanceof Matrix, 'Should return a Matrix instance');
    assert.ok(typeof pruned.get === 'function', 'Should have get method');
    const sp = countSparsity(pruned);
    assert.ok(sp >= 0.3 && sp <= 0.7, `Sparsity should be ~0.5, got ${sp}`);
  });

  it('Bug #8-9: Pruned weights work in forward pass', async () => {
    const { magnitudePrune } = await import('../src/pruning.js');
    const net = new Network();
    net.dense(2, 4, 'relu').dense(4, 1, 'sigmoid');
    const input = Matrix.fromArray([[0.5, 0.5]]);
    net.layers[0].weights = magnitudePrune(net.layers[0].weights, 0.5);
    const pred = net.predict(input);
    assert.ok(isFinite(pred.get(0, 0)), 'Should produce finite prediction');
  });
});
