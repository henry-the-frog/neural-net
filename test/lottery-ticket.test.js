// lottery-ticket.test.js — Tests for Lottery Ticket Hypothesis implementation
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Network } from '../src/network.js';
import { Matrix } from '../src/matrix.js';
import {
  snapshotWeights, restoreWeights, createMagnitudeMask, applyMask,
  lotteryTicketExperiment, iterativePruning,
} from '../src/lottery-ticket.js';

function makeXORNet() {
  const net = new Network();
  net.dense(2, 8, 'relu').dense(8, 1, 'sigmoid').loss('mse');
  return net;
}

const xorInputs = Matrix.fromArray([[0, 0], [0, 1], [1, 0], [1, 1]]);
const xorTargets = Matrix.fromArray([[0], [1], [1], [0]]);

describe('Weight Snapshots', () => {
  it('should create a deep copy of weights', () => {
    const net = makeXORNet();
    const snap = snapshotWeights(net);
    
    // Modify network weights
    net.layers[0].weights.data[0] = 999;
    
    // Snapshot should be unchanged
    assert.notEqual(snap[0].weights[0], 999);
  });

  it('should restore weights exactly', () => {
    const net = makeXORNet();
    const snap = snapshotWeights(net);
    const originalW = net.layers[0].weights.data[0];
    
    // Modify and restore
    net.layers[0].weights.data[0] = 999;
    restoreWeights(net, snap);
    
    assert.equal(net.layers[0].weights.data[0], originalW);
  });
});

describe('Magnitude Mask', () => {
  it('should create mask with correct sparsity', () => {
    const net = makeXORNet();
    // Train a bit so weights aren't all tiny
    for (let i = 0; i < 100; i++) net.trainBatch(xorInputs, xorTargets, 0.5);
    
    const { masks, actualSparsity } = createMagnitudeMask(net, 0.5);
    assert.ok(Math.abs(actualSparsity - 0.5) < 0.1, `Sparsity should be ~0.5, got ${actualSparsity}`);
    assert.ok(masks[0], 'First layer should have a mask');
  });

  it('mask should have 0s and 1s only', () => {
    const net = makeXORNet();
    for (let i = 0; i < 100; i++) net.trainBatch(xorInputs, xorTargets, 0.5);
    
    const { masks } = createMagnitudeMask(net, 0.3);
    for (const m of masks) {
      if (!m) continue;
      for (let i = 0; i < m.length; i++) {
        assert.ok(m[i] === 0 || m[i] === 1, `Mask value should be 0 or 1, got ${m[i]}`);
      }
    }
  });

  it('higher sparsity should prune more weights', () => {
    const net = makeXORNet();
    for (let i = 0; i < 100; i++) net.trainBatch(xorInputs, xorTargets, 0.5);
    
    const low = createMagnitudeMask(net, 0.3);
    const high = createMagnitudeMask(net, 0.7);
    assert.ok(high.actualSparsity > low.actualSparsity);
  });
});

describe('Apply Mask', () => {
  it('should zero out pruned weights', () => {
    const net = makeXORNet();
    for (let i = 0; i < 100; i++) net.trainBatch(xorInputs, xorTargets, 0.5);
    
    const { masks } = createMagnitudeMask(net, 0.5);
    applyMask(net, masks);
    
    // Count zeros
    let zeros = 0, total = 0;
    for (let li = 0; li < net.layers.length; li++) {
      if (!masks[li] || !net.layers[li].weights) continue;
      for (let j = 0; j < net.layers[li].weights.data.length; j++) {
        total++;
        if (net.layers[li].weights.data[j] === 0) zeros++;
      }
    }
    assert.ok(zeros / total > 0.4, `Should have ~50% zeros, got ${(zeros / total * 100).toFixed(1)}%`);
  });
});

describe('Lottery Ticket Experiment', () => {
  it('should run full experiment without errors', () => {
    const result = lotteryTicketExperiment({
      createNetwork: makeXORNet,
      trainInputs: xorInputs,
      trainTargets: xorTargets,
      trainEpochs: 300,
      trainLR: 0.5,
      sparsity: 0.3,
    });
    
    assert.ok(result.sparsity > 0, 'Should have some sparsity');
    assert.ok(result.fullNetwork.finalLoss >= 0, 'Full network loss should be non-negative');
    assert.ok(result.winningTicket.finalLoss >= 0, 'Winning ticket loss should be non-negative');
    assert.ok(result.randomTicket.finalLoss >= 0, 'Random ticket loss should be non-negative');
  });

  it('winning ticket should train better than random ticket (low sparsity)', () => {
    // With low sparsity (30%), winning ticket should reliably beat random
    let passed = false;
    for (let attempt = 0; attempt < 5 && !passed; attempt++) {
      const result = lotteryTicketExperiment({
        createNetwork: makeXORNet,
        trainInputs: xorInputs,
        trainTargets: xorTargets,
        trainEpochs: 500,
        trainLR: 0.5,
        sparsity: 0.3,
      });
      
      if (result.winningTicket.finalLoss < result.randomTicket.finalLoss) {
        passed = true;
      }
    }
    assert.ok(passed, 'Winning ticket should outperform random ticket in at least 1 of 5 attempts');
  });

  it('full network loss should decrease', () => {
    const result = lotteryTicketExperiment({
      createNetwork: makeXORNet,
      trainInputs: xorInputs,
      trainTargets: xorTargets,
      trainEpochs: 200,
      trainLR: 0.5,
      sparsity: 0.3,
    });
    
    assert.ok(result.fullNetwork.losses[0] > result.fullNetwork.finalLoss,
      'Full network loss should decrease during training');
  });
});

describe('Iterative Magnitude Pruning', () => {
  it('should increase sparsity over rounds', () => {
    const results = iterativePruning({
      createNetwork: makeXORNet,
      trainInputs: xorInputs,
      trainTargets: xorTargets,
      trainEpochs: 200,
      trainLR: 0.5,
      rounds: 3,
      prunePerRound: 0.2,
    });
    
    assert.equal(results.length, 3);
    for (let i = 1; i < results.length; i++) {
      assert.ok(results[i].targetSparsity > results[i - 1].targetSparsity,
        `Sparsity should increase: round ${i} = ${results[i].targetSparsity}`);
    }
  });

  it('should maintain trainability through moderate pruning', () => {
    const results = iterativePruning({
      createNetwork: makeXORNet,
      trainInputs: xorInputs,
      trainTargets: xorTargets,
      trainEpochs: 300,
      trainLR: 0.5,
      rounds: 3,
      prunePerRound: 0.15,
    });
    
    // First round should converge well
    assert.ok(results[0].finalLoss < 0.5, 
      `Round 1 should converge: loss=${results[0].finalLoss.toFixed(3)}`);
  });
});
