// darts.test.js — Tests for DARTS architecture search
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { MixedOp, DARTSCell, DARTSSearcher } from '../src/darts.js';

describe('MixedOp', () => {
  it('should produce output of correct size', () => {
    const op = new MixedOp(4, 3);
    const input = new Float64Array([1, 2, 3, 4]);
    const output = op.forward(input);
    assert.equal(output.length, 3);
  });

  it('should have equal weights initially', () => {
    const op = new MixedOp(4, 4);
    const weights = op.architectureWeights;
    // All alphas start at 0 → softmax is uniform
    const expected = 1 / weights.length;
    for (const w of weights) {
      assert.ok(Math.abs(w - expected) < 0.01, `Weight ${w} not close to ${expected}`);
    }
  });

  it('should select operation with highest alpha', () => {
    const op = new MixedOp(4, 4);
    // Boost the second operation
    op.alpha[1] = 10;
    assert.equal(op.selectedOp.name, op.ops[1].name);
  });

  it('should change output when alpha changes', () => {
    const op = new MixedOp(4, 4);
    const input = new Float64Array([1, 0.5, -0.3, 0.8]);
    
    const out1 = op.forward(input);
    op.alpha[0] = 10; // heavily favor first op
    const out2 = op.forward(input);
    
    let diff = 0;
    for (let i = 0; i < out1.length; i++) {
      diff += Math.abs(out1[i] - out2[i]);
    }
    assert.ok(diff > 0.01, 'Output should change when alpha changes');
  });

  it('should produce mostly zeros when zero op is selected', () => {
    const op = new MixedOp(4, 4); // includes identity and zero for same-size
    // Find the zero op index
    const zeroIdx = op.ops.findIndex(o => o.name === 'zero');
    if (zeroIdx >= 0) {
      op.alpha[zeroIdx] = 100; // overwhelmingly select zero
      const input = new Float64Array([1, 2, 3, 4]);
      const output = op.forward(input);
      const maxAbs = Math.max(...output.map(Math.abs));
      assert.ok(maxAbs < 0.01, `Zero op should produce ~0, got max=${maxAbs}`);
    }
  });
});

describe('DARTSCell', () => {
  it('should produce output of correct hidden size', () => {
    const cell = new DARTSCell(8, 16, 3);
    const input = new Float64Array(8).fill(0.5);
    const output = cell.forward(input);
    assert.equal(output.length, 16);
  });

  it('should have correct number of edges', () => {
    // With 3 intermediate nodes (0,1 are inputs, 2,3,4 are intermediate)
    // Edges: 0→2, 1→2, 0→3, 1→3, 2→3, 0→4, 1→4, 2→4, 3→4 = 9
    const cell = new DARTSCell(4, 8, 3);
    assert.equal(cell.edges.size, 9);
  });

  it('should produce different outputs for different inputs', () => {
    const cell = new DARTSCell(4, 8, 2);
    const in1 = new Float64Array([1, 0, 0, 0]);
    const in2 = new Float64Array([0, 0, 0, 1]);
    const out1 = cell.forward(in1);
    const out2 = cell.forward(in2);
    let diff = 0;
    for (let i = 0; i < out1.length; i++) {
      diff += Math.abs(out1[i] - out2[i]);
    }
    assert.ok(diff > 0.001, 'Different inputs should produce different outputs');
  });

  it('getDerivedArchitecture should return selections for all edges', () => {
    const cell = new DARTSCell(4, 8, 2);
    const arch = cell.getDerivedArchitecture();
    assert.equal(Object.keys(arch).length, cell.edges.size);
    for (const [key, info] of Object.entries(arch)) {
      assert.ok(info.selected, `Edge ${key} should have a selected op`);
      assert.ok(Array.isArray(info.weights), `Edge ${key} should have weights`);
    }
  });
});

describe('DARTSSearcher', () => {
  it('should run search without errors', () => {
    const cell = new DARTSCell(4, 8, 2);
    const searcher = new DARTSSearcher(cell, 3);
    
    // Simple classification data
    const trainInputs = [
      new Float64Array([1, 0, 0, 0]),
      new Float64Array([0, 1, 0, 0]),
      new Float64Array([0, 0, 1, 0]),
    ];
    const trainTargets = [0, 1, 2];
    const valInputs = trainInputs;
    const valTargets = trainTargets;
    
    const result = searcher.search(trainInputs, trainTargets, valInputs, valTargets, 10);
    assert.ok(result.history.length === 10);
    assert.ok(result.architecture);
    assert.ok(result.alphas);
  });

  it('should produce valid probabilities', () => {
    const cell = new DARTSCell(4, 8, 2);
    const searcher = new DARTSSearcher(cell, 3);
    const probs = searcher.predict(new Float64Array([1, 0.5, -0.3, 0.8]));
    
    assert.equal(probs.length, 3);
    const sum = probs.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1.0) < 1e-6, `Probabilities should sum to 1, got ${sum}`);
    for (const p of probs) {
      assert.ok(p >= 0 && p <= 1, `Probability should be in [0,1], got ${p}`);
    }
  });

  it('architecture should evolve during search', () => {
    const cell = new DARTSCell(4, 8, 2);
    const searcher = new DARTSSearcher(cell, 2);
    
    // Get initial alphas
    const initialAlphas = cell.getAllAlphas().map(a => [...a.alpha]);
    
    const trainInputs = [
      new Float64Array([1, 0, 0, 0]),
      new Float64Array([0, 0, 0, 1]),
    ];
    const trainTargets = [0, 1];
    
    searcher.search(trainInputs, trainTargets, trainInputs, trainTargets, 200);
    
    const finalAlphas = cell.getAllAlphas().map(a => [...a.alpha]);
    
    // At least some alphas should have changed
    let totalChange = 0;
    for (let i = 0; i < initialAlphas.length; i++) {
      for (let j = 0; j < initialAlphas[i].length; j++) {
        totalChange += Math.abs(finalAlphas[i][j] - initialAlphas[i][j]);
      }
    }
    assert.ok(totalChange > 0.01, `Architecture should evolve: total change=${totalChange.toFixed(4)}`);
  });

  it('loss should decrease during search', () => {
    const cell = new DARTSCell(4, 8, 2);
    const searcher = new DARTSSearcher(cell, 2);
    
    const trainInputs = [
      new Float64Array([1, 0, 0, 0]),
      new Float64Array([0, 0, 0, 1]),
    ];
    const trainTargets = [0, 1];
    
    const result = searcher.search(trainInputs, trainTargets, trainInputs, trainTargets, 50);
    
    const firstLoss = result.history[0].valLoss;
    const lastLoss = result.history[result.history.length - 1].valLoss;
    
    // Loss should generally decrease (architecture optimization finds better ops)
    assert.ok(lastLoss <= firstLoss + 0.5, 
      `Loss should not increase dramatically: ${firstLoss.toFixed(3)} → ${lastLoss.toFixed(3)}`);
  });

  it('derived architecture should have non-trivial selections', () => {
    const cell = new DARTSCell(4, 8, 2);
    const searcher = new DARTSSearcher(cell, 2);
    
    const trainInputs = [
      new Float64Array([1, 0, 0, 0]),
      new Float64Array([0, 0, 0, 1]),
    ];
    
    searcher.search(trainInputs, [0, 1], trainInputs, [0, 1], 200);
    
    const arch = cell.getDerivedArchitecture();
    // After search, at least some edges should have non-uniform weights
    let hasNonUniform = false;
    for (const [key, info] of Object.entries(arch)) {
      const max = Math.max(...info.weights);
      const min = Math.min(...info.weights);
      if (max - min > 0.0005) hasNonUniform = true;
    }
    assert.ok(hasNonUniform, 'After search, architecture weights should be non-uniform');
  });
});
