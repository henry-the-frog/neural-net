// rwkv.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { RWKVTimeBlock } from './rwkv.js';
import { Matrix } from './matrix.js';

describe('RWKV', () => {
  test('forward produces correct shape', () => {
    const rwkv = new RWKVTimeBlock(8);
    const x = Matrix.random(5, 8);
    const out = rwkv.forward(x);
    assert.equal(out.rows, 5);
    assert.equal(out.cols, 8);
  });

  test('output is non-zero', () => {
    const rwkv = new RWKVTimeBlock(8);
    const x = Matrix.random(3, 8);
    const out = rwkv.forward(x);
    let nonZero = 0;
    for (let i = 0; i < out.data.length; i++) {
      if (Math.abs(out.data[i]) > 1e-6) nonZero++;
    }
    assert.ok(nonZero > out.data.length * 0.3, 'Most outputs should be non-zero');
  });

  test('different inputs produce different outputs', () => {
    const rwkv = new RWKVTimeBlock(8);
    const x1 = Matrix.random(3, 8);
    const x2 = Matrix.random(3, 8);
    const out1 = rwkv.forward(x1);
    const out2 = rwkv.forward(x2);
    
    let diff = 0;
    for (let i = 0; i < out1.data.length; i++) diff += Math.abs(out1.data[i] - out2.data[i]);
    assert.ok(diff > 0.01, 'Different inputs should give different outputs');
  });

  test('step mode matches forward mode', () => {
    const rwkv = new RWKVTimeBlock(4);
    const x = Matrix.random(3, 4);
    
    // Full forward
    const fullOut = rwkv.forward(x);
    
    // Step by step
    let state = rwkv.initState();
    const stepOutputs = [];
    for (let t = 0; t < 3; t++) {
      const xt = new Float64Array(4);
      for (let d = 0; d < 4; d++) xt[d] = x.get(t, d);
      const result = rwkv.step(xt, state);
      stepOutputs.push(result.output);
      state = result.state;
    }
    
    // Compare — should be identical
    for (let t = 0; t < 3; t++) {
      for (let d = 0; d < 4; d++) {
        const fullVal = fullOut.get(t, d);
        const stepVal = stepOutputs[t][d];
        assert.ok(Math.abs(fullVal - stepVal) < 1e-8,
          `Mismatch at t=${t}, d=${d}: full=${fullVal}, step=${stepVal}`);
      }
    }
  });

  test('step state evolves', () => {
    const rwkv = new RWKVTimeBlock(4);
    let state = rwkv.initState();
    
    const xt = new Float64Array([1, 0, 0, 0]);
    const result1 = rwkv.step(xt, state);
    const result2 = rwkv.step(xt, result1.state);
    
    // State should evolve (a and b should accumulate)
    let stateChanged = false;
    for (let d = 0; d < 4; d++) {
      if (Math.abs(result1.state.a[d] - result2.state.a[d]) > 1e-10) stateChanged = true;
    }
    assert.ok(stateChanged, 'State should evolve over steps');
  });

  test('O(1) memory per token in step mode', () => {
    const rwkv = new RWKVTimeBlock(8);
    let state = rwkv.initState();
    
    // Process 100 tokens — state size should remain constant
    for (let t = 0; t < 100; t++) {
      const xt = new Float64Array(8).map(() => Math.random());
      const result = rwkv.step(xt, state);
      state = result.state;
    }
    
    // State is always 2 arrays of length dim
    assert.equal(state.a.length, 8);
    assert.equal(state.b.length, 8);
  });
});
