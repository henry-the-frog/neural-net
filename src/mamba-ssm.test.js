// mamba-ssm.test.js
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { SelectiveSSM } from './mamba-ssm.js';
import { Matrix } from './matrix.js';

describe('Mamba Selective SSM', () => {
  test('forward produces correct shape', () => {
    const ssm = new SelectiveSSM(8, 4, 16);
    const x = Matrix.random(5, 8);
    const out = ssm.forward(x);
    assert.equal(out.rows, 5);
    assert.equal(out.cols, 8);
  });

  test('different inputs produce different outputs', () => {
    const ssm = new SelectiveSSM(4, 4, 8);
    const x1 = Matrix.random(3, 4);
    const x2 = Matrix.random(3, 4);
    const out1 = ssm.forward(x1);
    const out2 = ssm.forward(x2);
    
    let diff = 0;
    for (let i = 0; i < out1.data.length; i++) diff += Math.abs(out1.data[i] - out2.data[i]);
    assert.ok(diff > 0.01);
  });

  test('output depends on sequence order (not permutation invariant)', () => {
    const ssm = new SelectiveSSM(4, 4, 8);
    const x = Matrix.random(4, 4);
    const out1 = ssm.forward(x);
    
    // Reverse the sequence
    const xRev = new Matrix(4, 4);
    for (let t = 0; t < 4; t++) {
      for (let d = 0; d < 4; d++) xRev.set(t, d, x.get(3 - t, d));
    }
    const out2 = ssm.forward(xRev);
    
    let diff = 0;
    for (let i = 0; i < out1.data.length; i++) diff += Math.abs(out1.data[i] - out2.data[i]);
    assert.ok(diff > 0.01, 'Reversed input should produce different output (sequential model)');
  });

  test('step mode matches forward mode', () => {
    const ssm = new SelectiveSSM(4, 4, 8);
    const x = Matrix.random(3, 4);
    
    const fullOut = ssm.forward(x);
    
    let state = ssm.initState();
    const stepOutputs = [];
    for (let t = 0; t < 3; t++) {
      const xt = new Float64Array(4);
      for (let d = 0; d < 4; d++) xt[d] = x.get(t, d);
      const result = ssm.step(xt, state);
      stepOutputs.push(result.output);
      state = result.state;
    }
    
    for (let t = 0; t < 3; t++) {
      for (let d = 0; d < 4; d++) {
        assert.ok(Math.abs(fullOut.get(t, d) - stepOutputs[t][d]) < 1e-8,
          `Mismatch at t=${t}, d=${d}: ${fullOut.get(t, d)} vs ${stepOutputs[t][d]}`);
      }
    }
  });

  test('state is O(dInner * dState) — constant per token', () => {
    const ssm = new SelectiveSSM(8, 4, 16);
    let state = ssm.initState();
    
    for (let t = 0; t < 50; t++) {
      const xt = new Float64Array(8).map(() => Math.random());
      const result = ssm.step(xt, state);
      state = result.state;
    }
    
    assert.equal(state.length, 16); // dInner
    assert.equal(state[0].length, 4); // dState
  });

  test('outputs are finite', () => {
    const ssm = new SelectiveSSM(8, 8, 16);
    const x = Matrix.random(10, 8);
    const out = ssm.forward(x);
    for (let i = 0; i < out.data.length; i++) {
      assert.ok(isFinite(out.data[i]), `Output[${i}] = ${out.data[i]} is not finite`);
    }
  });
});
