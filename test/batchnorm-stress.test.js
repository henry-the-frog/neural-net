// batchnorm-stress.test.js — Numerical gradient verification for BatchNorm
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { BatchNorm } from '../src/batchnorm.js';
import { Matrix } from '../src/matrix.js';

function relErr(a, n) {
  return Math.abs(a - n) / Math.max(Math.abs(a), Math.abs(n), 1e-8);
}

describe('BatchNorm Gradient Correctness', () => {
  it('input gradient matches numerical gradient', () => {
    const bn = new BatchNorm(3);
    const input = new Matrix(4, 3, new Float64Array([
      1.0, 2.0, 3.0,
      4.0, 5.0, 6.0,
      7.0, 8.0, 9.0,
      10.0, 11.0, 12.0,
    ]));
    const dOutput = new Matrix(4, 3, new Float64Array([
      0.1, -0.2, 0.3,
      -0.1, 0.4, 0.1,
      0.2, -0.1, -0.2,
      0.3, 0.1, -0.3,
    ]));
    
    bn.forward(input);
    const dInput = bn.backward(dOutput);
    
    let maxErr = 0;
    const eps = 1e-5;
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 3; j++) {
        const orig = input.get(i, j);
        
        input.set(i, j, orig + eps);
        const outPlus = bn.forward(input);
        let lossPlus = 0;
        for (let r = 0; r < 4; r++)
          for (let c = 0; c < 3; c++)
            lossPlus += outPlus.get(r, c) * dOutput.get(r, c);
        
        input.set(i, j, orig - eps);
        const outMinus = bn.forward(input);
        let lossMinus = 0;
        for (let r = 0; r < 4; r++)
          for (let c = 0; c < 3; c++)
            lossMinus += outMinus.get(r, c) * dOutput.get(r, c);
        
        input.set(i, j, orig);
        
        const ng = (lossPlus - lossMinus) / (2 * eps);
        const ag = dInput.get(i, j);
        const err = relErr(ag, ng);
        maxErr = Math.max(maxErr, err);
      }
    }
    
    assert.ok(maxErr < 0.01, `BatchNorm input gradient max error: ${maxErr.toExponential(2)}`);
  });

  it('gamma gradient matches numerical gradient', () => {
    const bn = new BatchNorm(3);
    const input = Matrix.random(4, 3);
    const dOutput = Matrix.random(4, 3);
    
    bn.forward(input);
    bn.backward(dOutput);
    
    let maxErr = 0;
    const eps = 1e-5;
    for (let d = 0; d < 3; d++) {
      const orig = bn.gamma.get(0, d);
      
      bn.gamma.set(0, d, orig + eps);
      const outPlus = bn.forward(input);
      let lossPlus = 0;
      for (let r = 0; r < 4; r++)
        for (let c = 0; c < 3; c++)
          lossPlus += outPlus.get(r, c) * dOutput.get(r, c);
      
      bn.gamma.set(0, d, orig - eps);
      const outMinus = bn.forward(input);
      let lossMinus = 0;
      for (let r = 0; r < 4; r++)
        for (let c = 0; c < 3; c++)
          lossMinus += outMinus.get(r, c) * dOutput.get(r, c);
      
      bn.gamma.set(0, d, orig);
      
      const ng = (lossPlus - lossMinus) / (2 * eps);
      const ag = bn.dGamma.get(0, d);
      const err = relErr(ag, ng);
      maxErr = Math.max(maxErr, err);
    }
    
    assert.ok(maxErr < 0.01, `BatchNorm gamma gradient max error: ${maxErr.toExponential(2)}`);
  });

  it('output is normalized per feature', () => {
    const bn = new BatchNorm(4);
    const input = Matrix.random(8, 4);
    const output = bn.forward(input);
    
    for (let j = 0; j < 4; j++) {
      let mean = 0;
      for (let i = 0; i < 8; i++) mean += output.get(i, j);
      mean /= 8;
      assert.ok(Math.abs(mean) < 0.1, `Feature ${j} mean should be ~0: ${mean}`);
    }
  });
});
