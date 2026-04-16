// layernorm-stress.test.js — Verify LayerNorm backward correctness
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { LayerNorm } from '../src/transformer.js';
import { Matrix } from '../src/matrix.js';

function numGradInput(norm, input, dOutput, b, idx, eps = 1e-5) {
  const orig = input.get(b, idx);
  
  input.set(b, idx, orig + eps);
  const outPlus = norm.forward(input);
  let lossPlus = 0;
  for (let r = 0; r < outPlus.rows; r++)
    for (let c = 0; c < outPlus.cols; c++)
      lossPlus += outPlus.get(r, c) * dOutput.get(r, c);
  
  input.set(b, idx, orig - eps);
  const outMinus = norm.forward(input);
  let lossMinus = 0;
  for (let r = 0; r < outMinus.rows; r++)
    for (let c = 0; c < outMinus.cols; c++)
      lossMinus += outMinus.get(r, c) * dOutput.get(r, c);
  
  input.set(b, idx, orig);
  return (lossPlus - lossMinus) / (2 * eps);
}

function relErr(a, n) {
  return Math.abs(a - n) / Math.max(Math.abs(a), Math.abs(n), 1e-8);
}

describe('LayerNorm Gradient Correctness', () => {
  it('input gradient matches numerical gradient', () => {
    const norm = new LayerNorm(4);
    const input = new Matrix(1, 4, new Float64Array([1.0, 2.0, 3.0, 4.0]));
    norm.forward(input);
    const dOutput = new Matrix(1, 4, new Float64Array([0.1, 0.2, -0.1, 0.3]));
    const dInput = norm.backward(dOutput);
    
    let maxErr = 0;
    let details = [];
    for (let d = 0; d < 4; d++) {
      const ng = numGradInput(norm, input, dOutput, 0, d);
      const ag = dInput.get(0, d);
      const err = relErr(ag, ng);
      maxErr = Math.max(maxErr, err);
      details.push(`d=${d}: analytical=${ag.toFixed(6)}, numerical=${ng.toFixed(6)}, err=${err.toExponential(2)}`);
    }
    
    if (maxErr > 0.01) {
      console.log('  ⚠️ BUG: LayerNorm backward is INCORRECT');
      for (const d of details) console.log('    ' + d);
    }
    
    assert.ok(maxErr < 0.01, `LayerNorm input gradient error too high: ${maxErr.toExponential(2)}`);
  });

  it('gamma gradient matches numerical gradient', () => {
    const norm = new LayerNorm(4);
    const input = new Matrix(1, 4, new Float64Array([1.0, 2.0, 3.0, 4.0]));
    const dOutput = new Matrix(1, 4, new Float64Array([0.1, 0.2, -0.1, 0.3]));
    
    // Compute analytical gradients
    norm.forward(input);
    norm.backward(dOutput);
    
    // Numerical gradient for gamma
    let maxErr = 0;
    for (let d = 0; d < 4; d++) {
      const orig = norm.gamma.get(0, d);
      
      norm.gamma.set(0, d, orig + 1e-5);
      const outPlus = norm.forward(input);
      let lossPlus = 0;
      for (let c = 0; c < 4; c++) lossPlus += outPlus.get(0, c) * dOutput.get(0, c);
      
      norm.gamma.set(0, d, orig - 1e-5);
      const outMinus = norm.forward(input);
      let lossMinus = 0;
      for (let c = 0; c < 4; c++) lossMinus += outMinus.get(0, c) * dOutput.get(0, c);
      
      norm.gamma.set(0, d, orig);
      
      const ng = (lossPlus - lossMinus) / (2e-5);
      const ag = norm.dWeights.get(0, d);
      const err = relErr(ag, ng);
      maxErr = Math.max(maxErr, err);
    }
    
    assert.ok(maxErr < 0.01, `Gamma gradient max error: ${maxErr.toExponential(2)}`);
  });

  it('output is normalized (mean ~0, std ~1)', () => {
    const norm = new LayerNorm(8);
    const input = new Matrix(2, 8);
    for (let i = 0; i < 16; i++) input.data[i] = i * 0.5 - 3;
    const output = norm.forward(input);
    
    for (let b = 0; b < 2; b++) {
      let mean = 0;
      for (let d = 0; d < 8; d++) mean += output.get(b, d);
      mean /= 8;
      assert.ok(Math.abs(mean) < 1e-5, `Row ${b} mean should be ~0: ${mean}`);
      
      let variance = 0;
      for (let d = 0; d < 8; d++) variance += (output.get(b, d) - mean) ** 2;
      variance /= 8;
      assert.ok(Math.abs(variance - 1) < 0.1, `Row ${b} variance should be ~1: ${variance}`);
    }
  });

  it('multi-position sequence', () => {
    const norm = new LayerNorm(4);
    const input = Matrix.random(2, 12); // batch=2, seq=3, dModel=4
    const output = norm.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 12);
  });
});
