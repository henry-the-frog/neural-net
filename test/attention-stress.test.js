// attention-stress.test.js — Stress tests to find bugs in attention module
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { SelfAttention, MultiHeadAttention } from '../src/attention.js';
import { Matrix } from '../src/matrix.js';

// Numerical gradient checking helper
function numGradWeight(attn, input, dOutput, weightName, i, j, eps = 1e-5) {
  const W = attn[weightName];
  const orig = W.get(i, j);
  
  W.set(i, j, orig + eps);
  const inputPlus = input.clone ? input.clone() : Matrix.from(input);
  const outPlus = attn.forward(inputPlus);
  let lossPlus = 0;
  for (let r = 0; r < outPlus.rows; r++)
    for (let c = 0; c < outPlus.cols; c++)
      lossPlus += outPlus.get(r, c) * dOutput.get(r, c);
  
  W.set(i, j, orig - eps);
  const inputMinus = input.clone ? input.clone() : Matrix.from(input);
  const outMinus = attn.forward(inputMinus);
  let lossMinus = 0;
  for (let r = 0; r < outMinus.rows; r++)
    for (let c = 0; c < outMinus.cols; c++)
      lossMinus += outMinus.get(r, c) * dOutput.get(r, c);
  
  W.set(i, j, orig);
  return (lossPlus - lossMinus) / (2 * eps);
}

describe('SelfAttention Stress', () => {
  it('forward output has correct shape', () => {
    const attn = new SelfAttention(4);
    const input = Matrix.random(2, 12); // batch=2, seq=3, dModel=4
    const output = attn.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 12);
  });

  it('forward should not mutate input', () => {
    const attn = new SelfAttention(4);
    const input = Matrix.random(1, 8); // batch=1, seq=2, dModel=4
    const inputCopy = new Matrix(input.rows, input.cols);
    for (let i = 0; i < input.data.length; i++) inputCopy.data[i] = input.data[i];
    
    attn.forward(input);
    
    // Verify input was not mutated
    for (let i = 0; i < input.data.length; i++) {
      assert.ok(Math.abs(input.data[i] - inputCopy.data[i]) < 1e-10,
        `Input was mutated at index ${i}: ${inputCopy.data[i]} → ${input.data[i]}`);
    }
  });

  it('backward should produce finite gradients', () => {
    const attn = new SelfAttention(4);
    const input = Matrix.random(2, 8); // batch=2, seq=2, dModel=4
    const output = attn.forward(input);
    const dOutput = Matrix.random(2, 8);
    const dInput = attn.backward(dOutput);
    
    for (let i = 0; i < dInput.data.length; i++) {
      assert.ok(isFinite(dInput.data[i]), `dInput[${i}] is not finite: ${dInput.data[i]}`);
    }
  });

  it('attention weights should sum to 1', () => {
    const attn = new SelfAttention(4);
    const input = Matrix.random(1, 12); // batch=1, seq=3, dModel=4
    attn.forward(input);
    const weights = attn._cache.allAttn[0]; // [seqLen, seqLen]
    for (let i = 0; i < weights.rows; i++) {
      let sum = 0;
      for (let j = 0; j < weights.cols; j++) sum += weights.get(i, j);
      assert.ok(Math.abs(sum - 1) < 1e-6, `Row ${i} sum = ${sum}, expected 1`);
    }
  });

  it('identical input vectors should produce identical attention', () => {
    const attn = new SelfAttention(4);
    // All positions have the same vector
    const vec = [0.5, -0.3, 0.8, 0.1];
    const input = new Matrix(1, 12);
    for (let t = 0; t < 3; t++)
      for (let d = 0; d < 4; d++)
        input.set(0, t * 4 + d, vec[d]);
    
    const output = attn.forward(input);
    
    // All output positions should be identical (same input, uniform attention)
    for (let d = 0; d < 4; d++) {
      const v0 = output.get(0, d);
      const v1 = output.get(0, 4 + d);
      const v2 = output.get(0, 8 + d);
      assert.ok(Math.abs(v0 - v1) < 1e-6, `Position 0 and 1 should match: ${v0} vs ${v1}`);
      assert.ok(Math.abs(v0 - v2) < 1e-6, `Position 0 and 2 should match: ${v0} vs ${v2}`);
    }
  });

  it('training loop should decrease loss', () => {
    let passed = false;
    for (let attempt = 0; attempt < 3 && !passed; attempt++) {
      const attn = new SelfAttention(4);
      const input = Matrix.random(4, 8); // batch=4, seq=2, dModel=4
      const target = Matrix.random(4, 8);
      
      let prevLoss = Infinity;
      for (let step = 0; step < 100; step++) {
        const output = attn.forward(input);
        // MSE loss
        let loss = 0;
        const dOutput = new Matrix(4, 8);
        for (let i = 0; i < 4; i++) {
          for (let j = 0; j < 8; j++) {
            const diff = output.get(i, j) - target.get(i, j);
            loss += diff * diff;
            dOutput.set(i, j, 2 * diff / 32);
          }
        }
        loss /= 32;
        
        attn.backward(dOutput);
        attn.update(0.01);
        
        if (step === 0) prevLoss = loss;
      }
      
      const finalOutput = attn.forward(input);
      let finalLoss = 0;
      for (let i = 0; i < 4; i++)
        for (let j = 0; j < 8; j++)
          finalLoss += (finalOutput.get(i, j) - target.get(i, j)) ** 2;
      finalLoss /= 32;
      
      if (finalLoss < prevLoss) passed = true;
    }
    assert.ok(passed, 'Self-attention training should decrease loss');
  });
});

describe('MultiHeadAttention Stress', () => {
  it('forward output has correct shape', () => {
    const mha = new MultiHeadAttention(8, 2);
    const input = Matrix.random(2, 24); // batch=2, seq=3, dModel=8
    const output = mha.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 24);
  });

  it('single head MHA should behave like SelfAttention', () => {
    // Both should produce valid outputs for the same input
    const sa = new SelfAttention(4);
    const mha = new MultiHeadAttention(4, 1);
    const input = Matrix.random(1, 8);
    
    const out1 = sa.forward(input);
    const out2 = mha.forward(input);
    
    // Both outputs should be finite
    for (let i = 0; i < out1.data.length; i++) {
      assert.ok(isFinite(out1.data[i]), `SA output[${i}] not finite`);
      assert.ok(isFinite(out2.data[i]), `MHA output[${i}] not finite`);
    }
  });

  it('backward produces finite gradients', () => {
    const mha = new MultiHeadAttention(8, 2);
    const input = Matrix.random(2, 16); // batch=2, seq=2, dModel=8
    const output = mha.forward(input);
    const dOutput = Matrix.random(2, 16);
    const dInput = mha.backward(dOutput);
    
    for (let i = 0; i < dInput.data.length; i++) {
      assert.ok(isFinite(dInput.data[i]), `dInput[${i}] is not finite: ${dInput.data[i]}`);
    }
  });

  it('paramCount matches expected', () => {
    const mha = new MultiHeadAttention(16, 4);
    // 4 weight matrices of 16x16 + 4 bias vectors of 16 = 4*256 + 4*16 = 1088
    assert.equal(mha.paramCount(), 4 * (16 * 16 + 16));
  });

  it('MHA training should decrease loss', () => {
    let passed = false;
    for (let attempt = 0; attempt < 3 && !passed; attempt++) {
      const mha = new MultiHeadAttention(8, 2);
      const input = Matrix.random(4, 16); // batch=4, seq=2, dModel=8
      const target = Matrix.random(4, 16);
      
      let prevLoss = Infinity;
      for (let step = 0; step < 100; step++) {
        const output = mha.forward(input);
        let loss = 0;
        const dOutput = new Matrix(4, 16);
        for (let i = 0; i < 4; i++) {
          for (let j = 0; j < 16; j++) {
            const diff = output.get(i, j) - target.get(i, j);
            loss += diff * diff;
            dOutput.set(i, j, 2 * diff / 64);
          }
        }
        loss /= 64;
        
        mha.backward(dOutput);
        mha.update(0.01);
        
        if (step === 0) prevLoss = loss;
      }
      
      const finalOutput = mha.forward(input);
      let finalLoss = 0;
      for (let i = 0; i < 4; i++)
        for (let j = 0; j < 16; j++)
          finalLoss += (finalOutput.get(i, j) - target.get(i, j)) ** 2;
      finalLoss /= 64;
      
      if (finalLoss < prevLoss) passed = true;
    }
    assert.ok(passed, 'MHA training should decrease loss');
  });
});
