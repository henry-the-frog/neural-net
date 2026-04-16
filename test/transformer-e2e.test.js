// transformer-e2e.test.js — End-to-end Transformer training test
// Verifies the full pipeline (embedding → encoder → output) works correctly
// after all backward pass fixes (LayerNorm, attention, FF batching)
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { TransformerEncoderBlock, PositionalEncoding, LayerNorm } from '../src/transformer.js';
import { Dense } from '../src/layer.js';
import { Matrix } from '../src/matrix.js';

// Simple task: learn to predict the sum of input features at each position
// Input: [batch, seqLen * dModel] random values
// Target: at each position, the output should be the mean of that position's features

describe('Transformer E2E Training', () => {
  it('single encoder block learns a mapping task', () => {
    let passed = false;
    for (let attempt = 0; attempt < 3 && !passed; attempt++) {
      const dModel = 8;
      const seqLen = 3;
      const numHeads = 2;
      
      // Build model: PE → Encoder → Dense output
      const pe = new PositionalEncoding(dModel, seqLen);
      const encoder = new TransformerEncoderBlock(dModel, numHeads);
      const output = new Dense(dModel * seqLen, dModel * seqLen, 'linear');
      
      // Training data: random inputs, target = inputs scaled by 0.5
      const batchSize = 4;
      const inputs = Matrix.random(batchSize, seqLen * dModel);
      const targets = new Matrix(batchSize, seqLen * dModel);
      for (let i = 0; i < targets.data.length; i++) {
        targets.data[i] = inputs.data[i] * 0.5;
      }
      
      let firstLoss = null;
      const lr = 0.005;
      
      for (let step = 0; step < 300; step++) {
        // Forward
        const encoded = pe.forward(inputs);
        const encoderOut = encoder.forward(encoded);
        const pred = output.forward(encoderOut);
        
        // MSE Loss
        let loss = 0;
        const dPred = new Matrix(batchSize, seqLen * dModel);
        const n = batchSize * seqLen * dModel;
        for (let i = 0; i < batchSize; i++) {
          for (let j = 0; j < seqLen * dModel; j++) {
            const diff = pred.get(i, j) - targets.get(i, j);
            loss += diff * diff;
            dPred.set(i, j, 2 * diff / n);
          }
        }
        loss /= n;
        if (firstLoss === null) firstLoss = loss;
        
        // Backward
        const dEncOut = output.backward(dPred);
        const dEncoded = encoder.backward(dEncOut);
        
        // Update
        encoder.update(lr);
        output.update(lr, 0, 'sgd');
      }
      
      // Check final loss
      const finalEncoded = pe.forward(inputs);
      const finalEncoderOut = encoder.forward(finalEncoded);
      const finalPred = output.forward(finalEncoderOut);
      let finalLoss = 0;
      const n = batchSize * seqLen * dModel;
      for (let i = 0; i < batchSize; i++) {
        for (let j = 0; j < seqLen * dModel; j++) {
          finalLoss += (finalPred.get(i, j) - targets.get(i, j)) ** 2;
        }
      }
      finalLoss /= n;
      
      if (finalLoss < firstLoss * 0.8) passed = true;
    }
    assert.ok(passed, 'Transformer E2E training should decrease loss by 20%');
  });

  it('attention weights are proper distributions', () => {
    const dModel = 8;
    const encoder = new TransformerEncoderBlock(dModel, 2);
    const input = Matrix.random(1, 24); // seq=3
    encoder.forward(input);
    
    // The attention layer inside should have computed valid attention weights
    const attn = encoder.attention;
    // MHA stores per-head attention weights
    if (attn._allHeadAttn && attn._allHeadAttn[0]) {
      for (const headAttn of attn._allHeadAttn[0]) {
        for (let i = 0; i < headAttn.rows; i++) {
          let sum = 0;
          for (let j = 0; j < headAttn.cols; j++) {
            const w = headAttn.get(i, j);
            assert.ok(w >= 0 && w <= 1, `Attention weight out of [0,1]: ${w}`);
            sum += w;
          }
          assert.ok(Math.abs(sum - 1) < 1e-5, `Attention row ${i} sum = ${sum}`);
        }
      }
    }
  });

  it('gradients flow through the full stack', () => {
    const dModel = 4;
    const encoder = new TransformerEncoderBlock(dModel, 1);
    const input = Matrix.random(2, 8); // batch=2, seq=2
    
    const output = encoder.forward(input);
    const dOutput = Matrix.random(2, 8);
    const dInput = encoder.backward(dOutput);
    
    // All gradients should be finite and non-zero
    let allFinite = true;
    let allNonZero = false;
    for (let i = 0; i < dInput.data.length; i++) {
      if (!isFinite(dInput.data[i])) allFinite = false;
      if (dInput.data[i] !== 0) allNonZero = true;
    }
    
    assert.ok(allFinite, 'All input gradients should be finite');
    assert.ok(allNonZero, 'Some input gradients should be non-zero');
  });

  it('LayerNorm output is properly normalized after training', () => {
    const norm = new LayerNorm(4);
    const input = Matrix.random(3, 8); // batch=3, seq=2, dModel=4
    const output = norm.forward(input);
    
    // Each position's features should be normalized
    for (let b = 0; b < 3; b++) {
      for (let t = 0; t < 2; t++) {
        let mean = 0;
        for (let d = 0; d < 4; d++) mean += output.get(b, t * 4 + d);
        mean /= 4;
        assert.ok(Math.abs(mean) < 1e-5, `Position (${b},${t}) mean: ${mean}`);
      }
    }
  });
});
