// systematic-gradient-check.test.js — Comprehensive backward pass verification
// This test file systematically checks ALL modules that implement backward()
// by comparing analytical gradients against numerical gradients.
// This is the missing infrastructure that should have caught all 5 backward bugs.
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix } from '../src/matrix.js';

function relErr(a, n) {
  return Math.abs(a - n) / Math.max(Math.abs(a), Math.abs(n), 1e-8);
}

// Generic numerical gradient check for any layer with forward(input) and backward(dOutput)
function checkLayerGradients(layer, input, opts = {}) {
  const { eps = 1e-5, tolerance = 0.01 } = opts;
  
  // Forward pass
  const output = layer.forward(input);
  const dOutput = Matrix.random(output.rows, output.cols);
  
  // Analytical gradient
  const dInput = layer.backward(dOutput);
  
  // Numerical gradient
  let maxErr = 0;
  let worstIdx = '';
  const sampled = Math.min(input.rows * input.cols, 20); // Sample up to 20 elements
  const indices = [];
  for (let i = 0; i < input.rows; i++) {
    for (let j = 0; j < input.cols; j++) {
      indices.push([i, j]);
    }
  }
  // Shuffle and take first `sampled`
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  
  for (let k = 0; k < Math.min(sampled, indices.length); k++) {
    const [i, j] = indices[k];
    const orig = input.get(i, j);
    
    input.set(i, j, orig + eps);
    const outPlus = layer.forward(input);
    let lossPlus = 0;
    for (let r = 0; r < outPlus.rows; r++)
      for (let c = 0; c < outPlus.cols; c++)
        lossPlus += outPlus.get(r, c) * dOutput.get(r, c);
    
    input.set(i, j, orig - eps);
    const outMinus = layer.forward(input);
    let lossMinus = 0;
    for (let r = 0; r < outMinus.rows; r++)
      for (let c = 0; c < outMinus.cols; c++)
        lossMinus += outMinus.get(r, c) * dOutput.get(r, c);
    
    input.set(i, j, orig);
    
    const ng = (lossPlus - lossMinus) / (2 * eps);
    const ag = dInput.get(i, j);
    const err = relErr(ag, ng);
    if (err > maxErr) {
      maxErr = err;
      worstIdx = `[${i},${j}] analytical=${ag.toFixed(6)} numerical=${ng.toFixed(6)}`;
    }
  }
  
  return { maxErr, worstIdx, passed: maxErr < tolerance };
}

describe('Systematic Gradient Check — All Modules', () => {
  
  // Dense layer
  it('Dense (linear)', async () => {
    const { Dense } = await import('../src/layer.js');
    const layer = new Dense(4, 3, 'linear');
    const input = Matrix.random(2, 4);
    const { maxErr, passed, worstIdx } = checkLayerGradients(layer, input);
    assert.ok(passed, `Dense gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });

  it('Dense (relu)', async () => {
    const { Dense } = await import('../src/layer.js');
    const layer = new Dense(4, 3, 'relu');
    // Avoid inputs near 0 where ReLU gradient is discontinuous
    const input = new Matrix(2, 4);
    for (let i = 0; i < 8; i++) input.data[i] = (Math.random() - 0.3) * 2 + 0.5;
    const { maxErr, passed, worstIdx } = checkLayerGradients(layer, input, { tolerance: 0.05 });
    assert.ok(passed, `Dense ReLU gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });

  it('Dense (sigmoid)', async () => {
    const { Dense } = await import('../src/layer.js');
    const layer = new Dense(4, 3, 'sigmoid');
    const input = Matrix.random(2, 4);
    const { maxErr, passed, worstIdx } = checkLayerGradients(layer, input);
    assert.ok(passed, `Dense sigmoid gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });

  // BatchNorm
  it('BatchNorm', async () => {
    const { BatchNorm } = await import('../src/batchnorm.js');
    const layer = new BatchNorm(4);
    const input = Matrix.random(4, 4);
    const { maxErr, passed, worstIdx } = checkLayerGradients(layer, input);
    assert.ok(passed, `BatchNorm gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });

  // LayerNorm
  it('LayerNorm', async () => {
    const { LayerNorm } = await import('../src/transformer.js');
    const layer = new LayerNorm(4);
    const input = Matrix.random(2, 8); // batch=2, seq=2, dModel=4
    const { maxErr, passed, worstIdx } = checkLayerGradients(layer, input);
    assert.ok(passed, `LayerNorm gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });

  // Conv2D (linear activation for clean gradient)
  it('Conv2D', async () => {
    const { Conv2D } = await import('../src/conv.js');
    const layer = new Conv2D(3, 3, 1, 1, 2, 'linear');
    const input = Matrix.random(2, 9); // batch=2, 3x3x1
    const { maxErr, passed, worstIdx } = checkLayerGradients(layer, input);
    assert.ok(passed, `Conv2D gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });

  // SelfAttention
  it('SelfAttention', async () => {
    const { SelfAttention } = await import('../src/attention.js');
    const layer = new SelfAttention(4);
    const input = Matrix.random(1, 8); // batch=1, seq=2, dModel=4
    const { maxErr, passed, worstIdx } = checkLayerGradients(layer, input, { tolerance: 0.05 });
    assert.ok(passed, `SelfAttention gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });

  // MultiHeadAttention
  it('MultiHeadAttention', async () => {
    const { MultiHeadAttention } = await import('../src/attention.js');
    const layer = new MultiHeadAttention(8, 2);
    const input = Matrix.random(1, 16); // batch=1, seq=2, dModel=8
    const { maxErr, passed, worstIdx } = checkLayerGradients(layer, input, { tolerance: 0.05 });
    assert.ok(passed, `MultiHeadAttention gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });

  // TransformerEncoderBlock
  it('TransformerEncoderBlock', async () => {
    const { TransformerEncoderBlock } = await import('../src/transformer.js');
    const layer = new TransformerEncoderBlock(4, 1);
    const input = Matrix.random(1, 8); // batch=1, seq=2, dModel=4
    const { maxErr, passed, worstIdx } = checkLayerGradients(layer, input, { tolerance: 0.1 });
    assert.ok(passed, `TransformerEncoderBlock gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });

  // Dropout (in eval mode)
  it('Dropout (eval mode)', async () => {
    const { Dropout } = await import('../src/dropout.js');
    const layer = new Dropout(0.5);
    layer.training = false; // In eval mode, dropout is identity
    const input = Matrix.random(2, 4);
    const { maxErr, passed, worstIdx } = checkLayerGradients(layer, input);
    assert.ok(passed, `Dropout gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });

  // Embedding
  it('Embedding', async () => {
    const { Embedding } = await import('../src/embedding.js');
    const layer = new Embedding(10, 4);
    // Embedding input is indices, not continuous — skip numerical gradient
    // Instead verify forward/backward shapes
    const input = new Matrix(2, 3, new Float64Array([1, 3, 5, 2, 4, 6]));
    const output = layer.forward(input);
    assert.equal(output.rows, 2);
    assert.equal(output.cols, 12); // 3 * 4
    const dOutput = Matrix.random(2, 12);
    const dInput = layer.backward(dOutput);
    // dWeights should be set
    assert.ok(layer.dWeights, 'Embedding should set dWeights');
  });

  // Residual block
  it('ResidualBlock', async () => {
    const { Residual } = await import('../src/residual.js');
    const { Dense } = await import('../src/layer.js');
    const inner = new Dense(4, 4, 'linear');
    const layer = new Residual(inner);
    const input = Matrix.random(2, 4);
    const { maxErr, passed, worstIdx } = checkLayerGradients(layer, input, { tolerance: 0.05 });
    assert.ok(passed, `Residual gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });

  // Conv1D
  it('Conv1D', async () => {
    const { Conv1D } = await import('../src/conv1d.js');
    const layer = new Conv1D(8, 1, 2, 3, 'linear'); // seqLen=8, inCh=1, outCh=2, kernel=3
    const input = Matrix.random(2, 8); // batch=2
    const { maxErr, passed, worstIdx } = checkLayerGradients(layer, input);
    assert.ok(passed, `Conv1D gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });

  // RNN
  it('RNN', async () => {
    const { RNN } = await import('../src/rnn.js');
    const layer = new RNN(2, 3, 2); // inputSize=2, hiddenSize=3, seqLen=2
    const input = Matrix.random(1, 4); // batch=1, seqLen*inputSize = 2*2 = 4
    const { maxErr, passed, worstIdx } = checkLayerGradients(layer, input, { tolerance: 0.05 });
    assert.ok(passed, `RNN gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });

  // LSTM
  it('LSTM', async () => {
    const { LSTM } = await import('../src/rnn.js');
    const layer = new LSTM(2, 3, 2); // inputSize=2, hiddenSize=3, seqLen=2
    const input = Matrix.random(1, 4);
    const { maxErr, passed, worstIdx } = checkLayerGradients(layer, input, { tolerance: 0.05 });
    assert.ok(passed, `LSTM gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });

  // GRU
  it('GRU', async () => {
    const { GRU } = await import('../src/rnn.js');
    const layer = new GRU(2, 3, 2);
    const input = Matrix.random(1, 4);
    const { maxErr, passed, worstIdx } = checkLayerGradients(layer, input, { tolerance: 0.05 });
    assert.ok(passed, `GRU gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });
});
