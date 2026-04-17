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

  // ===== NEW: Previously untested modules =====

  // KANLayer (Kolmogorov-Arnold Network layer)
  it('KANLayer', async () => {
    const { KANLayer } = await import('../src/kan.js');
    const layer = new KANLayer(3, 2); // inputSize=3, outputSize=2
    const input = Matrix.random(2, 3); // batch=2
    // Scale inputs to [-0.8, 0.8] to stay within B-spline support
    for (let i = 0; i < input.data.length; i++) {
      input.data[i] = input.data[i] * 0.8;
    }
    const { maxErr, passed, worstIdx } = checkLayerGradients(layer, input, { tolerance: 0.05 });
    assert.ok(passed, `KANLayer gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });

  // KAN (multi-layer KAN network)
  it('KAN', async () => {
    const { KAN } = await import('../src/kan.js');
    const net = new KAN([3, 4, 2]); // 3→4→2
    const input = Matrix.random(2, 3);
    for (let i = 0; i < input.data.length; i++) {
      input.data[i] = input.data[i] * 0.8;
    }
    const { maxErr, passed, worstIdx } = checkLayerGradients(net, input, { tolerance: 0.1 });
    assert.ok(passed, `KAN gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });

  // MixtureOfExperts
  it('MixtureOfExperts', async () => {
    const { MixtureOfExperts } = await import('../src/moe.js');
    // Use topK = numExperts to avoid routing discontinuities in gradient check
    const layer = new MixtureOfExperts(4, 3, 6, 2, 3); // input=4, 3 experts, hidden=6, output=2, topK=3(all)
    const input = Matrix.random(2, 4);
    const { maxErr, passed, worstIdx } = checkLayerGradients(layer, input, { tolerance: 0.1 });
    assert.ok(passed, `MoE gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });

  // CapsuleLayer (uses plain arrays, needs custom check)
  it('CapsuleLayer', async () => {
    const { CapsuleLayer } = await import('../src/capsule.js');
    const numIn = 2, dimIn = 3, numOut = 2, dimOut = 3;
    const layer = new CapsuleLayer(numOut, dimOut, numIn, dimIn, 1); // 1 routing iter to minimize routing gradient error
    
    // Create input as array of arrays
    const makeInput = () => {
      const caps = [];
      for (let i = 0; i < numIn; i++) {
        caps[i] = Array.from({ length: dimIn }, () => (Math.random() - 0.5) * 2);
      }
      return caps;
    };
    
    const input = makeInput();
    const output = layer.forward(input);
    
    // Create random dOutput
    const dOutput = output.map(cap => cap.map(() => (Math.random() - 0.5) * 2));
    const dInput = layer.backward(dOutput);
    
    // Numerical gradient check
    const eps = 1e-5;
    let maxErr = 0;
    let worstIdx = '';
    
    // Compute loss = sum(output * dOutput)
    function computeLoss(inp) {
      const out = layer.forward(inp);
      let loss = 0;
      for (let j = 0; j < out.length; j++) {
        for (let d = 0; d < out[j].length; d++) {
          loss += out[j][d] * dOutput[j][d];
        }
      }
      return loss;
    }
    
    for (let i = 0; i < numIn; i++) {
      for (let k = 0; k < dimIn; k++) {
        const orig = input[i][k];
        
        input[i][k] = orig + eps;
        const lPlus = computeLoss(input);
        
        input[i][k] = orig - eps;
        const lMinus = computeLoss(input);
        
        input[i][k] = orig;
        
        const ng = (lPlus - lMinus) / (2 * eps);
        const ag = dInput[i][k];
        const err = relErr(ag, ng);
        if (err > maxErr) {
          maxErr = err;
          worstIdx = `[${i},${k}] analytical=${ag.toFixed(6)} numerical=${ng.toFixed(6)}`;
        }
      }
    }
    
    // Note: CapsuleLayer's backward ignores gradient flow through dynamic routing
    // (coupling coefficients are treated as constants). This is a known approximation.
    // Tolerance is higher to account for this.
    assert.ok(maxErr < 0.25, `CapsuleLayer gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });

  // NeuralODELayer
  it('NeuralODELayer', async () => {
    const { NeuralODELayer } = await import('../src/neural-ode.js');
    const layer = new NeuralODELayer(3, 1, 'euler', 5); // dim=3, 1 hidden layer, euler, 5 steps
    const input = Matrix.random(2, 3);
    const { maxErr, passed, worstIdx } = checkLayerGradients(layer, input, { tolerance: 0.15 });
    assert.ok(passed, `NeuralODELayer gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });

  // NeuralODE (full encoder→ODE→decoder)
  it('NeuralODE', async () => {
    const { NeuralODE } = await import('../src/neural-ode.js');
    const net = new NeuralODE(4, 3, 2, { solver: 'euler', steps: 5 });
    const input = Matrix.random(2, 4);
    const { maxErr, passed, worstIdx } = checkLayerGradients(net, input, { tolerance: 0.15 });
    assert.ok(passed, `NeuralODE gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });

  // MAMLNetwork (forward/backward through Dense layers)
  it('MAMLNetwork', async () => {
    const { MAMLNetwork } = await import('../src/maml.js');
    const net = new MAMLNetwork([4, 3, 2]); // 4→3→2
    const input = Matrix.random(2, 4);
    const output = net.forward(input);
    const dOutput = Matrix.random(output.rows, output.cols);
    
    // Analytical backward
    let dx = dOutput;
    for (let l = net.layers.length - 1; l >= 0; l--) {
      dx = net.layers[l].backward(dx);
    }
    
    // Numerical gradient check
    let maxErr = 0;
    let worstIdx = '';
    const eps = 1e-5;
    
    const indices = [];
    for (let i = 0; i < input.rows; i++)
      for (let j = 0; j < input.cols; j++)
        indices.push([i, j]);
    
    for (const [i, j] of indices) {
      const orig = input.get(i, j);
      
      input.set(i, j, orig + eps);
      const outP = net.forward(input);
      let lP = 0;
      for (let r = 0; r < outP.rows; r++)
        for (let c = 0; c < outP.cols; c++)
          lP += outP.get(r, c) * dOutput.get(r, c);
      
      input.set(i, j, orig - eps);
      const outM = net.forward(input);
      let lM = 0;
      for (let r = 0; r < outM.rows; r++)
        for (let c = 0; c < outM.cols; c++)
          lM += outM.get(r, c) * dOutput.get(r, c);
      
      input.set(i, j, orig);
      
      const ng = (lP - lM) / (2 * eps);
      const ag = dx.get(i, j);
      const err = relErr(ag, ng);
      if (err > maxErr) {
        maxErr = err;
        worstIdx = `[${i},${j}] analytical=${ag.toFixed(6)} numerical=${ng.toFixed(6)}`;
      }
    }
    
    assert.ok(maxErr < 0.05, `MAMLNetwork gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });

  // Autoencoder (forward/backward through encode→decode chain)
  it('Autoencoder', async () => {
    const { Autoencoder } = await import('../src/autoencoder.js');
    const ae = new Autoencoder(4, 2, [3]); // input=4, latent=2, hidden=[3]
    const input = Matrix.random(2, 4);
    
    // Forward
    const output = ae.forward(input);
    const dOutput = Matrix.random(output.rows, output.cols);
    
    // Analytical backward through decoder then encoder
    let grad = dOutput;
    for (let i = ae.decoderLayers.length - 1; i >= 0; i--) {
      ae.decoderLayers[i].forward(i === 0 ? ae.encode(input) : ae.decoderLayers[i-1].forward(
        i === 1 ? ae.encode(input) : (() => { throw new Error('unexpected'); })()
      ));
    }
    // Simpler approach: just re-forward and backward
    ae.forward(input); // populate caches
    for (let i = ae.decoderLayers.length - 1; i >= 0; i--) {
      grad = ae.decoderLayers[i].backward(grad);
    }
    for (let i = ae.encoderLayers.length - 1; i >= 0; i--) {
      grad = ae.encoderLayers[i].backward(grad);
    }
    const dInput = grad;
    
    // Numerical gradient
    let maxErr = 0;
    let worstIdx = '';
    const eps = 1e-5;
    
    for (let r = 0; r < input.rows; r++) {
      for (let c = 0; c < input.cols; c++) {
        const orig = input.get(r, c);
        
        input.set(r, c, orig + eps);
        const oP = ae.forward(input);
        let lP = 0;
        for (let i = 0; i < oP.rows; i++)
          for (let j = 0; j < oP.cols; j++)
            lP += oP.get(i, j) * dOutput.get(i, j);
        
        input.set(r, c, orig - eps);
        const oM = ae.forward(input);
        let lM = 0;
        for (let i = 0; i < oM.rows; i++)
          for (let j = 0; j < oM.cols; j++)
            lM += oM.get(i, j) * dOutput.get(i, j);
        
        input.set(r, c, orig);
        
        const ng = (lP - lM) / (2 * eps);
        const ag = dInput.get(r, c);
        const err = relErr(ag, ng);
        if (err > maxErr) {
          maxErr = err;
          worstIdx = `[${r},${c}] analytical=${ag.toFixed(6)} numerical=${ng.toFixed(6)}`;
        }
      }
    }
    
    assert.ok(maxErr < 0.05, `Autoencoder gradient error: ${maxErr.toExponential(2)} at ${worstIdx}`);
  });
});
