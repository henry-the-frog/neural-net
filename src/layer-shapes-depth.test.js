// layer-shapes-depth.test.js — Verify output shapes for all layer types

import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { Dense } from './layer.js';
import { Matrix } from './matrix.js';

// Helper to check matrix dimensions
function assertShape(m, rows, cols, label) {
  assert.equal(m.rows, rows, `${label}: expected ${rows} rows, got ${m.rows}`);
  assert.equal(m.cols, cols, `${label}: expected ${cols} cols, got ${m.cols}`);
}

describe('Dense Layer Shapes', () => {
  it('forward: single sample', () => {
    const layer = new Dense(4, 3, 'relu');
    const input = Matrix.random(1, 4);
    const output = layer.forward(input);
    assertShape(output, 1, 3, 'Dense(4→3) single');
  });

  it('forward: batch of 8', () => {
    const layer = new Dense(10, 5, 'relu');
    const input = Matrix.random(8, 10);
    const output = layer.forward(input);
    assertShape(output, 8, 5, 'Dense(10→5) batch=8');
  });

  it('backward: returns correct gradient shape', () => {
    const layer = new Dense(6, 4, 'relu');
    layer.training = false; // disable dropout
    const input = Matrix.random(3, 6);
    layer.forward(input);
    const dOutput = Matrix.random(3, 4);
    const dInput = layer.backward(dOutput);
    assertShape(dInput, 3, 6, 'Dense backward gradient');
  });

  it('weight gradients have correct shape', () => {
    const layer = new Dense(5, 3, 'relu');
    layer.training = false;
    const input = Matrix.random(4, 5);
    layer.forward(input);
    layer.backward(Matrix.random(4, 3));
    assertShape(layer.dWeights, 5, 3, 'dWeights');
    assertShape(layer.dBiases, 1, 3, 'dBiases');
  });

  it('large dimensions', () => {
    const layer = new Dense(256, 128, 'relu');
    const input = Matrix.random(32, 256);
    const output = layer.forward(input);
    assertShape(output, 32, 128, 'Dense(256→128) batch=32');
  });

  it('single neuron output', () => {
    const layer = new Dense(10, 1, 'sigmoid');
    const input = Matrix.random(5, 10);
    const output = layer.forward(input);
    assertShape(output, 5, 1, 'Dense(10→1)');
  });
});

describe('Dense with Activations', () => {
  for (const act of ['relu', 'sigmoid', 'tanh', 'softmax']) {
    it(`${act} activation preserves shape`, () => {
      const layer = new Dense(4, 3, act);
      layer.training = false;
      const input = Matrix.random(2, 4);
      const output = layer.forward(input);
      assertShape(output, 2, 3, `Dense with ${act}`);
    });
  }
});

describe('Dense with Dropout', () => {
  it('dropout preserves shape during training', () => {
    const layer = new Dense(8, 4, 'relu', { dropout: 0.5 });
    layer.training = true;
    const input = Matrix.random(3, 8);
    const output = layer.forward(input);
    assertShape(output, 3, 4, 'Dense+dropout training');
  });

  it('dropout preserves shape during inference', () => {
    const layer = new Dense(8, 4, 'relu', { dropout: 0.5 });
    layer.training = false;
    const input = Matrix.random(3, 8);
    const output = layer.forward(input);
    assertShape(output, 3, 4, 'Dense+dropout inference');
  });
});

describe('Multi-Layer Pipeline', () => {
  it('3-layer pipeline shapes', () => {
    const l1 = new Dense(10, 8, 'relu');
    const l2 = new Dense(8, 6, 'relu');
    const l3 = new Dense(6, 3, 'softmax');
    [l1, l2, l3].forEach(l => l.training = false);

    const input = Matrix.random(4, 10);
    const h1 = l1.forward(input);
    assertShape(h1, 4, 8, 'layer 1');
    const h2 = l2.forward(h1);
    assertShape(h2, 4, 6, 'layer 2');
    const out = l3.forward(h2);
    assertShape(out, 4, 3, 'output');
  });

  it('backward propagation through 3 layers', () => {
    const l1 = new Dense(10, 8, 'relu');
    const l2 = new Dense(8, 6, 'relu');
    const l3 = new Dense(6, 3, 'softmax');
    [l1, l2, l3].forEach(l => l.training = false);

    const input = Matrix.random(4, 10);
    const h1 = l1.forward(input);
    const h2 = l2.forward(h1);
    const out = l3.forward(h2);

    const dOut = Matrix.random(4, 3);
    const d3 = l3.backward(dOut);
    assertShape(d3, 4, 6, 'backward layer 3');
    const d2 = l2.backward(d3);
    assertShape(d2, 4, 8, 'backward layer 2');
    const d1 = l1.backward(d2);
    assertShape(d1, 4, 10, 'backward layer 1');
  });
});
