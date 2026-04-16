#!/usr/bin/env node
/**
 * neural-net benchmark — measure performance of key operations
 */

import { Matrix } from '../src/matrix.js';
import { Dense } from '../src/layer.js';
import { Network } from '../src/network.js';
import { TransformerEncoderBlock } from '../src/transformer.js';

function bench(name, fn, iterations = 100) {
  // Warmup
  for (let i = 0; i < 5; i++) fn();
  
  const t0 = performance.now();
  for (let i = 0; i < iterations; i++) fn();
  const elapsed = performance.now() - t0;
  
  console.log(`${name}: ${(elapsed / iterations).toFixed(2)}ms/op (${iterations} ops in ${elapsed.toFixed(0)}ms)`);
}

console.log('=== Neural Net Benchmarks ===\n');

// Matrix operations
const m1 = Matrix.random(100, 100);
const m2 = Matrix.random(100, 100);

bench('Matrix 100x100 multiply', () => m1.dot(m2));
bench('Matrix 100x100 add', () => m1.add(m2));
bench('Matrix 100x100 transpose', () => m1.T());

// Dense layer forward+backward
const dense = new Dense(100, 50, 'relu');
const input = Matrix.random(32, 100); // batch=32

bench('Dense 100→50 forward (batch=32)', () => dense.forward(input));

const output = dense.forward(input);
const dOutput = Matrix.random(32, 50);
bench('Dense 100→50 backward (batch=32)', () => dense.backward(dOutput));

// Network: XOR training step
const net = new Network();
net.dense(2, 16, 'relu');
net.dense(16, 1, 'sigmoid');
net.loss('mse');
const xorIn = Matrix.fromArray([[0,0],[0,1],[1,0],[1,1]]);
const xorOut = Matrix.fromArray([[0],[1],[1],[0]]);

bench('XOR training step', () => {
  net.train({ inputs: xorIn, targets: xorOut }, { epochs: 1, learningRate: 0.5, batchSize: 4 });
});

// Transformer
const encoder = new TransformerEncoderBlock(16, 2);
const tInput = Matrix.random(4, 48); // batch=4, seq=3, dModel=16

bench('Transformer 16d/2h forward (batch=4)', () => encoder.forward(tInput));

const tOutput = encoder.forward(tInput);
const dT = Matrix.random(4, 48);
bench('Transformer 16d/2h backward (batch=4)', () => encoder.backward(dT));

console.log('\nDone.');
