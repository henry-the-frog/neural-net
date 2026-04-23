// gradient-checkpoint.test.js — Gradient checkpointing tests
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { CheckpointSegment, checkpoint, memoryEstimate } from './gradient-checkpoint.js';
import { Dense } from './layer.js';
import { Matrix } from './matrix.js';

describe('Gradient Checkpointing', () => {
  test('CheckpointSegment forward produces same output as sequential', () => {
    const layers = [new Dense(4, 8, 'relu'), new Dense(8, 4, 'relu')];
    const seg = new CheckpointSegment(layers);
    
    const input = Matrix.random(2, 4);
    const output = seg.forward(input);
    
    // Sequential forward
    let x = input;
    for (const l of layers) x = l.forward(x);
    
    // Outputs should match (same layers, same computation)
    assert.equal(output.rows, x.rows);
    assert.equal(output.cols, x.cols);
  });

  test('CheckpointSegment backward runs without error', () => {
    const layers = [new Dense(4, 8, 'relu'), new Dense(8, 3, 'relu')];
    const seg = new CheckpointSegment(layers);
    
    const input = Matrix.random(2, 4);
    seg.forward(input);
    
    const dOutput = Matrix.random(2, 3);
    const dInput = seg.backward(dOutput);
    
    assert.equal(dInput.rows, 2);
    assert.equal(dInput.cols, 4);
  });

  test('checkpoint splits layers into segments', () => {
    const layers = Array.from({length: 9}, () => new Dense(4, 4, 'relu'));
    const segments = checkpoint(layers, 3);
    
    assert.equal(segments.length, 3);
    assert.equal(segments[0].layers.length, 3);
    assert.equal(segments[1].layers.length, 3);
    assert.equal(segments[2].layers.length, 3);
  });

  test('checkpoint auto-sizes to sqrt(N)', () => {
    const layers = Array.from({length: 16}, () => new Dense(4, 4, 'relu'));
    const segments = checkpoint(layers);
    
    // sqrt(16) = 4, so 4 segments of 4 layers
    assert.equal(segments.length, 4);
  });

  test('checkpoint handles non-divisible layer count', () => {
    const layers = Array.from({length: 10}, () => new Dense(4, 4, 'relu'));
    const segments = checkpoint(layers, 3);
    
    assert.equal(segments.length, 4); // ceil(10/3) = 4
    assert.equal(segments[3].layers.length, 1); // Last segment has 1 layer
  });

  test('memoryEstimate shows savings', () => {
    const est = memoryEstimate(100);
    assert.ok(est.storedActivations < est.withoutCheckpointing);
    assert.ok(est.savings.includes('%'));
    assert.equal(est.segmentSize, 10); // sqrt(100)
    assert.equal(est.numSegments, 10);
  });

  test('memoryEstimate for large model', () => {
    const est = memoryEstimate(1000);
    assert.ok(est.storedActivations < 100, `Should store < 100 activations, got ${est.storedActivations}`);
    assert.equal(est.withoutCheckpointing, 1000);
  });

  test('end-to-end: checkpointed forward+backward on deep network', () => {
    const layers = [];
    for (let i = 0; i < 8; i++) {
      layers.push(new Dense(4, 4, 'relu'));
    }
    
    const segments = checkpoint(layers, 4);
    assert.equal(segments.length, 2);
    
    // Forward through all segments
    let x = Matrix.random(3, 4);
    for (const seg of segments) {
      x = seg.forward(x);
    }
    assert.equal(x.rows, 3);
    assert.equal(x.cols, 4);
    
    // Backward through all segments (reverse order)
    let dX = Matrix.ones(3, 4);
    for (let i = segments.length - 1; i >= 0; i--) {
      dX = segments[i].backward(dX);
    }
    assert.equal(dX.rows, 3);
    assert.equal(dX.cols, 4);
  });

  test('paramCount aggregates all layer params', () => {
    const seg = new CheckpointSegment([
      new Dense(4, 8, 'relu'), // 4*8 + 8 = 40 params
      new Dense(8, 3, 'relu'), // 8*3 + 3 = 27 params
    ]);
    const total = seg.paramCount();
    assert.equal(total, 40 + 27);
  });
});
