import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { tensorParallelSplit, tensorParallelGather, pipelineStages, pipelineBubbleRatio } from './model-parallel.js';

describe('Model Parallelism', () => {
  test('tensor parallel split and gather is identity', () => {
    const weights = [1, 2, 3, 4, 5, 6, 7, 8];
    const shards = tensorParallelSplit(weights, 4);
    assert.equal(shards.length, 4);
    const gathered = tensorParallelGather(shards);
    assert.deepEqual(gathered, weights);
  });

  test('pipeline stages cover all layers', () => {
    const layers = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5'];
    const stages = pipelineStages(layers, 3);
    assert.equal(stages.length, 3);
    assert.deepEqual(stages.flat(), layers);
  });

  test('pipeline bubble ratio decreases with more micro-batches', () => {
    const r1 = pipelineBubbleRatio(4, 4);
    const r2 = pipelineBubbleRatio(32, 4);
    assert.ok(r2 < r1, 'More micro-batches should reduce bubble');
  });
});
