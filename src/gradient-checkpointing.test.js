// gradient-checkpointing.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  checkpointingAnalysis, optimalCheckpoints,
  checkpointSchedule, simulateCheckpointedPass
} from './gradient-checkpointing.js';

describe('Gradient Checkpointing', () => {
  it('saves significant memory', () => {
    const result = checkpointingAnalysis(32, 6, 100);
    console.log(`  32 layers, 6 checkpoints:`);
    console.log(`    Standard: ${result.standardMemory} units`);
    console.log(`    Checkpoint: ${result.checkpointMemory} units`);
    console.log(`    Savings: ${result.memorySavings}`);
    console.log(`    Compute overhead: ${result.computeOverhead}`);
    assert.ok(result.checkpointMemory < result.standardMemory);
  });

  it('optimal checkpoints is √N', () => {
    assert.equal(optimalCheckpoints(16), 4);
    assert.equal(optimalCheckpoints(64), 8);
    assert.equal(optimalCheckpoints(100), 10);
  });

  it('optimal gives best memory savings', () => {
    const N = 64;
    const optimal = optimalCheckpoints(N);
    const optResult = checkpointingAnalysis(N, optimal, 1);

    // Compare with suboptimal
    const subResult = checkpointingAnalysis(N, 2, 1);
    assert.ok(optResult.checkpointMemory <= subResult.checkpointMemory,
      `Optimal (${optResult.checkpointMemory}) should be ≤ suboptimal (${subResult.checkpointMemory})`);
  });

  it('checkpoint schedule is evenly spaced', () => {
    const schedule = checkpointSchedule(32, 4);
    assert.equal(schedule.length, 4);
    assert.equal(schedule[0], 0);
    // Should be roughly evenly spaced
    for (let i = 1; i < schedule.length; i++) {
      assert.ok(schedule[i] > schedule[i-1], 'Should be increasing');
    }
  });

  it('simulated pass: stored + recomputed covers all layers', () => {
    const schedule = checkpointSchedule(10, 3);
    const sim = simulateCheckpointedPass(10, schedule);

    const allLayers = new Set([...sim.stored, ...sim.recomputed]);
    // Every layer should be either stored or recomputed
    for (let i = 0; i < 10; i++) {
      assert.ok(allLayers.has(i), `Layer ${i} should be covered`);
    }
  });

  it('peak memory is much less than total layers', () => {
    const schedule = checkpointSchedule(64, 8);
    const sim = simulateCheckpointedPass(64, schedule);

    console.log(`  64 layers: peak memory = ${sim.peakMemory} (vs 64 standard)`);
    assert.ok(sim.peakMemory < 64, 'Peak memory should be less than total');
    assert.ok(sim.peakMemory <= 16, 'Peak memory should be ≤ √64 + √64 = 16');
  });

  it('Llama-70B scale analysis', () => {
    const layers = 80;
    const activationMB = 256; // ~256MB per layer activation for batch=1, 8K context
    const optimal = optimalCheckpoints(layers);
    const result = checkpointingAnalysis(layers, optimal, activationMB);

    console.log(`  Llama-70B (${layers} layers, ${activationMB}MB/layer):`);
    console.log(`    Standard: ${(result.standardMemory/1024).toFixed(1)}GB`);
    console.log(`    Checkpointed: ${(result.checkpointMemory/1024).toFixed(1)}GB`);
    console.log(`    Savings: ${result.memorySavings}, Compute: +${result.computeOverhead}`);

    assert.ok(parseFloat(result.memorySavings) > 50, 'Should save > 50% memory');
  });
});
