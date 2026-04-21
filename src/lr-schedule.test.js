// lr-schedule.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { cosineWithWarmup, generateSchedule, warmupStableDecay, presets } from './lr-schedule.js';

describe('Learning Rate Schedule', () => {
  it('warmup starts near zero and reaches maxLr', () => {
    const lr0 = cosineWithWarmup(0, 1e-3, 1e-4, 100, 1000);
    const lr99 = cosineWithWarmup(99, 1e-3, 1e-4, 100, 1000);
    assert.ok(lr0 < 2e-5, `Step 0 should be near zero: ${lr0}`);
    assert.ok(Math.abs(lr99 - 1e-3) < 2e-5, `Step 99 should be near maxLr: ${lr99}`);
  });

  it('cosine decays to minLr', () => {
    const lr = cosineWithWarmup(999, 1e-3, 1e-4, 100, 1000);
    assert.ok(Math.abs(lr - 1e-4) < 2e-5, `Last step should be near minLr: ${lr}`);
  });

  it('peak is at warmup boundary', () => {
    const lrBefore = cosineWithWarmup(98, 1e-3, 1e-4, 100, 1000);
    const lrAt = cosineWithWarmup(100, 1e-3, 1e-4, 100, 1000);
    const lrAfter = cosineWithWarmup(200, 1e-3, 1e-4, 100, 1000);
    assert.ok(lrAt >= lrBefore, 'Peak at warmup end');
    assert.ok(lrAt >= lrAfter, 'Decays after warmup');
  });

  it('generates full schedule', () => {
    const schedule = generateSchedule(1e-3, 1e-4, 10, 100);
    assert.equal(schedule.length, 100);
    assert.ok(schedule[0] < schedule[9], 'Should increase during warmup');
    assert.ok(schedule[10] > schedule[99], 'Should decrease after warmup');
  });

  it('WSD schedule: warmup → stable → decay', () => {
    const warmup = warmupStableDecay(5, 1e-3, 1e-4, 10, 40, 100);
    const stable = warmupStableDecay(30, 1e-3, 1e-4, 10, 40, 100);
    const decay = warmupStableDecay(80, 1e-3, 1e-4, 10, 40, 100);
    
    assert.ok(warmup < 1e-3, 'Warmup phase');
    assert.ok(Math.abs(stable - 1e-3) < 1e-6, `Stable phase: ${stable}`);
    assert.ok(decay < 1e-3 && decay > 1e-4, `Decay phase: ${decay}`);
  });

  it('presets have reasonable values', () => {
    for (const [name, p] of Object.entries(presets)) {
      assert.ok(p.maxLr > p.minLr, `${name}: maxLr > minLr`);
      assert.ok(p.warmupSteps < p.totalSteps, `${name}: warmup < total`);
    }
  });
});
