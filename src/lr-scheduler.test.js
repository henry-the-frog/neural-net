// lr-scheduler.test.js — Tests for Learning Rate Schedulers
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  ConstantLR,
  CosineDecay,
  StepDecay,
  LinearWarmup,
  WarmupScheduler,
  OneCycle,
  ExponentialDecay,
} from './lr-scheduler.js';

describe('ConstantLR', () => {
  it('always returns the same LR', () => {
    const s = new ConstantLR(0.01);
    for (let i = 0; i < 100; i++) {
      assert.equal(s.step(), 0.01);
    }
  });
  
  it('tracks step count', () => {
    const s = new ConstantLR(0.01);
    s.step(); s.step(); s.step();
    assert.equal(s.getStep(), 3);
  });
  
  it('reset works', () => {
    const s = new ConstantLR(0.01);
    s.step(); s.step();
    s.reset();
    assert.equal(s.getStep(), 0);
  });
});

describe('CosineDecay', () => {
  it('starts at lrMax', () => {
    const s = new CosineDecay(0.01, 100);
    assert.ok(Math.abs(s.step() - 0.01) < 1e-10);
  });
  
  it('ends at lrMin', () => {
    const s = new CosineDecay(0.01, 100, 0.001);
    for (let i = 0; i < 100; i++) s.step();
    assert.ok(Math.abs(s.getLR() - 0.001) < 1e-10);
  });
  
  it('midpoint is (lrMax + lrMin) / 2', () => {
    const s = new CosineDecay(0.01, 100, 0);
    for (let i = 0; i < 50; i++) s.step();
    assert.ok(Math.abs(s.getLR() - 0.005) < 1e-10);
  });
  
  it('monotonically decreasing', () => {
    const s = new CosineDecay(0.01, 100);
    let prev = s.step();
    for (let i = 1; i < 100; i++) {
      const curr = s.step();
      assert.ok(curr <= prev + 1e-12, `Step ${i}: ${curr} > ${prev}`);
      prev = curr;
    }
  });
  
  it('clamps at totalSteps', () => {
    const s = new CosineDecay(0.01, 10, 0.001);
    for (let i = 0; i < 20; i++) s.step();
    assert.ok(Math.abs(s.getLR() - 0.001) < 1e-10);
  });
});

describe('StepDecay', () => {
  it('constant before first milestone', () => {
    const s = new StepDecay(0.1, [30, 60], 0.1);
    for (let i = 0; i < 30; i++) {
      assert.ok(Math.abs(s.step() - 0.1) < 1e-10);
    }
  });
  
  it('decays at milestone', () => {
    const s = new StepDecay(0.1, [10], 0.1);
    for (let i = 0; i < 10; i++) s.step(); // Steps 0-9
    assert.ok(Math.abs(s.step() - 0.01) < 1e-10); // Step 10
  });
  
  it('multiple milestones', () => {
    const s = new StepDecay(1.0, [5, 10], 0.5);
    for (let i = 0; i < 5; i++) s.step();
    assert.ok(Math.abs(s.step() - 0.5) < 1e-10);
    for (let i = 0; i < 4; i++) s.step();
    assert.ok(Math.abs(s.step() - 0.25) < 1e-10);
  });
});

describe('LinearWarmup', () => {
  it('starts at 0', () => {
    const s = new LinearWarmup(0.01, 100);
    assert.ok(Math.abs(s.step()) < 1e-10);
  });
  
  it('reaches lrMax at warmupSteps', () => {
    const s = new LinearWarmup(0.01, 10);
    for (let i = 0; i < 10; i++) s.step();
    assert.ok(Math.abs(s.getLR() - 0.01) < 1e-10);
  });
  
  it('linear interpolation', () => {
    const s = new LinearWarmup(0.1, 10);
    for (let i = 0; i < 5; i++) s.step();
    assert.ok(Math.abs(s.getLR() - 0.05) < 1e-10);
  });
  
  it('stays at lrMax after warmup', () => {
    const s = new LinearWarmup(0.01, 5);
    for (let i = 0; i < 20; i++) s.step();
    assert.ok(Math.abs(s.getLR() - 0.01) < 1e-10);
  });
});

describe('WarmupScheduler', () => {
  it('warmup + cosine decay', () => {
    const base = new CosineDecay(0.01, 100);
    const s = new WarmupScheduler(base, 10);
    
    // First step: warmup scale = 0
    assert.ok(s.step() < 1e-10);
    
    // At step 10: warmup complete, should be close to base LR
    for (let i = 1; i < 10; i++) s.step();
    const lr10 = s.step();
    assert.ok(lr10 > 0.005, `LR at step 10 should be meaningful: ${lr10}`);
  });
  
  it('delegates to base after warmup', () => {
    const base = new ConstantLR(0.01);
    const s = new WarmupScheduler(base, 5);
    for (let i = 0; i < 5; i++) s.step();
    assert.ok(Math.abs(s.step() - 0.01) < 1e-10);
  });
});

describe('OneCycle', () => {
  it('starts at lrInit (lrMax / divFactor)', () => {
    const s = new OneCycle(0.01, 100, { divFactor: 25 });
    assert.ok(Math.abs(s.step() - 0.01 / 25) < 1e-10);
  });
  
  it('peaks at lrMax during warmup', () => {
    const s = new OneCycle(0.01, 100, { pctStart: 0.3 });
    let maxLR = 0;
    for (let i = 0; i < 100; i++) {
      const lr = s.step();
      if (lr > maxLR) maxLR = lr;
    }
    assert.ok(Math.abs(maxLR - 0.01) < 0.001, `Peak should be ~0.01, got ${maxLR}`);
  });
  
  it('ends near lrFinal (lrMax / finalDivFactor)', () => {
    const s = new OneCycle(0.01, 100);
    for (let i = 0; i < 100; i++) s.step();
    const finalLR = s.getLR();
    const expected = 0.01 / 10000;
    assert.ok(Math.abs(finalLR - expected) < expected * 0.1, `Final LR: ${finalLR} vs expected ${expected}`);
  });
  
  it('has clear warmup and decay phases', () => {
    const s = new OneCycle(0.01, 100, { pctStart: 0.3 });
    const lrs = [];
    for (let i = 0; i < 100; i++) lrs.push(s.step());
    
    // Warmup phase: increasing
    for (let i = 1; i < 30; i++) {
      assert.ok(lrs[i] >= lrs[i - 1] - 1e-12, `Should increase during warmup: step ${i}`);
    }
    
    // Decay phase: decreasing
    for (let i = 31; i < 100; i++) {
      assert.ok(lrs[i] <= lrs[i - 1] + 1e-12, `Should decrease during decay: step ${i}`);
    }
  });
});

describe('ExponentialDecay', () => {
  it('starts at lrInit', () => {
    const s = new ExponentialDecay(0.1, 0.9);
    assert.ok(Math.abs(s.step() - 0.1) < 1e-10);
  });
  
  it('decays exponentially', () => {
    const s = new ExponentialDecay(1.0, 0.5);
    assert.ok(Math.abs(s.step() - 1.0) < 1e-10);
    assert.ok(Math.abs(s.step() - 0.5) < 1e-10);
    assert.ok(Math.abs(s.step() - 0.25) < 1e-10);
  });
  
  it('decaySteps controls rate', () => {
    const s = new ExponentialDecay(1.0, 0.5, 10);
    // After 10 steps: 1.0 * 0.5^1 = 0.5
    for (let i = 0; i < 10; i++) s.step();
    assert.ok(Math.abs(s.getLR() - 0.5) < 1e-10);
  });
});
