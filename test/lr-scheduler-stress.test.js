// lr-scheduler-stress.test.js — Learning rate scheduler tests
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { StepLR, ExponentialLR, CosineAnnealingLR, WarmupLR, CyclicLR, OneCycleLR } from '../src/lr-scheduler.js';

describe('LR Scheduler Stress', () => {
  it('StepLR decreases at steps', () => {
    const sched = new StepLR(0.1, 10, 0.5); // lr=0.1, step=10, gamma=0.5
    const lr0 = sched.getLR(0);
    const lr10 = sched.getLR(10);
    const lr20 = sched.getLR(20);
    assert.equal(lr0, 0.1);
    assert.ok(Math.abs(lr10 - 0.05) < 1e-6, `After step 10: ${lr10}`);
    assert.ok(Math.abs(lr20 - 0.025) < 1e-6, `After step 20: ${lr20}`);
  });

  it('ExponentialLR decays smoothly', () => {
    const sched = new ExponentialLR(0.1, 0.99);
    let prev = sched.getLR(0);
    for (let step = 1; step < 100; step++) {
      const lr = sched.getLR(step);
      assert.ok(lr < prev, `LR should decrease: step ${step}`);
      assert.ok(lr > 0, 'LR should be positive');
      prev = lr;
    }
  });

  it('CosineAnnealing oscillates', () => {
    const sched = new CosineAnnealingLR(0.1, 50, 0.001); // max=0.1, T_max=50, min=0.001
    const lrStart = sched.getLR(0);
    const lrMid = sched.getLR(25);
    const lrEnd = sched.getLR(50);
    
    assert.ok(Math.abs(lrStart - 0.1) < 0.01, `Start should be ~0.1: ${lrStart}`);
    assert.ok(lrMid < lrStart, `Mid should be less: ${lrMid}`);
    assert.ok(Math.abs(lrEnd - 0.001) < 0.01, `End should be ~0.001: ${lrEnd}`);
  });

  it('WarmupLR increases then levels off', () => {
    const sched = new WarmupLR(0.1, 10); // lr=0.1, warmup=10 steps
    const lr0 = sched.getLR(0);
    const lr5 = sched.getLR(5);
    const lr10 = sched.getLR(10);
    const lr20 = sched.getLR(20);
    
    assert.ok(lr0 < lr5, 'LR should increase during warmup');
    assert.ok(lr5 < lr10, 'LR should keep increasing');
    assert.ok(Math.abs(lr10 - 0.1) < 0.02, `Should reach target at warmup end: ${lr10}`);
    assert.ok(Math.abs(lr20 - 0.1) < 0.02, `Should stay at target: ${lr20}`);
  });

  it('CyclicLR oscillates between bounds', () => {
    const sched = new CyclicLR(0.001, 0.1, 20); // min=0.001, max=0.1, cycle=20
    for (let step = 0; step < 100; step++) {
      const lr = sched.getLR(step);
      assert.ok(lr >= 0.001 - 0.001 && lr <= 0.1 + 0.001, 
        `LR should be in [0.001, 0.1]: ${lr} at step ${step}`);
      assert.ok(isFinite(lr), `LR should be finite at step ${step}`);
    }
  });

  it('all schedulers produce finite values', () => {
    const schedulers = [
      new StepLR(0.1, 10, 0.5),
      new ExponentialLR(0.1, 0.99),
      new CosineAnnealingLR(0.1, 50, 0.001),
      new WarmupLR(0.1, 10),
      new CyclicLR(0.001, 0.1, 20),
    ];
    
    for (const sched of schedulers) {
      for (let step = 0; step < 200; step++) {
        const lr = sched.getLR(step);
        assert.ok(isFinite(lr) && lr >= 0, `${sched.constructor.name} at step ${step}: ${lr}`);
      }
    }
  });
});
