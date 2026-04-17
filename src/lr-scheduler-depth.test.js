// lr-scheduler-depth.test.js — Learning rate scheduler depth tests

import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { StepLR, ExponentialLR, CosineAnnealingLR, WarmupLR, CyclicLR, OneCycleLR } from './lr-scheduler.js';

describe('StepLR', () => {
  it('stays constant within step', () => {
    const s = new StepLR(0.1, 10, 0.5);
    assert.equal(s.getLR(0), 0.1);
    assert.equal(s.getLR(5), 0.1);
    assert.equal(s.getLR(9), 0.1);
  });

  it('decays at step boundary', () => {
    const s = new StepLR(0.1, 10, 0.5);
    assert.ok(Math.abs(s.getLR(10) - 0.05) < 1e-10);
    assert.ok(Math.abs(s.getLR(20) - 0.025) < 1e-10);
  });
});

describe('ExponentialLR', () => {
  it('decays exponentially', () => {
    const s = new ExponentialLR(0.1, 0.9);
    assert.ok(Math.abs(s.getLR(0) - 0.1) < 1e-10);
    assert.ok(Math.abs(s.getLR(1) - 0.09) < 1e-10);
    assert.ok(Math.abs(s.getLR(2) - 0.081) < 1e-10);
  });

  it('always positive', () => {
    const s = new ExponentialLR(0.1, 0.99);
    for (let i = 0; i < 100; i++) {
      assert.ok(s.getLR(i) > 0);
    }
  });
});

describe('CosineAnnealingLR', () => {
  it('starts at base LR', () => {
    const s = new CosineAnnealingLR(0.1, 100, 0);
    assert.ok(Math.abs(s.getLR(0) - 0.1) < 1e-10);
  });

  it('ends at min LR', () => {
    const s = new CosineAnnealingLR(0.1, 100, 0);
    assert.ok(Math.abs(s.getLR(100) - 0) < 1e-10);
  });

  it('midpoint is halfway', () => {
    const s = new CosineAnnealingLR(0.1, 100, 0);
    assert.ok(Math.abs(s.getLR(50) - 0.05) < 0.001);
  });
});

describe('WarmupLR', () => {
  it('starts low and increases', () => {
    const s = new WarmupLR(0.1, 10);
    const lr0 = s.getLR(0);
    const lr5 = s.getLR(5);
    assert.ok(lr0 < lr5);
    assert.ok(lr5 < 0.1);
  });

  it('reaches base LR after warmup', () => {
    const s = new WarmupLR(0.1, 10);
    assert.equal(s.getLR(10), 0.1);
    assert.equal(s.getLR(20), 0.1);
  });
});

describe('CyclicLR', () => {
  it('oscillates between base and max', () => {
    const s = new CyclicLR(0.001, 0.1, 10);
    const lrs = [];
    for (let i = 0; i < 40; i++) lrs.push(s.getLR(i));
    
    const min = Math.min(...lrs);
    const max = Math.max(...lrs);
    assert.ok(min >= 0.001 - 0.001);
    assert.ok(max <= 0.1 + 0.001);
  });
});

describe('OneCycleLR', () => {
  it('warms up then decays', () => {
    const s = new OneCycleLR(0.1, 100);
    const lr0 = s.getLR(0);
    const lr30 = s.getLR(30); // Peak of warmup
    const lr100 = s.getLR(100);
    
    assert.ok(lr0 < lr30, 'Should warm up');
    assert.ok(lr100 < lr30, 'Should decay after peak');
  });
});
