import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { ConstantLR, LinearWarmup, CosineAnnealing, StepDecay, WarmupCosine, ExponentialDecay, CyclicLR, createScheduler } from '../src/scheduler.js';

describe('Learning Rate Schedulers', () => {
  describe('ConstantLR', () => {
    it('returns same lr always', () => {
      const s = new ConstantLR(0.01);
      assert.equal(s.getLR(0), 0.01);
      assert.equal(s.getLR(100), 0.01);
    });
  });

  describe('LinearWarmup', () => {
    it('starts at 0 and ramps up', () => {
      const s = new LinearWarmup(0.01, 100);
      assert.equal(s.getLR(0), 0);
      assert.ok(Math.abs(s.getLR(50) - 0.005) < 0.001);
      assert.ok(Math.abs(s.getLR(100) - 0.01) < 0.001);
    });
    it('stays at base after warmup', () => {
      const s = new LinearWarmup(0.01, 100);
      assert.equal(s.getLR(200), 0.01);
    });
  });

  describe('CosineAnnealing', () => {
    it('starts at max and decays to min', () => {
      const s = new CosineAnnealing(0.01, 0.0001, 100);
      assert.ok(Math.abs(s.getLR(0, 0) - 0.01) < 0.001);
      assert.ok(Math.abs(s.getLR(0, 100) - 0.0001) < 0.001);
    });
    it('follows cosine curve', () => {
      const s = new CosineAnnealing(0.01, 0, 100);
      const midLR = s.getLR(0, 50);
      assert.ok(Math.abs(midLR - 0.005) < 0.001); // cos(pi/2) = 0 → 0.5 * max
    });
  });

  describe('StepDecay', () => {
    it('decays at step intervals', () => {
      const s = new StepDecay(0.01, 0.1, 30);
      assert.ok(Math.abs(s.getLR(0, 0) - 0.01) < 0.0001);
      assert.ok(Math.abs(s.getLR(0, 30) - 0.001) < 0.0001);
      assert.ok(Math.abs(s.getLR(0, 60) - 0.0001) < 0.00001);
    });
  });

  describe('WarmupCosine', () => {
    it('warms up then decays', () => {
      const s = new WarmupCosine(0.01, 0.0001, 100, 50);
      // During warmup
      assert.equal(s.getLR(0, 0), 0);
      assert.ok(s.getLR(50, 0) > 0); // halfway through warmup
      // After warmup — should follow cosine
      assert.ok(s.getLR(100, 0) >= 0.009); // near max
      assert.ok(s.getLR(100, 50) < 0.001); // near min
    });
  });

  describe('ExponentialDecay', () => {
    it('decays exponentially', () => {
      const s = new ExponentialDecay(0.01, 0.96, 1000);
      const lr0 = s.getLR(0);
      const lr1000 = s.getLR(1000);
      assert.ok(lr0 > lr1000);
      assert.ok(Math.abs(lr1000 - 0.01 * 0.96) < 0.001);
    });
  });

  describe('CyclicLR', () => {
    it('cycles between base and max', () => {
      const s = new CyclicLR(0.001, 0.01, 100);
      const lrs = [];
      for (let i = 0; i < 400; i += 20) lrs.push(s.getLR(i));
      // Should have at least one peak and one valley
      const max = Math.max(...lrs);
      const min = Math.min(...lrs);
      assert.ok(max >= 0.005, `Max should be >= 0.005, got ${max}`);
      assert.ok(min <= 0.005, `Min should be <= 0.005, got ${min}`);
    });
  });

  describe('createScheduler', () => {
    it('creates all types', () => {
      assert.ok(createScheduler('constant', { lr: 0.01 }));
      assert.ok(createScheduler('warmup', { lr: 0.01, warmupSteps: 100 }));
      assert.ok(createScheduler('cosine', { lr: 0.01, totalEpochs: 100 }));
      assert.ok(createScheduler('step', { lr: 0.01, factor: 0.1, stepSize: 30 }));
      assert.ok(createScheduler('warmup_cosine', { lr: 0.01 }));
      assert.ok(createScheduler('exponential', { lr: 0.01 }));
      assert.ok(createScheduler('cyclic', { lr: 0.01 }));
    });
    it('throws on unknown', () => {
      assert.throws(() => createScheduler('unknown'));
    });
  });
});
