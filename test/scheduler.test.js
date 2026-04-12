// scheduler.test.js — Tests for learning rate schedulers

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  stepDecayFn as stepDecay, exponentialDecayFn as exponentialDecay, cosineAnnealingFn as cosineAnnealing,
  warmupFn as warmup, warmupCosineFn as warmupCosine, cyclicLRFn as cyclicLR, reduceLROnPlateau
} from '../src/scheduler.js';

describe('Learning Rate Schedulers', () => {
  describe('stepDecay', () => {
    it('should maintain LR within a step', () => {
      const sched = stepDecay(0.1, 0.5, 10);
      assert.equal(sched(0), 0.1);
      assert.equal(sched(5), 0.1);
      assert.equal(sched(9), 0.1);
    });

    it('should halve LR every 10 epochs', () => {
      const sched = stepDecay(0.1, 0.5, 10);
      assert.ok(Math.abs(sched(10) - 0.05) < 1e-10);
      assert.ok(Math.abs(sched(20) - 0.025) < 1e-10);
      assert.ok(Math.abs(sched(30) - 0.0125) < 1e-10);
    });
  });

  describe('exponentialDecay', () => {
    it('should decay smoothly', () => {
      const sched = exponentialDecay(0.1, 0.9);
      assert.equal(sched(0), 0.1);
      assert.ok(Math.abs(sched(1) - 0.09) < 1e-10);
      assert.ok(sched(10) < sched(0));
    });

    it('should approach zero', () => {
      const sched = exponentialDecay(0.1, 0.5);
      assert.ok(sched(100) < 0.001);
    });
  });

  describe('cosineAnnealing', () => {
    it('should start at initial LR', () => {
      const sched = cosineAnnealing(0.1, 100);
      assert.equal(sched(0), 0.1);
    });

    it('should end at min LR', () => {
      const sched = cosineAnnealing(0.1, 100, 0.001);
      const final = sched(100);
      assert.ok(Math.abs(final - 0.001) < 0.01, `Final LR: ${final}`);
    });

    it('should be at midpoint at half-way', () => {
      const sched = cosineAnnealing(0.1, 100, 0);
      const mid = sched(50);
      assert.ok(Math.abs(mid - 0.05) < 0.01, `Mid LR: ${mid}`);
    });
  });

  describe('warmup', () => {
    it('should linearly increase during warmup', () => {
      const sched = warmup(0.1, 10);
      assert.ok(Math.abs(sched(0) - 0.01) < 1e-10); // 1/10 of target
      assert.ok(Math.abs(sched(4) - 0.05) < 1e-10); // 5/10 of target
      assert.ok(Math.abs(sched(9) - 0.1) < 1e-10);  // 10/10 of target
    });

    it('should maintain target after warmup', () => {
      const sched = warmup(0.1, 5);
      assert.equal(sched(10), 0.1);
      assert.equal(sched(100), 0.1);
    });
  });

  describe('warmupCosine', () => {
    it('should warmup then decay', () => {
      const sched = warmupCosine(0.1, 5, 100);
      // During warmup
      assert.ok(sched(0) < sched(4));
      // Peak at end of warmup
      assert.ok(Math.abs(sched(5) - 0.1) < 0.02);
      // Decays after
      assert.ok(sched(90) < sched(10));
    });
  });

  describe('cyclicLR', () => {
    it('should oscillate between bounds', () => {
      const sched = cyclicLR(0.001, 0.1, 10);
      // Should always be in [baseLR, maxLR]
      for (let i = 0; i < 50; i++) {
        const lr = sched(i);
        assert.ok(lr >= 0.001 - 1e-10 && lr <= 0.1 + 1e-10,
          `LR ${lr} out of bounds at epoch ${i}`);
      }
    });

    it('should complete a full cycle', () => {
      const sched = cyclicLR(0.001, 0.1, 20);
      // At start of cycle: base LR
      assert.ok(sched(0) >= 0.001);
      // At quarter cycle: going up
      assert.ok(sched(5) > sched(0));
    });
  });

  describe('reduceLROnPlateau', () => {
    it('should maintain LR when loss is improving', () => {
      const plateau = reduceLROnPlateau(0.1, 0.5, 3);
      assert.equal(plateau.getLR(), 0.1);
      plateau.step(1.0);
      plateau.step(0.9);
      plateau.step(0.8);
      assert.equal(plateau.getLR(), 0.1);
    });

    it('should reduce LR after patience epochs of no improvement', () => {
      const plateau = reduceLROnPlateau(0.1, 0.5, 3);
      plateau.step(1.0); // best
      plateau.step(1.0); // no improve
      plateau.step(1.0); // no improve
      plateau.step(1.0); // no improve — triggers reduction
      assert.ok(Math.abs(plateau.getLR() - 0.05) < 1e-10, `LR: ${plateau.getLR()}`);
    });

    it('should reduce multiple times', () => {
      const plateau = reduceLROnPlateau(0.1, 0.5, 2);
      plateau.step(1.0);
      plateau.step(1.0);
      plateau.step(1.0); // reduce to 0.05
      assert.ok(Math.abs(plateau.getLR() - 0.05) < 1e-10);
      plateau.step(1.0);
      plateau.step(1.0); // reduce to 0.025
      assert.ok(Math.abs(plateau.getLR() - 0.025) < 1e-10);
    });
  });
});
