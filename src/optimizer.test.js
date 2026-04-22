// optimizer.test.js — Tests for ScheduledOptimizer
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { ScheduledOptimizer, SGD, Adam } from './optimizer.js';
import { WarmupScheduler, CosineDecay, ConstantLR } from './lr-scheduler.js';
import { Matrix } from './matrix.js';

describe('ScheduledOptimizer', () => {
  it('passes scheduler LR to optimizer', () => {
    const sgd = new SGD(0.05);
    const sched = new ConstantLR(0.05);
    const opt = new ScheduledOptimizer(sgd, sched);
    
    const params = new Matrix(1, 1, new Float64Array([1.0]));
    const grads = new Matrix(1, 1, new Float64Array([1.0]));
    const newParams = opt.update(params, grads);
    assert.ok(Math.abs(newParams.data[0] - 0.95) < 1e-10);
  });
  
  it('schedulerStep advances LR', () => {
    const sgd = new SGD(0.1);
    const sched = new CosineDecay(0.1, 100);
    const opt = new ScheduledOptimizer(sgd, sched);
    
    const lr0 = opt.getLR();
    opt.schedulerStep();
    opt.schedulerStep();
    const lr2 = opt.getLR();
    assert.ok(lr2 < lr0, 'LR should decrease');
  });
  
  it('works with Adam', () => {
    const adam = new Adam(0.001);
    const sched = new WarmupScheduler(new CosineDecay(0.001, 100), 10);
    const opt = new ScheduledOptimizer(adam, sched);
    
    let params = new Matrix(1, 2, new Float64Array([1.0, 2.0]));
    const grads = new Matrix(1, 2, new Float64Array([0.1, 0.2]));
    
    for (let i = 0; i < 20; i++) {
      opt.schedulerStep();
      params = opt.update(params, grads, 'test');
    }
    
    assert.notEqual(params.data[0], 1.0);
    assert.ok(opt.getLR() > 0);
  });
  
  it('reset clears state', () => {
    const sgd = new SGD(0.1);
    const sched = new CosineDecay(0.1, 100);
    const opt = new ScheduledOptimizer(sgd, sched);
    
    for (let i = 0; i < 10; i++) opt.schedulerStep();
    
    opt.reset();
    assert.equal(opt.getStep(), 0);
    assert.ok(Math.abs(opt.getLR() - 0.1) < 1e-10);
  });
  
  it('getStep tracks scheduler steps', () => {
    const opt = new ScheduledOptimizer(new SGD(0.1), new ConstantLR(0.1));
    assert.equal(opt.getStep(), 0);
    opt.schedulerStep();
    opt.schedulerStep();
    assert.equal(opt.getStep(), 2);
  });
});
