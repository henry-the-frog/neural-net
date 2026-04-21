// continuous-batching.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Request, ContinuousBatchingScheduler } from './continuous-batching.js';

describe('Continuous Batching', () => {
  // Simple token generator: returns token at (promptLen % vocabSize)
  const vocabSize = 8;
  const generateToken = (reqId, tokens) => tokens.length % vocabSize;

  it('processes single request', () => {
    const sched = new ContinuousBatchingScheduler({ maxBatchSize: 4 });
    sched.addRequest(new Request('r1', [0, 1, 2], 3));
    
    const result = sched.runAll(generateToken);
    assert.equal(result.totalCompleted, 1);
    assert.equal(sched.completed[0].generatedTokens.length, 3);
  });

  it('processes multiple requests concurrently', () => {
    const sched = new ContinuousBatchingScheduler({ maxBatchSize: 4 });
    for (let i = 0; i < 3; i++) {
      sched.addRequest(new Request(`r${i}`, [0, 1], 2));
    }

    const result = sched.runAll(generateToken);
    assert.equal(result.totalCompleted, 3);
  });

  it('respects maxBatchSize', () => {
    const sched = new ContinuousBatchingScheduler({ maxBatchSize: 2 });
    for (let i = 0; i < 5; i++) {
      sched.addRequest(new Request(`r${i}`, [0], 1));
    }

    // First iteration: admit 2, prefill 2
    const iter1 = sched.iterate(generateToken);
    assert.equal(iter1.prefilled, 2);
    assert.equal(sched.stats().running, 2);
    assert.equal(sched.stats().queued, 3);
  });

  it('evicts finished requests to make room', () => {
    const sched = new ContinuousBatchingScheduler({ maxBatchSize: 2 });
    // r1 and r2: 1 token each (finish after 1 decode)
    sched.addRequest(new Request('r1', [0], 1));
    sched.addRequest(new Request('r2', [0], 1));
    sched.addRequest(new Request('r3', [0], 1));

    // iter 1: admit r1, r2, prefill both (they generate 1 token → done)
    sched.iterate(generateToken);
    // iter 2: evict r1, r2, admit r3
    const iter2 = sched.iterate(generateToken);
    assert.equal(iter2.completed, 2, 'r1 and r2 should be evicted');
    assert.equal(iter2.prefilled, 1, 'r3 should be admitted');
  });

  it('EOS stops generation early', () => {
    const eosGen = (_, tokens) => tokens.length === 3 ? 0 : 1; // EOS=0 at token 3
    const sched = new ContinuousBatchingScheduler();
    sched.addRequest(new Request('r1', [1, 2], 10, 0));

    const result = sched.runAll(eosGen);
    assert.ok(sched.completed[0].generatedTokens.length < 10, 'Should stop at EOS');
  });

  it('staggered arrivals work correctly', () => {
    const sched = new ContinuousBatchingScheduler({ maxBatchSize: 4 });
    sched.addRequest(new Request('r1', [0], 3));
    
    sched.iterate(generateToken); // r1 prefills
    sched.iterate(generateToken); // r1 decodes
    
    // r2 arrives mid-stream
    sched.addRequest(new Request('r2', [0], 2));
    sched.iterate(generateToken); // r1 decodes, r2 prefills
    
    const result = sched.runAll(generateToken);
    assert.equal(result.totalCompleted, 2);
  });
});
