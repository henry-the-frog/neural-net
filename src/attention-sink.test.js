// attention-sink.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { AttentionSinkCache } from './attention-sink.js';

describe('Attention Sink Cache', () => {
  it('fills sink slots first', () => {
    const cache = new AttentionSinkCache(2, 3, 5);
    cache.append([1, 2], [3, 4]);
    cache.append([5, 6], [7, 8]);
    assert.equal(cache.stats().sinkTokens, 2);
    assert.equal(cache.stats().recentTokens, 0);
  });

  it('overflows to recent window after sinks filled', () => {
    const cache = new AttentionSinkCache(2, 2, 3);
    for (let i = 0; i < 5; i++) cache.append([i, i], [i, i]);
    assert.equal(cache.stats().sinkTokens, 2);
    assert.equal(cache.stats().recentTokens, 3);
    assert.equal(cache.size, 5);
  });

  it('evicts recent but preserves sinks', () => {
    const cache = new AttentionSinkCache(2, 2, 3);
    for (let i = 0; i < 20; i++) cache.append([i, i], [i, i]);

    const stats = cache.stats();
    assert.equal(stats.sinkTokens, 2, 'Sinks preserved');
    assert.equal(stats.recentTokens, 3, 'Recent bounded by window');
    assert.equal(stats.totalSeen, 20);
    assert.equal(stats.evicted, 15);

    // Verify sinks are the FIRST tokens (0, 1)
    const K = cache.getKeys();
    assert.ok(Math.abs(K.get(0, 0) - 0) < 0.01, 'First sink should be token 0');
    assert.ok(Math.abs(K.get(1, 0) - 1) < 0.01, 'Second sink should be token 1');

    // Verify recent are the LAST tokens (17, 18, 19)
    assert.ok(Math.abs(K.get(2, 0) - 17) < 0.01, 'First recent should be token 17');
    assert.ok(Math.abs(K.get(4, 0) - 19) < 0.01, 'Last recent should be token 19');
  });

  it('bounded memory', () => {
    const cache = new AttentionSinkCache(8, 4, 16);
    for (let i = 0; i < 1000; i++) {
      cache.append(new Array(8).fill(i), new Array(8).fill(i));
    }
    assert.equal(cache.size, 20, 'Should be bounded: 4 sinks + 16 recent');
    console.log(`  1000 tokens, cache: ${cache.stats().totalCached}/${cache.capacity}`);
  });

  it('getKeys/getValues return correct dimensions', () => {
    const cache = new AttentionSinkCache(4, 2, 3);
    for (let i = 0; i < 8; i++) cache.append([i, i, i, i], [i, i, i, i]);
    
    const K = cache.getKeys();
    const V = cache.getValues();
    assert.equal(K.rows, 5); // 2 sinks + 3 recent
    assert.equal(K.cols, 4);
    assert.equal(V.rows, 5);
  });

  it('clear resets everything', () => {
    const cache = new AttentionSinkCache(2, 2, 3);
    for (let i = 0; i < 10; i++) cache.append([i, i], [i, i]);
    cache.clear();
    assert.equal(cache.size, 0);
    assert.equal(cache.stats().totalSeen, 0);
  });
});
