// prefix-caching.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { PrefixCache } from './prefix-caching.js';

describe('Prefix Caching', () => {
  it('cache miss on empty cache', () => {
    const cache = new PrefixCache();
    const result = cache.lookup([1, 2, 3]);
    assert.equal(result.hit, false);
    assert.deepEqual(result.remainingTokens, [1, 2, 3]);
  });

  it('cache hit after store', () => {
    const cache = new PrefixCache();
    cache.store([1, 2, 3], 'kv-state-123');
    
    const result = cache.lookup([1, 2, 3, 4, 5]);
    assert.equal(result.hit, true);
    assert.equal(result.matchedLength, 3);
    assert.equal(result.kvState, 'kv-state-123');
    assert.deepEqual(result.remainingTokens, [4, 5]);
  });

  it('exact prefix match', () => {
    const cache = new PrefixCache();
    cache.store([10, 20, 30], 'state');
    
    const result = cache.lookup([10, 20, 30]);
    assert.equal(result.hit, true);
    assert.equal(result.matchedLength, 3);
    assert.deepEqual(result.remainingTokens, []);
  });

  it('no match for different prefix', () => {
    const cache = new PrefixCache();
    cache.store([1, 2, 3], 'state');
    
    const result = cache.lookup([4, 5, 6]);
    assert.equal(result.hit, false);
  });

  it('hit rate tracking', () => {
    const cache = new PrefixCache();
    cache.store([1, 2], 'state');
    
    cache.lookup([1, 2, 3]); // hit
    cache.lookup([1, 2, 4]); // hit
    cache.lookup([9, 9, 9]); // miss
    
    const stats = cache.stats();
    assert.equal(stats.hits, 2);
    assert.equal(stats.misses, 1);
    assert.equal(stats.hitRate, '66.7%');
  });

  it('LRU eviction when at capacity', () => {
    const cache = new PrefixCache(2);
    cache.store([1], 'a');
    cache.store([2], 'b');
    cache.store([3], 'c'); // should evict [1]

    assert.equal(cache.stats().entries, 2);
    assert.equal(cache.lookup([1]).hit, false);
    assert.equal(cache.lookup([2]).hit, true);
    assert.equal(cache.lookup([3]).hit, true);
  });

  it('shared system prompt scenario', () => {
    const cache = new PrefixCache();
    const systemPrompt = [100, 101, 102, 103, 104]; // "You are a helpful assistant..."
    
    cache.store(systemPrompt, 'system-kv');
    
    // Multiple requests with same system prompt
    const r1 = cache.lookup([...systemPrompt, 1, 2, 3]); // user: "hello"
    const r2 = cache.lookup([...systemPrompt, 4, 5]);     // user: "bye"
    const r3 = cache.lookup([...systemPrompt, 6, 7, 8]);  // user: "help me"
    
    assert.equal(r1.hit, true);
    assert.equal(r2.hit, true);
    assert.equal(r3.hit, true);
    assert.deepEqual(r1.remainingTokens, [1, 2, 3]);
    assert.deepEqual(r2.remainingTokens, [4, 5]);
    
    console.log(`  System prompt (5 tokens): cached, 3 requests saved ${5 * 3} token computations`);
  });
});
