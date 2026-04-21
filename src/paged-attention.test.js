// paged-attention.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { PagedKVCacheManager } from './paged-attention.js';

describe('Paged Attention', () => {
  it('basic request lifecycle', () => {
    const mgr = new PagedKVCacheManager(4, 4, 100);
    mgr.createRequest('r1');
    mgr.appendToken('r1', [1,2,3,4], [5,6,7,8]);
    assert.equal(mgr.requestSize('r1'), 1);
    mgr.freeRequest('r1');
    assert.equal(mgr.stats().usedPages, 0);
  });

  it('multiple tokens span pages', () => {
    const mgr = new PagedKVCacheManager(2, 2, 100); // block=2
    mgr.createRequest('r1');
    for (let i = 0; i < 5; i++) mgr.appendToken('r1', [i, i], [i, i]);
    
    assert.equal(mgr.requestSize('r1'), 5);
    // 5 tokens / block=2 = 3 pages
    assert.equal(mgr.stats().usedPages, 3);
  });

  it('multiple requests share page pool', () => {
    const mgr = new PagedKVCacheManager(2, 4, 100);
    mgr.createRequest('r1');
    mgr.createRequest('r2');

    for (let i = 0; i < 3; i++) {
      mgr.appendToken('r1', [i, i], [i, i]);
      mgr.appendToken('r2', [10+i, 10+i], [10+i, 10+i]);
    }

    assert.equal(mgr.requestSize('r1'), 3);
    assert.equal(mgr.requestSize('r2'), 3);
    assert.equal(mgr.stats().requests, 2);
  });

  it('freed pages are reused', () => {
    const mgr = new PagedKVCacheManager(2, 4, 10);
    
    // Fill with r1
    mgr.createRequest('r1');
    for (let i = 0; i < 8; i++) mgr.appendToken('r1', [i, i], [i, i]);
    const pagesAfterR1 = mgr.stats().totalPages;
    
    // Free r1
    mgr.freeRequest('r1');
    assert.equal(mgr.stats().freePages, pagesAfterR1);
    
    // r2 should reuse pages
    mgr.createRequest('r2');
    for (let i = 0; i < 4; i++) mgr.appendToken('r2', [i, i], [i, i]);
    assert.equal(mgr.stats().totalPages, pagesAfterR1, 'Should reuse, not allocate new');
  });

  it('out of pages throws', () => {
    const mgr = new PagedKVCacheManager(2, 2, 3); // max 3 pages × 2 = 6 tokens
    mgr.createRequest('r1');
    for (let i = 0; i < 6; i++) mgr.appendToken('r1', [i, i], [i, i]);
    
    assert.throws(
      () => mgr.appendToken('r1', [99, 99], [99, 99]),
      /Out of pages/
    );
  });

  it('getKeys/getValues returns all tokens', () => {
    const mgr = new PagedKVCacheManager(2, 2, 100);
    mgr.createRequest('r1');
    mgr.appendToken('r1', [1, 2], [3, 4]);
    mgr.appendToken('r1', [5, 6], [7, 8]);
    mgr.appendToken('r1', [9, 10], [11, 12]);

    const keys = mgr.getKeys('r1');
    assert.equal(keys.length, 3);
    assert.deepEqual(Array.from(keys[0]), [1, 2]);
    assert.deepEqual(Array.from(keys[2]), [9, 10]);
  });

  it('utilization tracking', () => {
    const mgr = new PagedKVCacheManager(4, 8, 100);
    mgr.createRequest('r1');
    for (let i = 0; i < 5; i++) mgr.appendToken('r1', new Array(4).fill(i), new Array(4).fill(i));
    
    const stats = mgr.stats();
    console.log(`  5 tokens, block=8: utilization=${stats.utilization}`);
    // 5/8 = 62.5% utilization
    assert.ok(parseFloat(stats.utilization) > 50);
  });
});
