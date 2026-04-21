// paged-attention.js — Paged Attention (vLLM-style)
// Paper: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
// (Kwon et al., 2023)
//
// Problem: Each request needs its own KV-cache, leading to memory fragmentation.
// A batch of 8 requests with 4K context each needs 8 separate contiguous allocations.
//
// Solution: Manage KV-cache like OS virtual memory:
// - Physical pages: fixed-size blocks of K/V vectors
// - Page table: maps logical positions → physical pages
// - Requests share a common page pool, allocated on demand

/**
 * Physical page: stores up to `blockSize` K/V vectors.
 */
class KVPage {
  constructor(blockSize, headDim) {
    this.blockSize = blockSize;
    this.headDim = headDim;
    this.keys = [];
    this.values = [];
  }

  append(k, v) {
    if (this.keys.length >= this.blockSize) return false;
    this.keys.push(Float64Array.from(k));
    this.values.push(Float64Array.from(v));
    return true;
  }

  get size() { return this.keys.length; }
  get full() { return this.keys.length >= this.blockSize; }
}

/**
 * Paged KV-Cache Manager
 * Manages a pool of physical pages shared across multiple requests.
 */
export class PagedKVCacheManager {
  constructor(headDim, blockSize = 16, maxPages = 1024) {
    this.headDim = headDim;
    this.blockSize = blockSize;
    this.maxPages = maxPages;

    // Page pool
    this.freePages = [];
    this.allocatedPages = 0;

    // Per-request page tables: requestId → [pageId, ...]
    this.pageTables = new Map();
    // pageId → KVPage
    this.pages = new Map();
    this.nextPageId = 0;
  }

  /**
   * Start a new request.
   */
  createRequest(requestId) {
    this.pageTables.set(requestId, []);
  }

  /**
   * Append a token's K/V to a request's cache.
   */
  appendToken(requestId, k, v) {
    const table = this.pageTables.get(requestId);
    if (!table) throw new Error(`Unknown request: ${requestId}`);

    // Try to append to last page
    if (table.length > 0) {
      const lastPageId = table[table.length - 1];
      const lastPage = this.pages.get(lastPageId);
      if (lastPage && !lastPage.full) {
        lastPage.append(k, v);
        return;
      }
    }

    // Need a new page
    const page = this._allocatePage();
    page.append(k, v);
    table.push(page._id);
  }

  /**
   * Get all K vectors for a request.
   */
  getKeys(requestId) {
    return this._getAllVectors(requestId, 'keys');
  }

  /**
   * Get all V vectors for a request.
   */
  getValues(requestId) {
    return this._getAllVectors(requestId, 'values');
  }

  /**
   * Release all pages for a completed request.
   */
  freeRequest(requestId) {
    const table = this.pageTables.get(requestId);
    if (table) {
      for (const pageId of table) {
        this.pages.delete(pageId);
        this.freePages.push(pageId);
      }
      this.pageTables.delete(requestId);
    }
  }

  /**
   * Get cache tokens for a request.
   */
  requestSize(requestId) {
    const table = this.pageTables.get(requestId);
    if (!table) return 0;
    let count = 0;
    for (const pageId of table) {
      const page = this.pages.get(pageId);
      if (page) count += page.size;
    }
    return count;
  }

  /**
   * Global statistics.
   */
  stats() {
    const totalPages = this.allocatedPages;
    const usedPages = this.pages.size;
    const freePages = this.freePages.length;
    const requests = this.pageTables.size;

    let totalTokens = 0;
    for (const [, page] of this.pages) totalTokens += page.size;

    return {
      totalPages,
      usedPages,
      freePages,
      requests,
      totalTokens,
      memoryElements: totalTokens * this.headDim * 2,
      utilization: usedPages > 0 ?
        (totalTokens / (usedPages * this.blockSize) * 100).toFixed(1) + '%' : '0%',
    };
  }

  // --- Private ---

  _allocatePage() {
    let pageId;
    if (this.freePages.length > 0) {
      pageId = this.freePages.pop();
    } else {
      if (this.allocatedPages >= this.maxPages) {
        throw new Error('Out of pages');
      }
      pageId = this.nextPageId++;
      this.allocatedPages++;
    }

    const page = new KVPage(this.blockSize, this.headDim);
    page._id = pageId;
    this.pages.set(pageId, page);
    return page;
  }

  _getAllVectors(requestId, field) {
    const table = this.pageTables.get(requestId);
    if (!table) return [];

    const vectors = [];
    for (const pageId of table) {
      const page = this.pages.get(pageId);
      if (page) {
        for (const vec of page[field]) vectors.push(vec);
      }
    }
    return vectors;
  }
}
