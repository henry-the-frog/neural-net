// prefix-caching.js — Prefix Caching for LLM Inference
// Used by: vLLM, SGLang
// When multiple requests share the same system prompt or prefix,
// cache the KV-cache for the shared prefix and reuse it.
// Massive speedup for chat applications where system prompts are constant.

/**
 * Prefix Cache: stores KV-cache for common prefixes.
 * Uses hash-based lookup to find matching cached prefixes.
 */
export class PrefixCache {
  constructor(maxEntries = 1024) {
    this.maxEntries = maxEntries;
    this.cache = new Map(); // hash → { kvState, tokenIds, hitCount, lastAccess }
    this.hits = 0;
    this.misses = 0;
  }

  /**
   * Hash a token sequence for cache lookup.
   */
  static hashTokens(tokens) {
    // Simple hash for token sequence
    let hash = 0;
    for (let i = 0; i < tokens.length; i++) {
      hash = ((hash << 5) - hash + tokens[i]) | 0;
    }
    return hash.toString(36);
  }

  /**
   * Look up a prefix in the cache.
   * Returns the longest cached prefix that matches the start of the given tokens.
   */
  lookup(tokens) {
    // Try progressively shorter prefixes
    for (let len = tokens.length; len > 0; len--) {
      const prefix = tokens.slice(0, len);
      const hash = PrefixCache.hashTokens(prefix);
      const entry = this.cache.get(hash);
      
      if (entry && arraysEqual(entry.tokenIds, prefix)) {
        entry.hitCount++;
        entry.lastAccess = Date.now();
        this.hits++;
        return {
          hit: true,
          matchedLength: len,
          kvState: entry.kvState,
          remainingTokens: tokens.slice(len),
        };
      }
    }

    this.misses++;
    return { hit: false, matchedLength: 0, kvState: null, remainingTokens: tokens };
  }

  /**
   * Store a prefix's KV-cache state.
   */
  store(tokens, kvState) {
    const hash = PrefixCache.hashTokens(tokens);
    
    // Evict if at capacity (LRU)
    if (this.cache.size >= this.maxEntries) {
      let oldest = null, oldestKey = null;
      for (const [key, entry] of this.cache) {
        if (!oldest || entry.lastAccess < oldest.lastAccess) {
          oldest = entry;
          oldestKey = key;
        }
      }
      if (oldestKey) this.cache.delete(oldestKey);
    }

    this.cache.set(hash, {
      tokenIds: [...tokens],
      kvState,
      hitCount: 0,
      lastAccess: Date.now(),
    });
  }

  stats() {
    const total = this.hits + this.misses;
    return {
      entries: this.cache.size,
      hits: this.hits,
      misses: this.misses,
      hitRate: total > 0 ? (this.hits / total * 100).toFixed(1) + '%' : '0%',
    };
  }

  clear() {
    this.cache.clear();
    this.hits = 0;
    this.misses = 0;
  }
}

function arraysEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
