// continuous-batching.js — Continuous Batching for LLM Serving
// Used by: vLLM, TGI, Orca
//
// Problem: Static batching wastes GPU when requests finish at different times.
// Solution: Dynamically add/remove requests from the batch each iteration.
//
// Key insight: In a batch, different requests are at different stages:
// - New requests need prefill (process full prompt)
// - Running requests need decode (generate one token at a time)
// - Finished requests should be evicted immediately

/**
 * Request state machine: QUEUED → PREFILL → RUNNING → DONE
 */
export class Request {
  constructor(id, promptTokens, maxNewTokens, eosToken = -1) {
    this.id = id;
    this.promptTokens = promptTokens;
    this.maxNewTokens = maxNewTokens;
    this.eosToken = eosToken;
    this.generatedTokens = [];
    this.state = 'queued';
    this.arrivedAt = Date.now();
    this.startedAt = null;
    this.finishedAt = null;
  }

  get totalTokens() {
    return this.promptTokens.length + this.generatedTokens.length;
  }

  get isDone() {
    return this.state === 'done';
  }
}

/**
 * Continuous Batching Scheduler
 *
 * Each iteration:
 * 1. Evict finished requests
 * 2. Add new requests from queue (up to batch capacity)
 * 3. Run prefill for new requests
 * 4. Run decode for running requests
 * 5. Check for completion (EOS or max tokens)
 */
export class ContinuousBatchingScheduler {
  constructor({ maxBatchSize = 8, maxSeqLen = 2048 } = {}) {
    this.maxBatchSize = maxBatchSize;
    this.maxSeqLen = maxSeqLen;

    this.queue = [];      // waiting requests
    this.running = [];    // active requests in batch
    this.completed = [];  // finished requests
    this.step = 0;
  }

  /**
   * Add a new request to the queue.
   */
  addRequest(request) {
    this.queue.push(request);
  }

  /**
   * Run one batch iteration.
   * @param {function} generateToken - (requestId, prompt) → nextToken
   * @returns {{ prefilled: number, decoded: number, completed: number }}
   */
  iterate(generateToken) {
    this.step++;
    let prefilled = 0, decoded = 0, completedCount = 0;

    // 1. Evict finished requests
    const stillRunning = [];
    for (const req of this.running) {
      if (req.isDone) {
        this.completed.push(req);
        completedCount++;
      } else {
        stillRunning.push(req);
      }
    }
    this.running = stillRunning;

    // 2. Admit new requests from queue
    while (this.running.length < this.maxBatchSize && this.queue.length > 0) {
      const req = this.queue.shift();
      req.state = 'prefill';
      req.startedAt = Date.now();
      this.running.push(req);
    }

    // 3. Process each request
    for (const req of this.running) {
      if (req.state === 'prefill') {
        // Prefill: process the entire prompt at once
        const allTokens = [...req.promptTokens];
        const nextToken = generateToken(req.id, allTokens);
        req.generatedTokens.push(nextToken);
        req.state = 'running';
        prefilled++;

        // Check if done
        if (nextToken === req.eosToken || req.generatedTokens.length >= req.maxNewTokens) {
          req.state = 'done';
          req.finishedAt = Date.now();
        }
      } else if (req.state === 'running') {
        // Decode: generate one token
        const allTokens = [...req.promptTokens, ...req.generatedTokens];
        const nextToken = generateToken(req.id, allTokens);
        req.generatedTokens.push(nextToken);
        decoded++;

        // Check if done
        if (nextToken === req.eosToken || req.generatedTokens.length >= req.maxNewTokens) {
          req.state = 'done';
          req.finishedAt = Date.now();
        }
      }
    }

    return { prefilled, decoded, completed: completedCount };
  }

  /**
   * Run until all requests are complete.
   */
  runAll(generateToken) {
    const iterStats = [];
    while (this.queue.length > 0 || this.running.length > 0) {
      const stats = this.iterate(generateToken);
      iterStats.push(stats);
    }
    return {
      iterations: iterStats.length,
      totalCompleted: this.completed.length,
      avgLatency: this.completed.length > 0 ?
        this.completed.reduce((sum, r) => sum + (r.finishedAt - r.startedAt), 0) / this.completed.length : 0,
    };
  }

  stats() {
    return {
      queued: this.queue.length,
      running: this.running.length,
      completed: this.completed.length,
      step: this.step,
    };
  }
}
