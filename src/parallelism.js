// parallelism.js — Distributed Training Concepts (Educational)
// How LLMs are trained across multiple GPUs:
// 1. Data Parallelism (DP/DDP): same model on all GPUs, different data
// 2. Tensor Parallelism (TP): split weight matrices across GPUs
// 3. Pipeline Parallelism (PP): split layers across GPUs
// 4. ZeRO: partition optimizer state, gradients, and parameters

/**
 * Data Parallelism Simulator
 * Each "worker" processes different data, gradients are averaged.
 */
export class DataParallelism {
  constructor(numWorkers) {
    this.numWorkers = numWorkers;
  }

  /**
   * Simulate: split batch, compute gradients per worker, all-reduce.
   * @param {number[][]} batch - full batch of data
   * @param {function} computeGrad - (data) → gradient array
   * @returns {{ avgGrad: number[], efficiency: number, communicationCost: string }}
   */
  step(batch, computeGrad) {
    const batchPerWorker = Math.ceil(batch.length / this.numWorkers);
    const grads = [];

    // Each worker computes gradients on its shard
    for (let w = 0; w < this.numWorkers; w++) {
      const start = w * batchPerWorker;
      const shard = batch.slice(start, start + batchPerWorker);
      if (shard.length > 0) grads.push(computeGrad(shard));
    }

    // All-reduce: average gradients
    const gradLen = grads[0].length;
    const avgGrad = new Array(gradLen).fill(0);
    for (const g of grads) {
      for (let i = 0; i < gradLen; i++) avgGrad[i] += g[i];
    }
    for (let i = 0; i < gradLen; i++) avgGrad[i] /= grads.length;

    return {
      avgGrad,
      efficiency: grads.length / this.numWorkers, // may be < 1 if batch too small
      communicationCost: `${this.numWorkers - 1} all-reduce ops`,
    };
  }

  /**
   * Throughput analysis.
   */
  analysis(batchSize, seqLen, modelParams) {
    const effectiveBatch = batchSize * this.numWorkers;
    const memoryPerGPU = modelParams; // each GPU holds full model + optimizer
    return {
      effectiveBatchSize: effectiveBatch,
      memoryPerGPU: `${(memoryPerGPU / 1e9).toFixed(1)}B params + optimizer state`,
      throughputScaling: `~${this.numWorkers}x (linear with good interconnect)`,
      limitation: 'Each GPU must hold the full model + optimizer',
    };
  }
}

/**
 * Tensor Parallelism Simulator
 * Splits weight matrices across GPUs (column-wise or row-wise).
 */
export class TensorParallelism {
  constructor(numGPUs) {
    this.numGPUs = numGPUs;
  }

  /**
   * Analyze how a weight matrix would be split.
   */
  splitAnalysis(rows, cols) {
    const colsPerGPU = Math.ceil(cols / this.numGPUs);
    return {
      originalSize: rows * cols,
      shardSize: rows * colsPerGPU,
      memoryPerGPU: `${(rows * colsPerGPU / 1e6).toFixed(1)}M params`,
      communicationCost: 'All-reduce after each layer',
      benefit: 'Each GPU holds 1/N of the model weights',
    };
  }
}

/**
 * Pipeline Parallelism Simulator
 * Splits layers across GPUs — micro-batching to fill the pipeline.
 */
export class PipelineParallelism {
  constructor(numStages, numMicroBatches) {
    this.numStages = numStages;
    this.numMicroBatches = numMicroBatches;
  }

  /**
   * Calculate pipeline efficiency (bubble ratio).
   * Bubble = idle time when stages wait for data.
   */
  efficiency() {
    // Total time units: numStages + numMicroBatches - 1
    const totalSteps = this.numStages + this.numMicroBatches - 1;
    const busySteps = this.numMicroBatches; // per stage
    const bubbleSteps = this.numStages - 1;

    return {
      totalSteps,
      busyStepsPerStage: busySteps,
      bubbleSteps,
      efficiency: (busySteps / totalSteps * 100).toFixed(1) + '%',
      bubbleRatio: (bubbleSteps / totalSteps * 100).toFixed(1) + '%',
    };
  }
}

/**
 * ZeRO (Zero Redundancy Optimizer) Analysis
 * Three stages of memory optimization.
 */
export function zeroAnalysis(modelParams, numGPUs) {
  const fp32Bytes = 4;
  const fp16Bytes = 2;

  // Standard (no ZeRO): each GPU stores model + optimizer + gradients
  const modelMem = modelParams * fp16Bytes;
  const optimizerMem = modelParams * fp32Bytes * 3; // params + m + v (AdamW)
  const gradMem = modelParams * fp16Bytes;
  const standard = modelMem + optimizerMem + gradMem;

  // ZeRO Stage 1: partition optimizer state
  const zero1 = modelMem + optimizerMem / numGPUs + gradMem;

  // ZeRO Stage 2: + partition gradients
  const zero2 = modelMem + optimizerMem / numGPUs + gradMem / numGPUs;

  // ZeRO Stage 3: + partition parameters
  const zero3 = (modelMem + optimizerMem + gradMem) / numGPUs;

  return {
    standard: formatBytes(standard),
    zero1: formatBytes(zero1),
    zero2: formatBytes(zero2),
    zero3: formatBytes(zero3),
    zero3Savings: ((1 - zero3 / standard) * 100).toFixed(1) + '%',
  };
}

function formatBytes(bytes) {
  if (bytes > 1e9) return (bytes / 1e9).toFixed(1) + 'GB';
  if (bytes > 1e6) return (bytes / 1e6).toFixed(1) + 'MB';
  return bytes + 'B';
}
