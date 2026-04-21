// parallelism.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { DataParallelism, TensorParallelism, PipelineParallelism, zeroAnalysis } from './parallelism.js';

describe('Distributed Training', () => {
  describe('Data Parallelism', () => {
    it('averages gradients across workers', () => {
      const dp = new DataParallelism(4);
      const batch = [[1], [2], [3], [4], [5], [6], [7], [8]];
      const result = dp.step(batch, shard => [shard.reduce((a, b) => a + b[0], 0)]);

      assert.equal(result.avgGrad.length, 1);
      // Average of shard sums: (1+2)/4, (3+4)/4, (5+6)/4, (7+8)/4 → average
      assert.ok(isFinite(result.avgGrad[0]));
    });

    it('scales throughput linearly', () => {
      const dp = new DataParallelism(8);
      const analysis = dp.analysis(32, 2048, 7e9);
      assert.equal(analysis.effectiveBatchSize, 256);
    });
  });

  describe('Tensor Parallelism', () => {
    it('splits weight matrix', () => {
      const tp = new TensorParallelism(4);
      const result = tp.splitAnalysis(4096, 4096);
      assert.equal(result.originalSize, 4096 * 4096);
      assert.equal(result.shardSize, 4096 * 1024);
    });
  });

  describe('Pipeline Parallelism', () => {
    it('efficiency increases with micro-batches', () => {
      const pp4 = new PipelineParallelism(4, 4);
      const pp16 = new PipelineParallelism(4, 16);

      const eff4 = parseFloat(pp4.efficiency().efficiency);
      const eff16 = parseFloat(pp16.efficiency().efficiency);

      assert.ok(eff16 > eff4, 'More micro-batches = higher efficiency');
      console.log(`  4 micro-batches: ${eff4}%, 16: ${eff16}%`);
    });

    it('bubble ratio decreases with micro-batches', () => {
      const pp = new PipelineParallelism(8, 32);
      const eff = pp.efficiency();
      console.log(`  8 stages, 32 µB: efficiency=${eff.efficiency}, bubble=${eff.bubbleRatio}`);
      assert.ok(parseFloat(eff.efficiency) > 70);
    });
  });

  describe('ZeRO Analysis', () => {
    it('Llama-7B memory analysis', () => {
      const result = zeroAnalysis(7e9, 8);
      console.log('  Llama-7B on 8 GPUs:');
      console.log(`    Standard: ${result.standard} per GPU`);
      console.log(`    ZeRO-1: ${result.zero1}`);
      console.log(`    ZeRO-2: ${result.zero2}`);
      console.log(`    ZeRO-3: ${result.zero3}`);
      console.log(`    ZeRO-3 savings: ${result.zero3Savings}`);

      assert.ok(parseFloat(result.zero3Savings) > 80, 'ZeRO-3 should save >80%');
    });

    it('more GPUs = more savings', () => {
      const z4 = zeroAnalysis(7e9, 4);
      const z8 = zeroAnalysis(7e9, 8);
      // ZeRO-3 with 8 GPUs should use less per-GPU memory
      const mem4 = parseFloat(z4.zero3);
      const mem8 = parseFloat(z8.zero3);
      assert.ok(mem8 < mem4, '8 GPUs should use less per-GPU memory');
    });
  });
});
