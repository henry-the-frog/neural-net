// lora.test.js — Tests for LoRA adapter
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { LoRAAdapter, LoRAConfig } from './lora.js';
import { Matrix } from './matrix.js';

describe('LoRA', () => {
  describe('LoRAAdapter', () => {
    it('initial adapter produces no change (B=0)', () => {
      const adapter = new LoRAAdapter(4, 4, 2);
      const W = Matrix.random(4, 4);
      const x = Matrix.random(1, 4);

      const withLora = adapter.forward(x, W);
      const withoutLora = x.dot(W);

      for (let c = 0; c < 4; c++)
        assert.ok(Math.abs(withLora.get(0, c) - withoutLora.get(0, c)) < 1e-10,
          'Initial LoRA should not change output');
    });

    it('after training (B non-zero), output changes', () => {
      const adapter = new LoRAAdapter(4, 4, 2);
      // Simulate training by setting B to non-zero
      for (let r = 0; r < 4; r++)
        for (let c = 0; c < 2; c++)
          adapter.B.set(r, c, Math.random() * 0.1);

      const W = Matrix.random(4, 4);
      const x = Matrix.random(1, 4);

      const withLora = adapter.forward(x, W);
      const withoutLora = x.dot(W);

      let diff = 0;
      for (let c = 0; c < 4; c++)
        diff += Math.abs(withLora.get(0, c) - withoutLora.get(0, c));
      assert.ok(diff > 0.001, `LoRA should change output: diff=${diff}`);
    });

    it('disable makes adapter transparent', () => {
      const adapter = new LoRAAdapter(4, 4, 2);
      adapter.B = Matrix.random(4, 2).mul(0.1);
      adapter.enabled = false;

      const W = Matrix.random(4, 4);
      const x = Matrix.random(1, 4);

      const withLora = adapter.forward(x, W);
      const withoutLora = x.dot(W);

      for (let c = 0; c < 4; c++)
        assert.ok(Math.abs(withLora.get(0, c) - withoutLora.get(0, c)) < 1e-10);
    });

    it('merge bakes adapter into base weight', () => {
      const adapter = new LoRAAdapter(4, 4, 2);
      adapter.B = Matrix.random(4, 2).mul(0.1);

      const W = Matrix.random(4, 4);
      const x = Matrix.random(2, 4);

      // Forward with adapter
      const withLora = adapter.forward(x, W);

      // Merge and forward without adapter
      const merged = adapter.merge(W);
      const withMerged = x.dot(merged);

      for (let r = 0; r < 2; r++)
        for (let c = 0; c < 4; c++)
          assert.ok(Math.abs(withLora.get(r, c) - withMerged.get(r, c)) < 1e-10,
            `Merged should match at (${r},${c})`);
    });

    it('param count is much less than full', () => {
      const adapter = new LoRAAdapter(4096, 4096, 8);
      const loraParams = adapter.paramCount();
      const fullParams = adapter.fullParamCount();
      const ratio = adapter.compressionRatio();

      console.log(`  Full: ${fullParams.toLocaleString()}, LoRA: ${loraParams.toLocaleString()}, Ratio: ${ratio.toFixed(0)}x`);
      
      assert.ok(loraParams < fullParams / 100, 'LoRA should be < 1% of full params');
      assert.ok(ratio > 200, 'Compression ratio should be > 200x');
    });
  });

  describe('export/import', () => {
    it('roundtrips adapter weights', () => {
      const adapter = new LoRAAdapter(4, 4, 2);
      adapter.B = Matrix.random(4, 2).mul(0.1);

      const exported = adapter.export();
      const imported = LoRAAdapter.import(exported, 4, 4);

      const W = Matrix.random(4, 4);
      const x = Matrix.random(1, 4);

      const out1 = adapter.forward(x, W);
      const out2 = imported.forward(x, W);

      for (let c = 0; c < 4; c++)
        assert.ok(Math.abs(out1.get(0, c) - out2.get(0, c)) < 1e-10);
    });
  });

  describe('LoRAConfig', () => {
    it('estimates params correctly', () => {
      const config = new LoRAConfig(8, 8, ['Wq', 'Wv']);
      const loraParams = config.estimateParams(4096, 32);
      const fullParams = config.estimateFullParams(4096, 32);

      console.log(`  Llama-7B LoRA r=8 (Wq+Wv): ${(loraParams/1e6).toFixed(1)}M / ${(fullParams/1e6).toFixed(1)}M`);

      assert.ok(loraParams < fullParams / 100, 'LoRA should be < 1% of full fine-tune');
    });

    it('rank affects param count linearly', () => {
      const r4 = new LoRAConfig(4).estimateParams(512, 12);
      const r8 = new LoRAConfig(8).estimateParams(512, 12);
      const r16 = new LoRAConfig(16).estimateParams(512, 12);

      assert.ok(Math.abs(r8 / r4 - 2) < 0.1, 'Doubling rank should double params');
      assert.ok(Math.abs(r16 / r8 - 2) < 0.1, 'Doubling rank should double params');
    });
  });
});
