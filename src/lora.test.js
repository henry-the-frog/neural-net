// lora.test.js — LoRA (Low-Rank Adaptation) tests
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { LoRALayer } from './lora.js';
import { Matrix } from './matrix.js';

describe('LoRA', () => {
  test('initial forward matches base weight (B=0)', () => {
    const W = Matrix.random(4, 8);
    const lora = new LoRALayer(W, 2);
    
    // B is initialized to 0, so ΔW = B@A = 0
    const input = Matrix.random(3, 8);
    const output = lora.forward(input);
    
    // Should match input @ W^T
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 4; j++) {
        let expected = 0;
        for (let k = 0; k < 8; k++) expected += input.get(i, k) * W.get(j, k);
        assert.ok(Math.abs(output.get(i, j) - expected) < 0.001,
          `Initial forward should match base at (${i},${j})`);
      }
    }
  });

  test('forward output shape', () => {
    const W = Matrix.random(4, 8);
    const lora = new LoRALayer(W, 2);
    const input = Matrix.random(5, 8);
    const out = lora.forward(input);
    assert.equal(out.rows, 5);
    assert.equal(out.cols, 4);
  });

  test('backward produces correct gradient shape', () => {
    const W = Matrix.random(4, 8);
    const lora = new LoRALayer(W, 2);
    const input = Matrix.random(3, 8);
    lora.forward(input);
    
    const dOut = Matrix.random(3, 4);
    const dInput = lora.backward(dOut);
    assert.equal(dInput.rows, 3);
    assert.equal(dInput.cols, 8);
    assert.equal(lora.dA.rows, 2);
    assert.equal(lora.dA.cols, 8);
    assert.equal(lora.dB.rows, 4);
    assert.equal(lora.dB.cols, 2);
  });

  test('update only modifies A and B, not base weight', () => {
    const W = Matrix.random(4, 8);
    const origW = new Float64Array(W.data);
    const lora = new LoRALayer(W, 2);
    
    const input = Matrix.random(3, 8);
    lora.forward(input);
    const dOut = Matrix.ones(3, 4);
    lora.backward(dOut);
    lora.update(0.01);
    
    // Base weight should not change
    for (let i = 0; i < W.data.length; i++) {
      assert.equal(W.data[i], origW[i], 'Base weight should be frozen');
    }
  });

  test('after training, forward differs from base', () => {
    const W = Matrix.random(4, 8);
    const lora = new LoRALayer(W, 2);
    
    const input = Matrix.random(3, 8);
    lora.forward(input);
    const dOut = Matrix.ones(3, 4);
    lora.backward(dOut);
    lora.update(0.1); // Big LR to ensure visible change
    
    const out = lora.forward(input);
    // Should differ from initial (B is no longer 0)
    let diff = 0;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 4; j++) {
        let base = 0;
        for (let k = 0; k < 8; k++) base += input.get(i, k) * W.get(j, k);
        diff += Math.abs(out.get(i, j) - base);
      }
    }
    assert.ok(diff > 0.001, 'After training, output should differ from base');
  });

  test('merge produces correct combined weight', () => {
    const W = Matrix.random(4, 8);
    const lora = new LoRALayer(W, 2);
    
    // Manually set B to non-zero
    for (let i = 0; i < lora.B.data.length; i++) lora.B.data[i] = 0.1;
    
    const merged = lora.merge();
    assert.equal(merged.rows, 4);
    assert.equal(merged.cols, 8);
    
    // merged should differ from W (since B is non-zero now)
    let diff = 0;
    for (let i = 0; i < merged.data.length; i++) {
      diff += Math.abs(merged.data[i] - W.data[i]);
    }
    assert.ok(diff > 0.01, 'Merged weight should differ from base');
  });

  test('paramCount is much less than baseParamCount', () => {
    const W = Matrix.random(64, 64); // 4096 params
    const lora = new LoRALayer(W, 4); // 64*4 + 4*64 = 512 params
    
    const savings = lora.savings();
    assert.equal(savings.loraParams, 512);
    assert.equal(savings.baseParams, 4096);
    assert.equal(savings.ratio, '12.50%');
    assert.equal(savings.savings, '87.50%');
  });

  test('higher rank = more expressive', () => {
    const W = Matrix.random(8, 8);
    const lora2 = new LoRALayer(W, 2);
    const lora8 = new LoRALayer(W, 8); // rank = dim → full rank
    
    assert.ok(lora8.paramCount() > lora2.paramCount());
  });
});
