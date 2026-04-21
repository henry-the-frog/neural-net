// modern-decoder.test.js — Tests for Llama-style decoder
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { RMSNorm, SwiGLUFFN, ModernDecoderBlock, ModernDecoder } from './modern-decoder.js';
import { Matrix } from './matrix.js';

describe('RMSNorm', () => {
  it('normalizes to unit RMS', () => {
    const norm = new RMSNorm(4);
    const input = new Matrix(1, 4);
    input.set(0, 0, 2); input.set(0, 1, 4); input.set(0, 2, 6); input.set(0, 3, 8);
    const output = norm.forward(input);

    // Check RMS of output ≈ 1.0 (since gamma = 1.0)
    let sumSq = 0;
    for (let i = 0; i < 4; i++) sumSq += output.get(0, i) ** 2;
    const rms = Math.sqrt(sumSq / 4);
    assert.ok(Math.abs(rms - 1.0) < 0.01, `RMS should be ~1.0, got ${rms}`);
  });

  it('preserves relative magnitudes', () => {
    const norm = new RMSNorm(4);
    const input = new Matrix(1, 4);
    input.set(0, 0, 1); input.set(0, 1, 2); input.set(0, 2, 3); input.set(0, 3, 4);
    const output = norm.forward(input);
    // Ratio should be preserved
    const ratio = output.get(0, 1) / output.get(0, 0);
    assert.ok(Math.abs(ratio - 2.0) < 0.01, `Ratio should be 2.0, got ${ratio}`);
  });
});

describe('SwiGLUFFN', () => {
  it('produces correct output shape', () => {
    const ffn = new SwiGLUFFN(8);
    const input = Matrix.random(3, 8); // 3 positions
    const output = ffn.forward(input);
    assert.equal(output.rows, 3);
    assert.equal(output.cols, 8);
  });

  it('Swish(0) = 0', () => {
    assert.ok(Math.abs(SwiGLUFFN.swish(0)) < 1e-10);
  });

  it('Swish is approximately linear for large positive x', () => {
    const x = 10;
    assert.ok(Math.abs(SwiGLUFFN.swish(x) - x) < 0.001);
  });

  it('Swish is smooth at 0 (non-zero gradient)', () => {
    const h = 1e-5;
    const grad = (SwiGLUFFN.swish(h) - SwiGLUFFN.swish(-h)) / (2 * h);
    assert.ok(Math.abs(grad - 0.5) < 0.01, `Gradient at 0 should be 0.5, got ${grad}`);
  });
});

describe('ModernDecoderBlock', () => {
  it('produces correct output shape', () => {
    const block = new ModernDecoderBlock(8, 4, 2);
    const input = Matrix.random(1, 16); // batch=1, seqLen=2
    const output = block.forward(input);
    assert.equal(output.rows, 1);
    assert.equal(output.cols, 16);
  });

  it('residual connections preserve information', () => {
    const block = new ModernDecoderBlock(4, 2, 1);
    const input = Matrix.random(1, 8); // 2 tokens
    const output = block.forward(input);

    // With random weights, output should be close to input (residual dominates)
    // but not identical (attention/FFN add signal)
    let diff = 0;
    for (let i = 0; i < input.cols; i++)
      diff += Math.abs(output.get(0, i) - input.get(0, i));
    assert.ok(diff > 0, 'Output should differ from input');
    assert.ok(isFinite(diff), 'Output should be finite');
  });

  it('KV-cache incremental generation', () => {
    const block = new ModernDecoderBlock(4, 2, 1);
    const full = Matrix.random(1, 8); // 2 tokens
    const fullOut = block.forward(full);

    block.clearCache();
    const t0 = new Matrix(1, 4);
    const t1 = new Matrix(1, 4);
    for (let d = 0; d < 4; d++) {
      t0.set(0, d, full.get(0, d));
      t1.set(0, d, full.get(0, 4 + d));
    }

    const out0 = block.forward(t0, true);
    const out1 = block.forward(t1, true);

    // Token 0 should match
    for (let d = 0; d < 4; d++) {
      assert.ok(
        Math.abs(out0.get(0, d) - fullOut.get(0, d)) < 1e-4,
        `Token 0 d=${d}: ${out0.get(0, d)} vs ${fullOut.get(0, d)}`
      );
    }
  });
});

describe('ModernDecoder (mini Llama)', () => {
  it('forward produces logits of correct shape', () => {
    const model = new ModernDecoder(2, 8, 4, 2, 32, { dHidden: 16 });
    const logits = model.forward([[0, 1, 2]], false);
    assert.equal(logits.rows, 1);
    assert.equal(logits.cols, 3 * 32); // seqLen * vocabSize
  });

  it('logits are finite', () => {
    const model = new ModernDecoder(2, 8, 4, 2, 32, { dHidden: 16 });
    const logits = model.forward([[0, 1]], false);
    for (let i = 0; i < logits.cols; i++)
      assert.ok(isFinite(logits.get(0, i)), `NaN at col ${i}`);
  });

  it('generate produces sequence of valid tokens', () => {
    const vocabSize = 16;
    const model = new ModernDecoder(2, 4, 2, 1, vocabSize, { dHidden: 8 });
    const generated = model.generate([0, 1], 5);
    assert.equal(generated.length, 2 + 5); // prompt + new tokens
    for (const t of generated) {
      assert.ok(t >= 0 && t < vocabSize, `Token ${t} out of range`);
    }
  });

  it('reports parameter count', () => {
    const model = new ModernDecoder(2, 8, 4, 2, 32, { dHidden: 16 });
    const params = model.paramCount();
    assert.ok(params > 0, `Should have params, got ${params}`);
    console.log(`Mini Llama: ${params} parameters (2 layers, d=8, vocab=32)`);
  });

  it('different prompts produce different outputs', () => {
    const model = new ModernDecoder(2, 4, 2, 1, 16, { dHidden: 8 });
    const out1 = model.forward([[0, 1]], false);
    const out2 = model.forward([[2, 3]], false);
    let diff = 0;
    for (let i = 0; i < out1.cols; i++) diff += Math.abs(out1.get(0, i) - out2.get(0, i));
    assert.ok(diff > 0.01, 'Different prompts should produce different logits');
  });
});
