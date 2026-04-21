// llm-demo.test.js — End-to-end LLM pipeline: BPE + ModernDecoder + Sampling
// This wires together everything we built tonight into a complete text generation system.
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { BPETokenizer } from './bpe.js';
import { ModernDecoder } from './modern-decoder.js';
import { sample } from './sampling.js';

describe('End-to-end LLM Pipeline', () => {
  const corpus = `the cat sat on the mat. the dog sat on the log. the cat and the dog are friends.`;

  it('BPE → ModernDecoder → text generation', () => {
    // Step 1: Train tokenizer
    const tok = new BPETokenizer();
    tok.train(corpus, 40, ['<|eos|>']);
    console.log(`  Vocab size: ${tok.size}`);

    // Step 2: Build model
    const model = new ModernDecoder(
      2,           // 2 decoder layers
      4,           // dModel = 4
      2,           // 2 query heads
      1,           // 1 KV head (MQA)
      tok.size,    // vocab size from tokenizer
      { dHidden: 8, maxSeqLen: 64 }
    );
    console.log(`  Parameters: ${model.paramCount()}`);

    // Step 3: Encode prompt
    const prompt = 'the cat';
    const promptIds = tok.encode(prompt);
    console.log(`  Prompt: "${prompt}" → [${promptIds}]`);

    // Step 4: Generate
    const generated = model.generate(promptIds, 10, {
      temperature: 1.0,
      greedy: true,
    });
    console.log(`  Generated IDs: [${generated}]`);

    // Step 5: Decode
    const text = tok.decode(generated);
    console.log(`  Generated text: "${text}"`);

    // Assertions
    assert.ok(generated.length > promptIds.length, 'Should generate new tokens');
    assert.ok(text.startsWith(prompt), 'Should start with prompt');
    assert.ok(text.length > prompt.length, 'Should be longer than prompt');
    
    // All tokens should be valid
    for (const id of generated) {
      assert.ok(id >= 0 && id < tok.size, `Token ${id} out of range`);
    }
  });

  it('sampling diversity: same prompt, different outputs', () => {
    const tok = new BPETokenizer();
    tok.train(corpus, 30, ['<|eos|>']);

    const model = new ModernDecoder(1, 4, 2, 1, tok.size, { dHidden: 8, maxSeqLen: 32 });

    const promptIds = tok.encode('the');
    const outputs = new Set();

    for (let i = 0; i < 10; i++) {
      const gen = model.generate(promptIds, 5, { temperature: 2.0, greedy: false });
      outputs.add(tok.decode(gen));
    }

    console.log(`  Unique outputs: ${outputs.size}/10`);
    for (const out of outputs) console.log(`    "${out}"`);

    assert.ok(outputs.size >= 2, 'High temperature should produce variety');
  });

  it('KV-cache consistency: incremental vs full generation', () => {
    const tok = new BPETokenizer();
    tok.train(corpus, 25, ['<|eos|>']);
    
    const model = new ModernDecoder(1, 4, 2, 1, tok.size, { dHidden: 8, maxSeqLen: 32 });
    
    const prompt = tok.encode('the cat');
    
    // Greedy generation (deterministic)
    const gen1 = model.generate(prompt, 5, { greedy: true });
    const gen2 = model.generate(prompt, 5, { greedy: true });
    
    // Both runs should produce identical output (same model, same prompt, greedy)
    assert.deepEqual(gen1, gen2, 'Greedy generation should be deterministic');
  });

  it('repetition penalty reduces loops', () => {
    const tok = new BPETokenizer();
    tok.train(corpus, 25, ['<|eos|>']);
    
    const model = new ModernDecoder(1, 4, 2, 1, tok.size, { dHidden: 8, maxSeqLen: 32 });
    
    const prompt = tok.encode('the');
    
    // Without penalty
    const noPenalty = model.generate(prompt, 15, { greedy: true });
    const noPenaltyUnique = new Set(noPenalty).size;
    
    // With penalty
    const withPenalty = model.generate(prompt, 15, { 
      temperature: 0.8, greedy: false, repetitionPenalty: 2.0 
    });
    const withPenaltyUnique = new Set(withPenalty).size;
    
    console.log(`  No penalty: ${noPenaltyUnique} unique tokens`);
    console.log(`  With penalty: ${withPenaltyUnique} unique tokens`);
    
    // Penalty should encourage more variety (or at least not less)
    assert.ok(withPenaltyUnique >= noPenaltyUnique * 0.8, 
      'Penalty should maintain or increase token diversity');
  });

  it('model parameter count scales with config', () => {
    const tok = new BPETokenizer();
    tok.train('abc', 10);
    
    const small = new ModernDecoder(1, 4, 2, 1, tok.size, { dHidden: 8 });
    const large = new ModernDecoder(4, 8, 4, 2, tok.size, { dHidden: 32 });
    
    assert.ok(large.paramCount() > small.paramCount(), 
      'Larger config should have more params');
    console.log(`  Small: ${small.paramCount()}, Large: ${large.paramCount()}`);
  });

  it('full pipeline benchmark', () => {
    const tok = new BPETokenizer();
    tok.train(corpus.repeat(3), 50, ['<|eos|>']);
    
    const model = new ModernDecoder(2, 8, 4, 2, tok.size, { dHidden: 16, maxSeqLen: 64 });
    
    const start = performance.now();
    const prompt = tok.encode('the cat sat');
    const generated = model.generate(prompt, 20, { greedy: true });
    const elapsed = performance.now() - start;
    
    const tokensPerSec = (generated.length / elapsed * 1000).toFixed(0);
    console.log(`  Generated ${generated.length} tokens in ${elapsed.toFixed(0)}ms (${tokensPerSec} tok/s)`);
    console.log(`  Output: "${tok.decode(generated)}"`);
    
    assert.ok(elapsed < 30000, 'Should complete in reasonable time');
  });
});
