// tokenizer-analysis.test.js
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { BPETokenizer } from './bpe.js';
import { analyzeTokenization, analyzeMerges, compareTexts, vocabularyCoverage } from './tokenizer-analysis.js';

describe('Tokenizer Analysis', () => {
  const corpus = 'the cat sat on the mat the dog sat on the log the cat and the dog';

  function makeTok() {
    const tok = new BPETokenizer();
    tok.train(corpus, 40, ['<|eos|>']);
    return tok;
  }

  it('analyzeTokenization provides stats', () => {
    const tok = makeTok();
    const result = analyzeTokenization(tok, 'the cat sat on the mat');
    console.log('  Stats:', JSON.stringify(result, null, 2));

    assert.ok(result.characters > 0);
    assert.ok(result.tokens > 0);
    assert.ok(parseFloat(result.compressionRatio) > 1, 'Should compress');
    assert.ok(result.topTokens.length > 0);
  });

  it('compression ratio > 1 for trained text', () => {
    const tok = makeTok();
    const result = analyzeTokenization(tok, corpus);
    assert.ok(parseFloat(result.compressionRatio) > 1.5,
      `Should compress well: ${result.compressionRatio}`);
  });

  it('analyzeMerges shows merge history', () => {
    const tok = makeTok();
    const merges = analyzeMerges(tok);
    assert.ok(merges.length > 0);
    // First merge should be most common pair in corpus
    console.log(`  First 3 merges: ${merges.slice(0, 3).map(m => m.pair + ' → ' + m.result).join(', ')}`);
  });

  it('compareTexts across different text types', () => {
    const tok = makeTok();
    const results = compareTexts(tok, [
      { label: 'Training', text: corpus },
      { label: 'Similar', text: 'the cat sat' },
      { label: 'Numbers', text: '123456789' },
    ]);

    assert.equal(results.length, 3);
    for (const r of results) {
      console.log(`  ${r.label}: ${r.compressionRatio}x compression, ${r.uniqueTokens} unique`);
    }
  });

  it('vocabularyCoverage for trained corpus', () => {
    const tok = makeTok();
    const cov = vocabularyCoverage(tok, corpus);
    assert.equal(cov.coverage, '100.0%', 'Training corpus should be fully covered');
  });

  it('vocabularyCoverage shows gaps for unseen chars', () => {
    const tok = makeTok();
    const cov = vocabularyCoverage(tok, 'xyz123!@#');
    assert.ok(cov.uncovered > 0, 'Unseen chars should be uncovered');
    console.log(`  Unseen text: ${cov.coverage} covered (${cov.uncovered} missing)`);
  });
});
