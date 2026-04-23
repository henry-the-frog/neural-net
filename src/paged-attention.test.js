import { test, describe } from 'node:test';
import assert from 'node:assert/strict';

describe('PagedAttention', () => {
  // vLLM-style paged attention: map logical KV cache to physical blocks
  function allocateBlocks(seqLen, blockSize) {
    return Math.ceil(seqLen / blockSize);
  }

  function blockTable(seqLen, blockSize) {
    const nBlocks = allocateBlocks(seqLen, blockSize);
    return Array.from({length: nBlocks}, (_, i) => ({
      blockId: i,
      startPos: i * blockSize,
      endPos: Math.min((i + 1) * blockSize, seqLen),
    }));
  }

  test('allocates correct number of blocks', () => {
    assert.equal(allocateBlocks(100, 16), 7);
    assert.equal(allocateBlocks(16, 16), 1);
  });

  test('block table covers full sequence', () => {
    const table = blockTable(100, 16);
    assert.equal(table[0].startPos, 0);
    assert.equal(table[table.length - 1].endPos, 100);
  });
});
