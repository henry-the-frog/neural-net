import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  MemoryBank, contentAddressing, locationAddressing,
  ReadHead, WriteHead, NTM,
} from '../src/ntm.js';

const approx = (a, b, eps = 0.01) => Math.abs(a - b) < eps;

describe('Memory Bank', () => {
  it('read with uniform weights gives average', () => {
    const mem = new MemoryBank(4, 3);
    mem.memory = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]];
    const weights = [0.25, 0.25, 0.25, 0.25];
    const result = mem.read(weights);
    assert.ok(approx(result[0], 0.5, 0.01));
    assert.ok(approx(result[1], 0.5, 0.01));
    assert.ok(approx(result[2], 0.5, 0.01));
  });

  it('read with one-hot weights selects row', () => {
    const mem = new MemoryBank(3, 2);
    mem.memory = [[1, 2], [3, 4], [5, 6]];
    const result = mem.read([0, 1, 0]);
    assert.ok(approx(result[0], 3));
    assert.ok(approx(result[1], 4));
  });

  it('write erases and adds', () => {
    const mem = new MemoryBank(2, 2);
    mem.memory = [[1, 1], [1, 1]];
    // Erase first row completely, add [5, 5]
    mem.write([1, 0], [1, 1], [5, 5]);
    assert.ok(approx(mem.memory[0][0], 5, 0.01));
    assert.ok(approx(mem.memory[1][0], 1, 0.01)); // Unchanged
  });

  it('reset clears memory', () => {
    const mem = new MemoryBank(3, 2);
    mem.memory[0] = [99, 99];
    mem.reset();
    assert.ok(mem.memory[0].every(v => Math.abs(v) < 0.01));
  });
});

describe('Content-Based Addressing', () => {
  it('focuses on matching row', () => {
    const memory = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const key = [1, 0, 0];
    const weights = contentAddressing(key, memory, 10);
    // Should strongly focus on first row
    assert.ok(weights[0] > 0.8, `Should focus on match: ${weights[0]}`);
    assert.ok(weights[1] < 0.15);
    assert.ok(weights[2] < 0.15);
  });

  it('weights sum to 1', () => {
    const memory = [[1, 2], [3, 4], [5, 6]];
    const key = [2, 3];
    const weights = contentAddressing(key, memory, 1);
    const sum = weights.reduce((a, b) => a + b, 0);
    assert.ok(approx(sum, 1, 0.001));
  });

  it('higher beta gives sharper focus', () => {
    const memory = [[1, 0], [0.9, 0.1], [0, 1]];
    const key = [1, 0];
    const w1 = contentAddressing(key, memory, 1);
    const w10 = contentAddressing(key, memory, 10);
    assert.ok(w10[0] > w1[0], `Higher beta should sharpen: ${w10[0]} vs ${w1[0]}`);
  });
});

describe('Location-Based Addressing', () => {
  it('pure interpolation with gate=0 returns previous', () => {
    const content = [0.8, 0.1, 0.1];
    const prev = [0.1, 0.1, 0.8];
    const result = locationAddressing(content, prev, {
      interpolationGate: 0,
      shiftWeights: [0, 1, 0], // no shift
      sharpening: 1,
    });
    for (let i = 0; i < 3; i++) {
      assert.ok(approx(result[i], prev[i], 0.05), `Should match previous at ${i}`);
    }
  });

  it('pure interpolation with gate=1 returns content', () => {
    const content = [0.8, 0.1, 0.1];
    const prev = [0.1, 0.1, 0.8];
    const result = locationAddressing(content, prev, {
      interpolationGate: 1,
      shiftWeights: [0, 1, 0],
      sharpening: 1,
    });
    for (let i = 0; i < 3; i++) {
      assert.ok(approx(result[i], content[i], 0.05));
    }
  });

  it('shift moves attention', () => {
    const content = [1, 0, 0, 0]; // Focus on slot 0
    const prev = [1, 0, 0, 0];
    const result = locationAddressing(content, prev, {
      interpolationGate: 1,
      shiftWeights: [1, 0, 0], // Shift left by 1 → focus moves to slot 1
      sharpening: 5,
    });
    // After left shift, focus should move from slot 0 to slot 1
    const maxIdx = result.indexOf(Math.max(...result));
    assert.ok(maxIdx !== 0 || result[1] > 0.1, `Should shift: ${result.map(v=>v.toFixed(2))}`);
  });
});

describe('Neural Turing Machine', () => {
  it('processes single step', () => {
    const ntm = new NTM(4, 2, 16, 8, 32);
    const output = ntm.step([1, 0, 0, 0]);
    assert.equal(output.length, 2);
    assert.ok(output.every(Number.isFinite));
  });

  it('processes sequence', () => {
    const ntm = new NTM(3, 2, 16, 8, 32);
    const inputs = Array.from({ length: 5 }, () =>
      Array.from({ length: 3 }, () => Math.random())
    );
    const outputs = ntm.processSequence(inputs);
    assert.equal(outputs.length, 5);
    assert.equal(outputs[0].length, 2);
  });

  it('memory changes after write', () => {
    const ntm = new NTM(3, 2, 8, 4, 16);
    const memBefore = ntm.memory.memory.map(row => [...row]);
    ntm.step([1, 1, 1]);
    const memAfter = ntm.memory.memory;

    let changed = false;
    for (let i = 0; i < ntm.memorySlots; i++) {
      for (let j = 0; j < ntm.slotSize; j++) {
        if (Math.abs(memAfter[i][j] - memBefore[i][j]) > 0.001) changed = true;
      }
    }
    assert.ok(changed, 'Memory should change after step');
  });

  it('reset clears state', () => {
    const ntm = new NTM(3, 2, 8, 4, 16);
    ntm.step([1, 1, 1]);
    ntm.reset();
    // Memory should be reset
    assert.ok(ntm.memory.memory[0].every(v => Math.abs(v) < 0.01));
    assert.ok(ntm.lastRead.every(v => v === 0));
  });

  it('different inputs produce different outputs', () => {
    const ntm = new NTM(4, 2, 16, 8, 32);
    const out1 = ntm.step([1, 0, 0, 0]);
    ntm.reset();
    const out2 = ntm.step([0, 0, 0, 1]);

    let different = false;
    for (let i = 0; i < 2; i++) {
      if (Math.abs(out1[i] - out2[i]) > 0.001) different = true;
    }
    assert.ok(different, 'Different inputs should produce different outputs');
  });
});
