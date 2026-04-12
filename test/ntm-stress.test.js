// ntm-stress.test.js — Deep stress tests for Neural Turing Machine
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  MemoryBank, contentAddressing, locationAddressing,
  ReadHead, WriteHead, NTM,
} from '../src/ntm.js';

describe('MemoryBank Stress', () => {
  it('read with uniform weights returns mean of all slots', () => {
    const bank = new MemoryBank(4, 3);
    bank.memory = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]];
    const weights = [0.25, 0.25, 0.25, 0.25];
    const result = bank.read(weights);
    assert.ok(Math.abs(result[0] - 0.5) < 0.01);
    assert.ok(Math.abs(result[1] - 0.5) < 0.01);
    assert.ok(Math.abs(result[2] - 0.5) < 0.01);
  });

  it('read with one-hot weights returns exact slot', () => {
    const bank = new MemoryBank(4, 3);
    bank.memory = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]];
    const weights = [0, 0, 1, 0];
    const result = bank.read(weights);
    assert.deepEqual(result, [7, 8, 9]);
  });

  it('write then read should retrieve written content', () => {
    const bank = new MemoryBank(4, 3);
    const writeWeights = [0, 0, 1, 0]; // Write to slot 2
    const erase = [1, 1, 1]; // Erase everything
    const add = [5, 10, 15]; // Write new content
    bank.write(writeWeights, erase, add);
    
    const readWeights = [0, 0, 1, 0];
    const result = bank.read(readWeights);
    assert.ok(Math.abs(result[0] - 5) < 0.01);
    assert.ok(Math.abs(result[1] - 10) < 0.01);
    assert.ok(Math.abs(result[2] - 15) < 0.01);
  });

  it('erase without add zeros out the slot', () => {
    const bank = new MemoryBank(4, 3);
    bank.memory[2] = [100, 200, 300];
    bank.write([0, 0, 1, 0], [1, 1, 1], [0, 0, 0]);
    assert.ok(Math.abs(bank.memory[2][0]) < 0.01);
    assert.ok(Math.abs(bank.memory[2][1]) < 0.01);
    assert.ok(Math.abs(bank.memory[2][2]) < 0.01);
  });

  it('partial erase preserves some content', () => {
    const bank = new MemoryBank(4, 3);
    bank.memory[0] = [10, 10, 10];
    bank.write([1, 0, 0, 0], [0.5, 0, 1], [0, 0, 0]);
    // mem[0] *= (1 - w * e) = 10 * (1 - 0.5) = 5 for dim 0
    // dim 1: unchanged (erase=0)
    // dim 2: 10 * (1 - 1) = 0
    assert.ok(Math.abs(bank.memory[0][0] - 5) < 0.01);
    assert.ok(Math.abs(bank.memory[0][1] - 10) < 0.01);
    assert.ok(Math.abs(bank.memory[0][2]) < 0.01);
  });

  it('multiple writes accumulate', () => {
    const bank = new MemoryBank(4, 3);
    bank.write([1, 0, 0, 0], [0, 0, 0], [1, 0, 0]); // Add [1,0,0] to slot 0
    bank.write([1, 0, 0, 0], [0, 0, 0], [0, 1, 0]); // Add [0,1,0] to slot 0
    bank.write([1, 0, 0, 0], [0, 0, 0], [0, 0, 1]); // Add [0,0,1] to slot 0
    // Slot 0 should be initial (0.001*3) + added = ~[1.001, 1.001, 1.001]
    const result = bank.read([1, 0, 0, 0]);
    assert.ok(Math.abs(result[0] - 1.001) < 0.01);
    assert.ok(Math.abs(result[1] - 1.001) < 0.01);
    assert.ok(Math.abs(result[2] - 1.001) < 0.01);
  });
});

describe('Content Addressing Stress', () => {
  it('identical key and memory slot gives highest weight', () => {
    const memory = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]];
    const key = [1, 0, 0];
    const weights = contentAddressing(key, memory, 10);
    
    // Slot 0 should have highest weight
    assert.ok(weights[0] > weights[1]);
    assert.ok(weights[0] > weights[2]);
  });

  it('high beta sharpens attention', () => {
    const memory = [[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0]];
    const key = [1, 0, 0];
    
    const softWeights = contentAddressing(key, memory, 1);
    const sharpWeights = contentAddressing(key, memory, 50);
    
    // Sharp weights should be more focused (higher max)
    assert.ok(Math.max(...sharpWeights) > Math.max(...softWeights));
  });

  it('weights sum to 1', () => {
    const memory = [[1, 2], [3, 4], [5, 6]];
    const key = [2, 3];
    const weights = contentAddressing(key, memory, 5);
    const sum = weights.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1) < 1e-6, `Weights should sum to 1: ${sum}`);
  });

  it('zero key should not crash (all similarities ~0)', () => {
    const memory = [[1, 2], [3, 4], [5, 6]];
    const key = [0, 0];
    const weights = contentAddressing(key, memory);
    assert.ok(weights.every(Number.isFinite));
    assert.ok(Math.abs(weights.reduce((a, b) => a + b, 0) - 1) < 1e-6);
  });
});

describe('Location Addressing', () => {
  it('shift moves attention position', () => {
    const content = [1, 0, 0, 0]; // Attention on slot 0
    const prev = [1, 0, 0, 0];
    
    // [0, 0, 1] shifts attention: slot 0 wraps to slot 3 (circular convolution)
    const shifted = locationAddressing(content, prev, {
      interpolationGate: 1,
      shiftWeights: [0, 0, 1],
      sharpening: 10,
    });
    // After shift, attention should be on a different slot than 0
    assert.ok(shifted[3] > shifted[0], 'Attention should move from slot 0');
  });

  it('interpolation blends content and previous', () => {
    const content = [1, 0, 0, 0];
    const prev = [0, 0, 0, 1];
    
    const halfBlend = locationAddressing(content, prev, {
      interpolationGate: 0.5,
      shiftWeights: [0, 1, 0], // No shift
      sharpening: 1,
    });
    
    // Should be a mix of slots 0 and 3
    assert.ok(halfBlend[0] > 0.1 && halfBlend[0] < 0.9);
    assert.ok(halfBlend[3] > 0.1 && halfBlend[3] < 0.9);
  });

  it('sharpening focuses attention', () => {
    const content = [0.3, 0.25, 0.25, 0.2];
    const prev = content;
    
    const blurry = locationAddressing(content, prev, {
      interpolationGate: 1, shiftWeights: [0, 1, 0], sharpening: 1,
    });
    const sharp = locationAddressing(content, prev, {
      interpolationGate: 1, shiftWeights: [0, 1, 0], sharpening: 10,
    });
    
    assert.ok(Math.max(...sharp) > Math.max(...blurry));
  });
});

describe('NTM Full System Stress', () => {
  it('produces finite output for random input', () => {
    const ntm = new NTM(4, 3, 16, 8, 32);
    const output = ntm.step([1, 0, 0.5, -0.3]);
    assert.equal(output.length, 3);
    assert.ok(output.every(Number.isFinite), 'Output should be finite');
  });

  it('processes sequence without NaN', () => {
    const ntm = new NTM(4, 3, 16, 8, 32);
    const inputs = Array.from({ length: 20 }, () =>
      Array.from({ length: 4 }, () => Math.random() * 2 - 1)
    );
    const outputs = ntm.processSequence(inputs);
    assert.equal(outputs.length, 20);
    for (let t = 0; t < 20; t++) {
      assert.ok(outputs[t].every(Number.isFinite), `Output at t=${t} has NaN/Inf`);
    }
  });

  it('reset clears memory and state', () => {
    const ntm = new NTM(4, 3, 8, 4, 16);
    // Process some inputs
    ntm.step([1, 0, 0, 0]);
    ntm.step([0, 1, 0, 0]);
    
    ntm.reset();
    
    // Memory should be reset to initial values
    for (let i = 0; i < ntm.memorySlots; i++) {
      for (let j = 0; j < ntm.slotSize; j++) {
        assert.ok(Math.abs(ntm.memory.memory[i][j] - 0.001) < 0.01);
      }
    }
    assert.equal(ntm.lastRead.every(v => v === 0), true);
  });

  it('different inputs produce different outputs', () => {
    const ntm = new NTM(4, 3, 8, 4, 16);
    const out1 = ntm.step([1, 0, 0, 0]);
    ntm.reset();
    const out2 = ntm.step([0, 0, 0, 1]);
    
    let diff = 0;
    for (let i = 0; i < out1.length; i++) {
      diff += Math.abs(out1[i] - out2[i]);
    }
    assert.ok(diff > 0.001, 'Different inputs should produce different outputs');
  });

  it('handles long sequences (100 steps) without degradation', () => {
    const ntm = new NTM(4, 3, 32, 8, 32);
    const inputs = Array.from({ length: 100 }, () =>
      Array.from({ length: 4 }, () => Math.random() * 2 - 1)
    );
    const outputs = ntm.processSequence(inputs);
    
    // All outputs should be finite
    for (let t = 0; t < 100; t++) {
      assert.ok(outputs[t].every(Number.isFinite), `Step ${t} has non-finite output`);
    }
    
    // Output magnitude shouldn't explode
    const lastMag = outputs[99].reduce((s, v) => s + v * v, 0);
    assert.ok(lastMag < 1e6, `Output magnitude shouldn't explode: ${lastMag}`);
  });

  it('memory changes after writes', () => {
    const ntm = new NTM(4, 3, 8, 4, 16);
    const initialMem = ntm.memory.memory.map(row => [...row]);
    
    // Process several inputs to trigger writes
    for (let i = 0; i < 10; i++) {
      ntm.step([Math.random(), Math.random(), Math.random(), Math.random()]);
    }
    
    // Memory should have changed
    let totalChange = 0;
    for (let i = 0; i < ntm.memorySlots; i++) {
      for (let j = 0; j < ntm.slotSize; j++) {
        totalChange += Math.abs(ntm.memory.memory[i][j] - initialMem[i][j]);
      }
    }
    assert.ok(totalChange > 0.01, `Memory should change after writes: ${totalChange.toFixed(4)}`);
  });

  it('read head attention is valid probability distribution', () => {
    const ntm = new NTM(4, 3, 8, 4, 16);
    ntm.step([1, 0, 0, 0]);
    
    const weights = ntm.readHead.prevWeights;
    const sum = weights.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1) < 1e-6, `Read attention should sum to 1: ${sum}`);
    assert.ok(weights.every(w => w >= 0), 'Weights should be non-negative');
  });
});

describe('Memory Read/Write Consistency', () => {
  it('write-then-read should preserve content across slots', () => {
    const bank = new MemoryBank(8, 4);
    
    // Write unique patterns to each slot
    for (let slot = 0; slot < 8; slot++) {
      const weights = new Array(8).fill(0);
      weights[slot] = 1;
      bank.write(weights, [1, 1, 1, 1], [slot + 1, slot * 2, slot * 3, slot + 10]);
    }
    
    // Read back each slot
    for (let slot = 0; slot < 8; slot++) {
      const weights = new Array(8).fill(0);
      weights[slot] = 1;
      const read = bank.read(weights);
      assert.ok(Math.abs(read[0] - (slot + 1)) < 0.01, `Slot ${slot} dim 0`);
      assert.ok(Math.abs(read[1] - (slot * 2)) < 0.01, `Slot ${slot} dim 1`);
    }
  });

  it('soft write distributes across slots proportionally', () => {
    const bank = new MemoryBank(4, 2);
    bank.write([0.5, 0.5, 0, 0], [1, 1], [10, 20]);
    
    // Slots 0 and 1 should have half the content
    assert.ok(Math.abs(bank.memory[0][0] - 5) < 0.01);
    assert.ok(Math.abs(bank.memory[1][0] - 5) < 0.01);
    // Slots 2 and 3 should be unchanged
    assert.ok(Math.abs(bank.memory[2][0] - 0.001) < 0.01);
  });
});
