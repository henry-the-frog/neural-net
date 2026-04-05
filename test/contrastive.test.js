import { describe, it } from 'node:test';
import { strict as assert } from 'node:assert';
import { cosineSimilarity, ntXentLoss, augment, ContrastiveLearner } from '../src/contrastive.js';

describe('Contrastive Learning', () => {
  describe('cosineSimilarity', () => {
    it('should be 1 for identical vectors', () => {
      assert.ok(Math.abs(cosineSimilarity([1, 2, 3], [1, 2, 3]) - 1) < 0.01);
    });

    it('should be -1 for opposite vectors', () => {
      assert.ok(Math.abs(cosineSimilarity([1, 0], [-1, 0]) + 1) < 0.01);
    });

    it('should be 0 for orthogonal vectors', () => {
      assert.ok(Math.abs(cosineSimilarity([1, 0], [0, 1])) < 0.01);
    });
  });

  describe('ntXentLoss', () => {
    it('should compute loss for simple pairs', () => {
      const embeddings = [
        [1, 0],   // sample 0, view 1
        [0, 1],   // sample 1, view 1
        [0.9, 0.1], // sample 0, view 2
        [0.1, 0.9], // sample 1, view 2
      ];
      const loss = ntXentLoss(embeddings, 0.5);
      assert.ok(typeof loss === 'number');
      assert.ok(!isNaN(loss));
      assert.ok(loss >= 0);
    });

    it('should give lower loss for more similar positive pairs', () => {
      // Perfect alignment
      const aligned = [[1,0],[0,1],[1,0],[0,1]];
      // Misaligned
      const misaligned = [[1,0],[0,1],[0,1],[1,0]];
      
      const lossAligned = ntXentLoss(aligned, 0.5);
      const lossMisaligned = ntXentLoss(misaligned, 0.5);
      assert.ok(lossAligned < lossMisaligned,
        `Aligned loss (${lossAligned.toFixed(3)}) should be < misaligned (${lossMisaligned.toFixed(3)})`);
    });
  });

  describe('augment', () => {
    it('should produce different views', () => {
      const data = [0.5, 0.5, 0.5, 0.5];
      const aug1 = augment(data);
      const aug2 = augment(data);
      // At least one element should differ
      const differs = aug1.some((v, i) => v !== aug2[i]);
      assert.ok(differs, 'Augmented views should differ');
    });

    it('should preserve approximate magnitude', () => {
      const data = [1, 0, 1, 0];
      const aug = augment(data, { noiseScale: 0.01, dropRate: 0 });
      for (let i = 0; i < data.length; i++) {
        assert.ok(Math.abs(aug[i] - data[i]) < 0.5, `Augmented value too different`);
      }
    });
  });

  describe('ContrastiveLearner', () => {
    it('should create with correct dimensions', () => {
      const cl = new ContrastiveLearner(10, 8);
      assert.equal(cl.inputDim, 10);
      assert.equal(cl.embedDim, 8);
    });

    it('should encode inputs', () => {
      const cl = new ContrastiveLearner(4, 3);
      const emb = cl.encode([0.5, 0.3, 0.8, 0.1]);
      assert.equal(emb.length, 3);
      assert.ok(!emb.some(isNaN));
    });

    it('should compute similarity', () => {
      const cl = new ContrastiveLearner(4, 8);
      const sim = cl.similarity([1, 0, 0, 0], [0, 0, 0, 1]);
      assert.ok(sim >= -1 && sim <= 1);
    });

    it('should train on data', () => {
      const cl = new ContrastiveLearner(4, 8, {
        hiddenDim: 16,
        projDim: 8,
        learningRate: 0.01,
      });
      const data = Array.from({ length: 20 }, () =>
        [Math.random(), Math.random(), Math.random(), Math.random()]
      );

      const { history } = cl.train(data, { epochs: 5, batchSize: 8 });
      assert.equal(history.length, 5);
      assert.ok(!history.some(isNaN));
    });

    it('should use onEpoch callback', () => {
      const cl = new ContrastiveLearner(4, 4);
      const calls = [];
      cl.train([[0.5, 0.5, 0.5, 0.5]], {
        epochs: 3,
        batchSize: 1,
        onEpoch: d => calls.push(d),
      });
      assert.equal(calls.length, 3);
    });
  });
});
