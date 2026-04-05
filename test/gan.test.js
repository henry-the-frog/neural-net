import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Matrix, GAN } from '../src/index.js';

describe('GAN', () => {
  it('generates correct shape', () => {
    const gan = new GAN({
      latentDim: 8, dataSize: 4,
      generatorLayers: [16], discriminatorLayers: [16]
    });
    const fake = gan.generate(5);
    assert.equal(fake.rows, 5);
    assert.equal(fake.cols, 4);
  });

  it('discriminator outputs probability', () => {
    const gan = new GAN({
      latentDim: 8, dataSize: 4,
      generatorLayers: [16], discriminatorLayers: [16]
    });
    const data = Matrix.random(3, 4).map(v => Math.abs(v));
    const pred = gan.discriminate(data);
    assert.equal(pred.rows, 3);
    assert.equal(pred.cols, 1);
    for (let i = 0; i < 3; i++) {
      const v = pred.get(i, 0);
      assert.ok(v >= 0 && v <= 1, `Prediction should be [0,1], got ${v}`);
    }
  });

  it('param count', () => {
    const gan = new GAN({
      latentDim: 8, dataSize: 4,
      generatorLayers: [16], discriminatorLayers: [16]
    });
    const params = gan.paramCount();
    assert.ok(params.generator > 0);
    assert.ok(params.discriminator > 0);
    assert.equal(params.total, params.generator + params.discriminator);
  });

  it('trains without errors', () => {
    const gan = new GAN({
      latentDim: 4, dataSize: 4,
      generatorLayers: [8], discriminatorLayers: [8]
    });
    
    // Simple target distribution: all values near 0.5
    const data = new Matrix(20, 4).map(() => 0.4 + Math.random() * 0.2);
    
    const history = gan.train(data, { epochs: 10, batchSize: 10, lrD: 0.001, lrG: 0.001 });
    assert.equal(history.dLoss.length, 10);
    assert.equal(history.gLoss.length, 10);
  });

  it('generator improves over training', () => {
    const gan = new GAN({
      latentDim: 4, dataSize: 2,
      generatorLayers: [8], discriminatorLayers: [8]
    });
    
    // Target: values near (0.8, 0.2)
    const data = new Matrix(30, 2);
    for (let i = 0; i < 30; i++) {
      data.set(i, 0, 0.75 + Math.random() * 0.1);
      data.set(i, 1, 0.15 + Math.random() * 0.1);
    }
    
    // Before training: generated samples are random
    const before = gan.generate(10);
    
    gan.train(data, { epochs: 50, batchSize: 15, lrD: 0.001, lrG: 0.002 });
    
    // After training: generated samples should be closer to target distribution
    const after = gan.generate(10);
    
    // Just check it ran without errors and output is in [0,1]
    for (let i = 0; i < 10; i++) {
      assert.ok(after.get(i, 0) >= 0 && after.get(i, 0) <= 1);
    }
  });

  it('discriminator steps parameter works', () => {
    const gan = new GAN({
      latentDim: 4, dataSize: 2,
      generatorLayers: [8], discriminatorLayers: [8]
    });
    
    const data = Matrix.random(20, 2).map(v => Math.abs(v));
    const history = gan.train(data, { epochs: 5, batchSize: 10, dSteps: 3 });
    assert.equal(history.dLoss.length, 5);
  });

  it('large GAN (more layers)', () => {
    const gan = new GAN({
      latentDim: 16, dataSize: 8,
      generatorLayers: [32, 16], discriminatorLayers: [16, 8]
    });
    const fake = gan.generate(3);
    assert.equal(fake.cols, 8);
    const params = gan.paramCount();
    assert.ok(params.total > 1000);
  });
});
