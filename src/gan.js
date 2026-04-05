// gan.js — Generative Adversarial Network (Goodfellow et al. 2014)

import { Matrix } from './matrix.js';
import { Dense } from './layer.js';
import { getLoss } from './loss.js';

/**
 * GAN — Generator vs Discriminator adversarial training
 * Generator: noise → fake data
 * Discriminator: data → real/fake probability
 */
export class GAN {
  constructor({
    latentDim = 16,
    dataSize,
    generatorLayers = [],
    discriminatorLayers = [],
  }) {
    this.latentDim = latentDim;
    this.dataSize = dataSize;
    
    // Build generator: latentDim → hidden → dataSize
    this.generator = [];
    let prevSize = latentDim;
    for (const h of generatorLayers) {
      this.generator.push(new Dense(prevSize, h, 'relu'));
      prevSize = h;
    }
    this.generator.push(new Dense(prevSize, dataSize, 'sigmoid'));
    
    // Build discriminator: dataSize → hidden → 1
    this.discriminator = [];
    prevSize = dataSize;
    for (const h of discriminatorLayers) {
      this.discriminator.push(new Dense(prevSize, h, 'relu'));
      prevSize = h;
    }
    this.discriminator.push(new Dense(prevSize, 1, 'sigmoid'));
    
    this.loss = getLoss('mse'); // Binary cross-entropy approximated by MSE
  }
  
  // Forward through generator
  generateFake(batchSize) {
    let x = Matrix.random(batchSize, this.latentDim);
    for (const layer of this.generator) x = layer.forward(x);
    return x;
  }
  
  // Forward through discriminator
  discriminate(data) {
    let x = data;
    for (const layer of this.discriminator) x = layer.forward(x);
    return x;
  }
  
  // Train discriminator on real and fake data
  trainDiscriminator(realData, learningRate = 0.001) {
    const batchSize = realData.rows;
    
    // Real data → should output 1
    const realPred = this.discriminate(realData);
    const realTargets = new Matrix(batchSize, 1).map(() => 0.9); // Label smoothing
    const realLoss = this.loss.compute(realPred, realTargets);
    
    let grad = this.loss.gradient(realPred, realTargets);
    for (let i = this.discriminator.length - 1; i >= 0; i--) {
      grad = this.discriminator[i].backward(grad);
    }
    for (const layer of this.discriminator) layer.update(learningRate, 0, 'sgd');
    
    // Fake data → should output 0
    const fakeData = this.generateFake(batchSize);
    const fakePred = this.discriminate(fakeData);
    const fakeTargets = Matrix.zeros(batchSize, 1);
    const fakeLoss = this.loss.compute(fakePred, fakeTargets);
    
    grad = this.loss.gradient(fakePred, fakeTargets);
    for (let i = this.discriminator.length - 1; i >= 0; i--) {
      grad = this.discriminator[i].backward(grad);
    }
    for (const layer of this.discriminator) layer.update(learningRate, 0, 'sgd');
    
    return { realLoss, fakeLoss, dLoss: (realLoss + fakeLoss) / 2 };
  }
  
  // Train generator to fool discriminator
  trainGenerator(batchSize, learningRate = 0.001) {
    // Generate fake data
    let x = Matrix.random(batchSize, this.latentDim);
    for (const layer of this.generator) x = layer.forward(x);
    
    // Pass through discriminator
    let pred = x;
    for (const layer of this.discriminator) pred = layer.forward(pred);
    
    // Generator wants discriminator to output 1 (think it's real)
    const targets = new Matrix(batchSize, 1).map(() => 1);
    const gLoss = this.loss.compute(pred, targets);
    
    // Backward through discriminator (frozen) then generator
    let grad = this.loss.gradient(pred, targets);
    for (let i = this.discriminator.length - 1; i >= 0; i--) {
      grad = this.discriminator[i].backward(grad);
    }
    // Continue backward through generator
    for (let i = this.generator.length - 1; i >= 0; i--) {
      grad = this.generator[i].backward(grad);
    }
    // Only update generator weights
    for (const layer of this.generator) layer.update(learningRate, 0, 'sgd');
    
    return gLoss;
  }
  
  // Full training loop
  train(data, { epochs = 100, batchSize = 32, lrD = 0.001, lrG = 0.001, dSteps = 1, verbose = false } = {}) {
    const n = data.rows;
    const history = { dLoss: [], gLoss: [] };
    
    for (const l of [...this.generator, ...this.discriminator]) l.training = true;
    
    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochDLoss = 0, epochGLoss = 0, batches = 0;
      
      const indices = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }
      
      for (let start = 0; start < n; start += batchSize) {
        const end = Math.min(start + batchSize, n);
        const batch = new Matrix(end - start, data.cols);
        for (let i = start; i < end; i++) {
          const idx = indices[i];
          for (let j = 0; j < data.cols; j++) batch.set(i - start, j, data.get(idx, j));
        }
        
        // Train discriminator
        let dLoss;
        for (let k = 0; k < dSteps; k++) {
          dLoss = this.trainDiscriminator(batch, lrD);
        }
        epochDLoss += dLoss.dLoss;
        
        // Train generator
        const gLoss = this.trainGenerator(batch.rows, lrG);
        epochGLoss += gLoss;
        batches++;
      }
      
      history.dLoss.push(epochDLoss / batches);
      history.gLoss.push(epochGLoss / batches);
      
      if (verbose && epoch % Math.max(1, Math.floor(epochs / 10)) === 0) {
        console.log(`Epoch ${epoch + 1}/${epochs} — D: ${(epochDLoss/batches).toFixed(4)} G: ${(epochGLoss/batches).toFixed(4)}`);
      }
    }
    
    for (const l of [...this.generator, ...this.discriminator]) l.training = false;
    return history;
  }
  
  // Generate samples
  generate(numSamples = 1) {
    return this.generateFake(numSamples);
  }
  
  paramCount() {
    const gParams = this.generator.reduce((s, l) => s + l.paramCount(), 0);
    const dParams = this.discriminator.reduce((s, l) => s + l.paramCount(), 0);
    return { generator: gParams, discriminator: dParams, total: gParams + dParams };
  }
}
