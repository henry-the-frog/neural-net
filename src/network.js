// network.js — Neural network: stack of layers with optimizer support

import { Dense } from './layer.js';
import { getLoss } from './loss.js';
import { Matrix } from './matrix.js';
import { createOptimizer } from './optimizer.js';

export class Network {
  constructor(layers) {
    this.layers = [];
    this.lossFunction = null;
    this._optimizer = null;
    this._optimizerName = 'sgd';
    // Accept optional initial layers array
    if (Array.isArray(layers)) {
      for (const layer of layers) this.add(layer);
    }
  }

  // Add a dense layer (convenience)
  dense(inputSize, outputSize, activation = 'relu') {
    this.layers.push(new Dense(inputSize, outputSize, activation));
    return this;
  }

  // Add any layer (Conv2D, MaxPool2D, Flatten, BatchNorm, Dropout, Dense, etc.)
  add(layer) {
    this.layers.push(layer);
    return this;
  }

  // Set loss function
  loss(name) {
    this.lossFunction = getLoss(name);
    return this;
  }

  // Set optimizer
  optimizer(nameOrObj, options = {}) {
    if (typeof nameOrObj === 'string') {
      this._optimizerName = nameOrObj;
      this._optimizer = createOptimizer(nameOrObj, options);
    } else {
      this._optimizer = nameOrObj;
      this._optimizerName = nameOrObj.name || 'custom';
    }
    return this;
  }

  // Forward pass through all layers
  forward(input) {
    let x = input;
    for (const layer of this.layers) {
      x = layer.forward(x);
    }
    return x;
  }

  // Predict (forward pass, no training)
  predict(input) {
    if (Array.isArray(input)) input = Matrix.fromArray(input);
    if (input.cols === undefined) input = Matrix.fromArray([input]);
    // Set eval mode temporarily
    const modes = this.layers.map(l => l.training);
    for (const l of this.layers) l.training = false;
    const result = this.forward(input);
    this.layers.forEach((l, i) => l.training = modes[i]);
    return result;
  }

  // Train on a batch
  trainBatch(input, target, learningRate = 0.01, momentum = 0, optimizerName = 'sgd') {
    if (Array.isArray(input)) input = Matrix.fromArray(input);
    if (Array.isArray(target)) target = Matrix.fromArray(target);

    // Forward
    const output = this.forward(input);

    // Compute loss
    const loss = this.lossFunction.compute(output, target);

    // Backward
    let grad = this.lossFunction.gradient(output, target);
    for (let i = this.layers.length - 1; i >= 0; i--) {
      grad = this.layers[i].backward(grad);
    }

    // Update weights — use optimizer object if available
    if (this._optimizer && this._optimizer.step) this._optimizer.step();
    
    for (let idx = 0; idx < this.layers.length; idx++) {
      const layer = this.layers[idx];
      if (this._optimizer && layer.dWeights) {
        // Use optimizer classes for Dense layers
        const batchSize = input.rows;
        const gradW = layer.dWeights.mul(1.0 / batchSize);
        const gradB = layer.dBiases.mul(1.0 / batchSize);
        layer.weights = this._optimizer.update(layer.weights, gradW, `L${idx}_w`);
        layer.biases = this._optimizer.update(layer.biases, gradB, `L${idx}_b`);
      } else if (this._optimizer && layer.dFilters) {
        // Conv2D layer — optimizer for filters and biases
        layer.filters = this._optimizer.update(layer.filters, layer.dFilters, `L${idx}_f`);
        layer.biases = this._optimizer.update(layer.biases, layer.dBiases, `L${idx}_b`);
      } else if (layer.update) {
        layer.update(learningRate, momentum, optimizerName);
      }
    }

    return loss;
  }

  // Train for multiple epochs
  train(data, { epochs = 100, learningRate = 0.01, batchSize = 32, momentum = 0, optimizer = 'sgd', verbose = false, onEpoch = null, lrSchedule = null, callbacks = [] } = {}) {
    const { inputs, targets } = data;
    const n = inputs.rows;
    const history = [];

    // Auto-create optimizer if not set and string provided
    if (!this._optimizer && optimizer !== 'sgd') {
      this.optimizer(optimizer, { lr: learningRate });
    }

    // Set training mode
    for (const l of this.layers) l.training = true;

    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0;
      let batches = 0;

      // Learning rate scheduling
      let lr = learningRate;
      if (lrSchedule === 'cosine') {
        lr = learningRate * 0.5 * (1 + Math.cos(Math.PI * epoch / epochs));
      } else if (lrSchedule === 'step') {
        if (epoch > epochs * 0.5) lr *= 0.1;
        if (epoch > epochs * 0.75) lr *= 0.1;
      } else if (lrSchedule === 'linear') {
        lr = learningRate * (1 - epoch / epochs);
      }

      // Shuffle indices
      const indices = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      for (let start = 0; start < n; start += batchSize) {
        const end = Math.min(start + batchSize, n);
        const batchIndices = indices.slice(start, end);

        // Create batch matrices
        const batchInput = new Matrix(batchIndices.length, inputs.cols);
        const batchTarget = new Matrix(batchIndices.length, targets.cols);

        for (let i = 0; i < batchIndices.length; i++) {
          const idx = batchIndices[i];
          for (let j = 0; j < inputs.cols; j++) batchInput.set(i, j, inputs.get(idx, j));
          for (let j = 0; j < targets.cols; j++) batchTarget.set(i, j, targets.get(idx, j));
        }

        epochLoss += this.trainBatch(batchInput, batchTarget, lr, momentum, optimizer);
        batches++;
      }

      epochLoss /= batches;
      history.push(epochLoss);

      if (verbose && (epoch % Math.max(1, Math.floor(epochs / 20)) === 0 || epoch === epochs - 1)) {
        console.log(`Epoch ${epoch + 1}/${epochs} — Loss: ${epochLoss.toFixed(6)}`);
      }

      if (onEpoch) onEpoch(epoch, epochLoss);

      // Run callbacks (e.g., EarlyStopping)
      let shouldStop = false;
      for (const cb of callbacks) {
        if (cb.onEpochEnd && cb.onEpochEnd(epoch, epochLoss, this)) {
          shouldStop = true;
        }
      }
      if (shouldStop) break;
    }

    // Set eval mode (disable dropout)
    for (const l of this.layers) l.training = false;

    return history;
  }

  // Train with gradient accumulation (simulate large batches with small micro-batches)
  trainWithGradientAccumulation(data, {
    epochs = 100,
    microBatchSize = 8,
    accumSteps = 4,
    learningRate = 0.01,
    optimizer = 'adam',
    verbose = false,
    onEpoch = null,
    lrSchedule = null,
    callbacks = []
  } = {}) {
    const { inputs, targets } = data;
    const n = inputs.rows;
    const effectiveBatch = microBatchSize * accumSteps;
    const history = [];

    // Auto-create optimizer
    if (!this._optimizer || this._optimizerName !== optimizer) {
      this.optimizer(optimizer, { lr: learningRate });
    }

    // Set training mode
    for (const l of this.layers) l.training = true;

    for (let epoch = 0; epoch < epochs; epoch++) {
      let epochLoss = 0;
      let totalBatches = 0;

      // Learning rate scheduling
      let lr = learningRate;
      if (lrSchedule === 'cosine') {
        lr = learningRate * 0.5 * (1 + Math.cos(Math.PI * epoch / epochs));
      } else if (lrSchedule === 'step') {
        if (epoch > epochs * 0.5) lr *= 0.1;
        if (epoch > epochs * 0.75) lr *= 0.1;
      } else if (lrSchedule === 'linear') {
        lr = learningRate * (1 - epoch / epochs);
      }

      // Shuffle indices
      const indices = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      // Process in effective batches, each split into micro-batches
      for (let start = 0; start < n; start += effectiveBatch) {
        // Accumulated gradients storage: key → {grad, count}
        const accumGrads = new Map();
        let accumLoss = 0;
        let microCount = 0;

        for (let step = 0; step < accumSteps; step++) {
          const mStart = start + step * microBatchSize;
          const mEnd = Math.min(mStart + microBatchSize, n);
          if (mStart >= n) break;

          const batchIndices = indices.slice(mStart, mEnd);
          const batchInput = new Matrix(batchIndices.length, inputs.cols);
          const batchTarget = new Matrix(batchIndices.length, targets.cols);
          for (let i = 0; i < batchIndices.length; i++) {
            const idx = batchIndices[i];
            for (let j = 0; j < inputs.cols; j++) batchInput.set(i, j, inputs.get(idx, j));
            for (let j = 0; j < targets.cols; j++) batchTarget.set(i, j, targets.get(idx, j));
          }

          // Forward
          const output = this.forward(batchInput);
          const loss = this.lossFunction.compute(output, batchTarget);
          accumLoss += loss;

          // Backward (computes gradients on each layer)
          let grad = this.lossFunction.gradient(output, batchTarget);
          for (let i = this.layers.length - 1; i >= 0; i--) {
            grad = this.layers[i].backward(grad);
          }

          // Accumulate gradients from each layer
          for (let idx = 0; idx < this.layers.length; idx++) {
            const layer = this.layers[idx];
            if (layer.dWeights) {
              const bs = batchIndices.length;
              const gW = layer.dWeights.mul(1.0 / bs);
              const gB = layer.dBiases.mul(1.0 / bs);
              const keyW = `L${idx}_w`;
              const keyB = `L${idx}_b`;
              if (accumGrads.has(keyW)) {
                accumGrads.set(keyW, accumGrads.get(keyW).add(gW));
                accumGrads.set(keyB, accumGrads.get(keyB).add(gB));
              } else {
                accumGrads.set(keyW, gW);
                accumGrads.set(keyB, gB);
              }
            } else if (layer.dFilters) {
              const keyF = `L${idx}_f`;
              const keyB = `L${idx}_b`;
              if (accumGrads.has(keyF)) {
                accumGrads.set(keyF, accumGrads.get(keyF).add(layer.dFilters));
                accumGrads.set(keyB, accumGrads.get(keyB).add(layer.dBiases));
              } else {
                accumGrads.set(keyF, layer.dFilters);
                accumGrads.set(keyB, layer.dBiases);
              }
            }
          }
          microCount++;
        }

        if (microCount === 0) continue;

        // Average accumulated gradients and apply optimizer
        if (this._optimizer && this._optimizer.step) this._optimizer.step();

        for (let idx = 0; idx < this.layers.length; idx++) {
          const layer = this.layers[idx];
          if (layer.dWeights) {
            const keyW = `L${idx}_w`;
            const keyB = `L${idx}_b`;
            if (accumGrads.has(keyW)) {
              const avgGradW = accumGrads.get(keyW).mul(1.0 / microCount);
              const avgGradB = accumGrads.get(keyB).mul(1.0 / microCount);
              layer.weights = this._optimizer.update(layer.weights, avgGradW, keyW);
              layer.biases = this._optimizer.update(layer.biases, avgGradB, keyB);
            }
          } else if (layer.dFilters) {
            const keyF = `L${idx}_f`;
            const keyB = `L${idx}_b`;
            if (accumGrads.has(keyF)) {
              const avgGradF = accumGrads.get(keyF).mul(1.0 / microCount);
              const avgGradB = accumGrads.get(keyB).mul(1.0 / microCount);
              layer.filters = this._optimizer.update(layer.filters, avgGradF, keyF);
              layer.biases = this._optimizer.update(layer.biases, avgGradB, keyB);
            }
          } else if (layer.update) {
            layer.update(lr, 0, optimizer);
          }
        }

        epochLoss += accumLoss / microCount;
        totalBatches++;
      }

      epochLoss /= Math.max(totalBatches, 1);
      history.push(epochLoss);

      if (verbose && (epoch % Math.max(1, Math.floor(epochs / 20)) === 0 || epoch === epochs - 1)) {
        console.log(`Epoch ${epoch + 1}/${epochs} — Loss: ${epochLoss.toFixed(6)} (effective batch: ${effectiveBatch})`);
      }

      if (onEpoch) onEpoch(epoch, epochLoss);

      // Run callbacks
      let shouldStop = false;
      for (const cb of callbacks) {
        if (cb.onEpochEnd && cb.onEpochEnd(epoch, epochLoss, this)) {
          shouldStop = true;
        }
      }
      if (shouldStop) break;
    }

    // Set eval mode
    for (const l of this.layers) l.training = false;

    return history;
  }

  // Evaluate accuracy on test data
  evaluate(inputs, targets) {
    const output = this.forward(inputs);
    const predicted = output.argmax();
    const actual = targets.argmax();

    let correct = 0;
    for (let i = 0; i < predicted.length; i++) {
      if (predicted[i] === actual[i]) correct++;
    }

    return {
      accuracy: correct / predicted.length,
      correct,
      total: predicted.length
    };
  }

  // Serialize to JSON
  toJSON() {
    return {
      layers: this.layers.map((l, i) => {
        const obj = { type: l.constructor.name, index: i };
        if (l.weights) obj.weights = l.weights.toArray();
        if (l.biases) obj.biases = l.biases.toArray();
        if (l.filters) obj.filters = l.filters.toArray();
        // Store config
        if (l.inputSize !== undefined) obj.inputSize = l.inputSize;
        if (l.outputSize !== undefined) obj.outputSize = l.outputSize;
        if (l.activation) obj.activation = l.activation.name;
        if (l.inputH !== undefined) {
          obj.inputH = l.inputH; obj.inputW = l.inputW; obj.inputC = l.inputC;
          obj.numFilters = l.numFilters; obj.filterSize = l.filterSize;
          obj.stride = l.stride; obj.padding = l.padding;
        }
        if (l.poolSize !== undefined) {
          obj.inputH = l.inputH; obj.inputW = l.inputW; obj.inputC = l.inputC;
          obj.poolSize = l.poolSize;
        }
        return obj;
      }),
      loss: this.lossFunction ? 'mse' : null
    };
  }

  // Deserialize from JSON
  static fromJSON(jsonStr) {
    const data = typeof jsonStr === 'string' ? JSON.parse(jsonStr) : jsonStr;
    const net = new Network();

    for (const layerData of data.layers) {
      if (layerData.type === 'Dense') {
        const layer = new Dense(layerData.inputSize, layerData.outputSize, layerData.activation);
        if (layerData.weights) layer.weights = Matrix.fromArray(layerData.weights);
        if (layerData.biases) {
          const b = Matrix.fromArray(layerData.biases);
          // Ensure biases are row vector [1, outputSize]
          layer.biases = b.rows === 1 ? b : new Matrix(1, b.rows).map((_, i, j) => b.get(j, 0));
        }
        net.layers.push(layer);
      }
      // Other layer types can be added here
    }

    if (data.loss) net.loss(data.loss);
    return net;
  }

  // Summary
  summary() {
    let totalParams = 0;
    const lines = ['Network Summary:'];
    lines.push('─'.repeat(60));
    lines.push(`${'Layer'.padEnd(20)} ${'Output'.padEnd(15)} ${'Params'.padEnd(10)} Info`);
    lines.push('─'.repeat(60));

    for (let i = 0; i < this.layers.length; i++) {
      const l = this.layers[i];
      const params = l.paramCount ? l.paramCount() : 0;
      totalParams += params;
      const name = l.constructor.name;
      const output = l.outputSize || '?';
      const info = l.activation ? l.activation.name : '';
      lines.push(`${name} ${i + 1}`.padEnd(20) + `${output}`.padEnd(15) + `${params}`.padEnd(10) + info);
    }

    lines.push('─'.repeat(60));
    lines.push(`Total parameters: ${totalParams}`);
    return lines.join('\n');
  }
}
