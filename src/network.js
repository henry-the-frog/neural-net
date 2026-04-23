// network.js — Neural network: stack of layers with optimizer support

import { Dense } from './layer.js';
import { getLoss } from './loss.js';
import { Matrix } from './matrix.js';
import { createOptimizer } from './optimizer.js';

// Import all layer types for serialization/deserialization
let Conv2D, MaxPool2D, Flatten, BatchNorm, RNN, LSTM, GRU, Dropout, Embedding, KANLayer, MixtureOfExperts;
try { ({ Conv2D, MaxPool2D, Flatten } = await import('./conv.js')); } catch {}
try { ({ BatchNorm } = await import('./batchnorm.js')); } catch {}
try { ({ RNN, LSTM, GRU } = await import('./rnn.js')); } catch {}
try { ({ Dropout } = await import('./dropout.js')); } catch {}
try { ({ Embedding } = await import('./embedding.js')); } catch {}
try { ({ KANLayer } = await import('./kan.js')); } catch {}
try { ({ MixtureOfExperts } = await import('./moe.js')); } catch {}

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
  // Set a default gradient clipping threshold for all training
  clipGradients(maxNorm) {
    this._clipGrad = maxNorm;
  }

  trainBatch(input, target, learningRate = 0.01, momentum = 0, optimizerName = 'sgd', options = {}) {
    if (Array.isArray(input)) input = Matrix.fromArray(input);
    if (Array.isArray(target)) target = Matrix.fromArray(target);

    const clipGrad = options.clipGrad || this._clipGrad || 0;

    // Forward
    const output = this.forward(input);

    // Compute loss
    const loss = this.lossFunction.compute(output, target);

    // Backward
    let grad = this.lossFunction.gradient(output, target);
    for (let i = this.layers.length - 1; i >= 0; i--) {
      grad = this.layers[i].backward(grad);
    }

    // Apply gradient clipping if requested
    if (clipGrad > 0) {
      for (const layer of this.layers) {
        if (layer.dWeights) {
          layer.dWeights = this._clipMatrix(layer.dWeights, clipGrad);
        }
        if (layer.dBiases) {
          layer.dBiases = this._clipMatrix(layer.dBiases, clipGrad);
        }
        if (layer.dFilters) {
          layer.dFilters = this._clipMatrix(layer.dFilters, clipGrad);
        }
      }
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

  // Clip matrix values element-wise to [-maxNorm, maxNorm]
  _clipMatrix(matrix, maxNorm) {
    const clipped = new Float64Array(matrix.data.length);
    for (let i = 0; i < matrix.data.length; i++) {
      clipped[i] = Math.max(-maxNorm, Math.min(maxNorm, matrix.data[i]));
    }
    return new Matrix(matrix.rows, matrix.cols, clipped);
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

  // Serialize to JSON (legacy — delegates to v2 toJSON below)
  // Note: the v2 toJSON method defined later in this class is the canonical one.
  // fromJSON handles both formats for backward compatibility.

  // Deserialize from JSON
  static fromJSON(jsonStr) {
    const data = typeof jsonStr === 'string' ? JSON.parse(jsonStr) : jsonStr;
    const net = new Network();

    for (const d of data.layers) {
      let layer;
      switch (d.type) {
        case 'Dense': {
          layer = new Dense(d.inputSize, d.outputSize, d.activation);
          if (d.weightShape) {
            layer.weights = new Matrix(d.weightShape[0], d.weightShape[1], new Float64Array(d.weights));
            layer.biases = new Matrix(d.biasShape[0], d.biasShape[1], new Float64Array(d.biases));
          } else if (d.weights) {
            layer.weights = Matrix.fromArray(d.weights);
            if (d.biases) {
              const b = Matrix.fromArray(d.biases);
              layer.biases = b.rows === 1 ? b : new Matrix(1, b.rows).map((_, i, j) => b.get(j, 0));
            }
          }
          break;
        }
        case 'Conv2D': {
          if (!Conv2D) throw new Error('Conv2D not available for deserialization');
          layer = new Conv2D(d.inputH, d.inputW, d.channels, d.numFilters, d.filterSize, 
            d.activation || 'relu', { stride: d.stride || 1, padding: d.padding || 0 });
          if (d.filterShape) {
            layer.filters = new Matrix(d.filterShape[0], d.filterShape[1], new Float64Array(d.filters));
          }
          if (d.biasShape) {
            layer.biases = new Matrix(d.biasShape[0], d.biasShape[1], new Float64Array(d.biases));
          }
          break;
        }
        case 'BatchNorm': {
          if (!BatchNorm) throw new Error('BatchNorm not available for deserialization');
          layer = new BatchNorm(d.size);
          layer.gamma = new Matrix(1, d.size, new Float64Array(d.gamma));
          layer.beta = new Matrix(1, d.size, new Float64Array(d.beta));
          layer.runningMean = new Matrix(1, d.size, new Float64Array(d.runningMean));
          layer.runningVar = new Matrix(1, d.size, new Float64Array(d.runningVar));
          layer.training = false; // Loaded models default to inference mode
          break;
        }
        case 'RNN': {
          if (!RNN) throw new Error('RNN not available for deserialization');
          layer = new RNN(d.inputSize, d.hiddenSize, { returnSequences: d.returnSequences });
          layer.Wih = new Matrix(d.inputSize, d.hiddenSize, new Float64Array(d.Wih));
          layer.Whh = new Matrix(d.hiddenSize, d.hiddenSize, new Float64Array(d.Whh));
          layer.bh = new Matrix(1, d.hiddenSize, new Float64Array(d.bh));
          break;
        }
        case 'LSTM': {
          if (!LSTM) throw new Error('LSTM not available for deserialization');
          layer = new LSTM(d.inputSize, d.hiddenSize, { returnSequences: d.returnSequences });
          for (const gate of ['Wi', 'Wf', 'Wc', 'Wo', 'bi', 'bf', 'bc', 'bo']) {
            if (d[gate]) {
              const orig = layer[gate];
              layer[gate] = new Matrix(orig.rows, orig.cols, new Float64Array(d[gate]));
            }
          }
          break;
        }
        case 'GRU': {
          if (!GRU) throw new Error('GRU not available for deserialization');
          layer = new GRU(d.inputSize, d.hiddenSize, { returnSequences: d.returnSequences });
          for (const name of ['Wz', 'Wr', 'Wh', 'bz', 'br', 'bh']) {
            if (d[name] && d[name + 'Shape']) {
              layer[name] = new Matrix(d[name + 'Shape'][0], d[name + 'Shape'][1], new Float64Array(d[name]));
            }
          }
          break;
        }
        case 'Dropout': {
          if (!Dropout) throw new Error('Dropout not available for deserialization');
          layer = new Dropout(d.rate);
          layer.training = false; // Loaded models default to inference mode
          break;
        }
        case 'Embedding': {
          if (!Embedding) throw new Error('Embedding not available for deserialization');
          layer = new Embedding(d.vocabSize, d.embedDim);
          if (d.weightShape) {
            layer.weights = new Matrix(d.weightShape[0], d.weightShape[1], new Float64Array(d.weights));
          }
          break;
        }
        case 'Flatten': {
          if (!Flatten) throw new Error('Flatten not available for deserialization');
          layer = new Flatten();
          if (d.inputH) layer.inputH = d.inputH;
          if (d.inputW) layer.inputW = d.inputW;
          if (d.channels || d.inputC) layer.inputC = d.channels || d.inputC;
          if (d.inputSize) layer.inputSize = d.inputSize;
          if (d.outputSize) layer.outputSize = d.outputSize;
          break;
        }
        case 'MaxPool2D': {
          if (!MaxPool2D) throw new Error('MaxPool2D not available for deserialization');
          layer = new MaxPool2D(d.inputH, d.inputW, d.channels || d.inputC, d.poolSize || 2);
          break;
        }
        case 'KANLayer': {
          if (!KANLayer) throw new Error('KANLayer not available for deserialization');
          layer = new KANLayer(d.inputSize, d.outputSize, d.numBasis, d.splineOrder, d.gridRange);
          // Restore coefficients
          if (d.coeffs) {
            layer.coeffs = d.coeffs.map(row => row.map(col => Array.from(col)));
          }
          if (d.residualWeights) {
            layer.residualWeights = d.residualWeights.map(row => Array.from(row));
          }
          break;
        }
        case 'MixtureOfExperts': {
          if (!MixtureOfExperts) throw new Error('MixtureOfExperts not available for deserialization');
          layer = new MixtureOfExperts(d.dModel || d.inputSize, d.numExperts, d.dHidden || 16, d.outputSize || d.dModel || d.inputSize, d.topK);
          // Restore router weights
          if (d.routerWeightShape && layer.routerW) {
            layer.routerW = new Matrix(d.routerWeightShape[0], d.routerWeightShape[1], new Float64Array(d.routerWeights));
            layer.routerB = new Matrix(d.routerBiasShape[0], d.routerBiasShape[1], new Float64Array(d.routerBiases));
          }
          // Restore expert weights (SwiGLU FFNs with W1, W2, W3)
          if (d.experts && layer.experts) {
            for (let i = 0; i < Math.min(d.experts.length, layer.experts.length); i++) {
              const ed = d.experts[i];
              const expert = layer.experts[i];
              if (ed.W1) {
                expert.W1 = new Matrix(ed.W1.shape[0], ed.W1.shape[1], new Float64Array(ed.W1.data));
              }
              if (ed.b1) {
                expert.b1 = new Matrix(ed.b1.shape[0], ed.b1.shape[1], new Float64Array(ed.b1.data));
              }
              if (ed.W2) {
                expert.W2 = new Matrix(ed.W2.shape[0], ed.W2.shape[1], new Float64Array(ed.W2.data));
              }
              if (ed.b2) {
                expert.b2 = new Matrix(ed.b2.shape[0], ed.b2.shape[1], new Float64Array(ed.b2.data));
              }
              // Legacy: W3 from SwiGLU experts (ignored for new ExpertFFN)
              if (ed.W3 && expert.W3) {
                expert.W3 = new Matrix(ed.W3.shape[0], ed.W3.shape[1], new Float64Array(ed.W3.data));
              }
            }
          }
          break;
        }
        default:
          throw new Error(`Unknown layer type for deserialization: ${d.type}`);
      }
      if (layer) net.layers.push(layer);
    }

    if (data.loss) net.loss(data.loss);
    if (data.optimizer) {
      try { net.optimizer(data.optimizer); } catch { /* optional */ }
    }
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

  // Serialize network to JSON-compatible object
  toJSON() {
    const layers = this.layers.map(layer => {
      const info = {
        type: layer.constructor.name,
      };

      if (layer instanceof Dense) {
        info.inputSize = layer.inputSize;
        info.outputSize = layer.outputSize;
        info.activation = layer.activation.name;
        info.weights = Array.from(layer.weights.data);
        info.weightShape = [layer.weights.rows, layer.weights.cols];
        info.biases = Array.from(layer.biases.data);
        info.biasShape = [layer.biases.rows, layer.biases.cols];
      } else if (layer.constructor.name === 'Conv2D') {
        info.inputH = layer.inputH;
        info.inputW = layer.inputW;
        info.channels = layer.inputC;
        info.numFilters = layer.numFilters;
        info.filterSize = layer.filterSize;
        info.activation = layer.activation?.name || 'linear';
        info.stride = layer.stride;
        info.padding = layer.padding;
        info.filters = Array.from(layer.filters.data);
        info.filterShape = [layer.filters.rows, layer.filters.cols];
        info.biases = Array.from(layer.biases.data);
        info.biasShape = [layer.biases.rows, layer.biases.cols];
      } else if (layer.constructor.name === 'BatchNorm') {
        info.size = layer.size;
        info.gamma = Array.from(layer.gamma.data);
        info.beta = Array.from(layer.beta.data);
        info.runningMean = Array.from(layer.runningMean.data);
        info.runningVar = Array.from(layer.runningVar.data);
      } else if (layer.constructor.name === 'RNN') {
        info.inputSize = layer.inputSize;
        info.hiddenSize = layer.hiddenSize;
        info.Wih = Array.from(layer.Wih.data);
        info.Whh = Array.from(layer.Whh.data);
        info.bh = Array.from(layer.bh.data);
      } else if (layer.constructor.name === 'LSTM') {
        info.inputSize = layer.inputSize;
        info.hiddenSize = layer.hiddenSize;
        info.Wi = Array.from(layer.Wi.data);
        info.Wf = Array.from(layer.Wf.data);
        info.Wc = Array.from(layer.Wc.data);
        info.Wo = Array.from(layer.Wo.data);
        info.bi = Array.from(layer.bi.data);
        info.bf = Array.from(layer.bf.data);
        info.bc = Array.from(layer.bc.data);
        info.bo = Array.from(layer.bo.data);
      } else if (layer.constructor.name === 'Flatten' || layer.constructor.name === 'MaxPool2D') {
        // These are stateless — just need constructor args
        if (layer.inputH) info.inputH = layer.inputH;
        if (layer.inputW) info.inputW = layer.inputW;
        if (layer.channels) info.channels = layer.channels;
        if (layer.inputC) info.inputC = layer.inputC;
        if (layer.poolSize) info.poolSize = layer.poolSize;
      } else if (layer.constructor.name === 'GRU') {
        info.inputSize = layer.inputSize;
        info.hiddenSize = layer.hiddenSize;
        info.returnSequences = layer.returnSequences;
        info.Wz = Array.from(layer.Wz.data); info.WzShape = [layer.Wz.rows, layer.Wz.cols];
        info.Wr = Array.from(layer.Wr.data); info.WrShape = [layer.Wr.rows, layer.Wr.cols];
        info.Wh = Array.from(layer.Wh.data); info.WhShape = [layer.Wh.rows, layer.Wh.cols];
        info.bz = Array.from(layer.bz.data); info.bzShape = [layer.bz.rows, layer.bz.cols];
        info.br = Array.from(layer.br.data); info.brShape = [layer.br.rows, layer.br.cols];
        info.bh = Array.from(layer.bh.data); info.bhShape = [layer.bh.rows, layer.bh.cols];
      } else if (layer.constructor.name === 'Dropout') {
        info.rate = layer.rate;
      } else if (layer.constructor.name === 'Embedding') {
        info.vocabSize = layer.vocabSize;
        info.embedDim = layer.embedDim;
        info.weights = Array.from(layer.weights.data);
        info.weightShape = [layer.weights.rows, layer.weights.cols];
      } else if (layer.constructor.name === 'KANLayer') {
        info.inputSize = layer.inputSize;
        info.outputSize = layer.outputSize;
        info.numBasis = layer.numBasis;
        info.splineOrder = layer.splineOrder;
        info.gridRange = layer.gridRange;
        // Serialize coefficients (nested array [inputSize][outputSize][numBasis])
        info.coeffs = layer.coeffs.map(row => row.map(col => Array.from(col)));
        // Serialize residual weights (nested array [inputSize][outputSize])
        info.residualWeights = layer.residualWeights.map(row => Array.from(row));
      } else if (layer.constructor.name === 'MixtureOfExperts') {
        info.dModel = layer.inputSize;
        info.inputSize = layer.inputSize;
        info.numExperts = layer.numExperts || layer.experts?.length;
        info.topK = layer.topK;
        info.dHidden = layer.dHidden || layer.experts?.[0]?.dHidden;
        info.outputSize = layer.outputSize;
        // Serialize router weights
        if (layer.routerW) {
          info.routerWeights = Array.from(layer.routerW.data);
          info.routerWeightShape = [layer.routerW.rows, layer.routerW.cols];
          info.routerBiases = Array.from(layer.routerB.data);
          info.routerBiasShape = [layer.routerB.rows, layer.routerB.cols];
        }
        // Serialize expert networks (Dense FFNs with W1, b1, W2, b2)
        if (layer.experts) {
          info.experts = layer.experts.map(expert => ({
            W1: {
              data: Array.from(expert.W1.data),
              shape: [expert.W1.rows, expert.W1.cols],
            },
            b1: {
              data: Array.from(expert.b1.data),
              shape: [expert.b1.rows, expert.b1.cols],
            },
            W2: {
              data: Array.from(expert.W2.data),
              shape: [expert.W2.rows, expert.W2.cols],
            },
            b2: {
              data: Array.from(expert.b2.data),
              shape: [expert.b2.rows, expert.b2.cols],
            },
            inputSize: expert.inputSize,
            dHidden: expert.dHidden,
            outputSize: expert.outputSize,
          }));
        }
      }

      return info;
    });

    return {
      version: 1,
      loss: this.lossFunction?.name || null,
      optimizer: this._optimizerName,
      layers,
    };
  }

  // Serialize to JSON string
  save() {
    return JSON.stringify(this.toJSON());
  }

  // Load network from JSON string or object
  static load(jsonOrString) {
    // Delegate to the comprehensive fromJSON method
    return Network.fromJSON(jsonOrString);
  }
}
