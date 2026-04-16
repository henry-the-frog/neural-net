/**
 * char-lm.js — Character-level Language Model
 * 
 * Demonstrates the neural-net library's LSTM + Dense layers
 * working end-to-end for a real task: predicting the next character
 * in a text sequence.
 * 
 * Usage:
 *   node examples/char-lm.js
 *   node examples/char-lm.js --generate "The "
 */

import { LSTM } from '../src/rnn.js';
import { Dense } from '../src/layer.js';
import { Matrix } from '../src/matrix.js';

// ============================================================
// Character encoding
// ============================================================

class CharEncoder {
  constructor(text) {
    this.chars = [...new Set(text)].sort();
    this.charToIdx = new Map();
    this.idxToChar = new Map();
    for (let i = 0; i < this.chars.length; i++) {
      this.charToIdx.set(this.chars[i], i);
      this.idxToChar.set(i, this.chars[i]);
    }
    this.vocabSize = this.chars.length;
  }
  
  encode(char) {
    return this.charToIdx.get(char) ?? 0;
  }
  
  decode(idx) {
    return this.idxToChar.get(idx) ?? '?';
  }
  
  oneHot(char) {
    const vec = new Array(this.vocabSize).fill(0);
    vec[this.encode(char)] = 1;
    return vec;
  }
}

// ============================================================
// Training data preparation
// ============================================================

function prepareData(text, encoder, seqLen) {
  const sequences = [];
  const targets = [];
  
  for (let i = 0; i < text.length - seqLen; i++) {
    const input = [];
    for (let j = 0; j < seqLen; j++) {
      input.push(...encoder.oneHot(text[i + j]));
    }
    sequences.push(input);
    targets.push(encoder.encode(text[i + seqLen]));
  }
  
  return { sequences, targets };
}

// ============================================================
// Softmax
// ============================================================

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

function crossEntropyLoss(probs, targetIdx) {
  return -Math.log(Math.max(probs[targetIdx], 1e-10));
}

// ============================================================
// Model
// ============================================================

class CharLM {
  constructor(vocabSize, hiddenSize, seqLen) {
    this.vocabSize = vocabSize;
    this.hiddenSize = hiddenSize;
    this.seqLen = seqLen;
    
    // LSTM takes one-hot encoded characters (vocabSize per step)
    this.lstm = new LSTM(vocabSize, hiddenSize, seqLen);
    // Output projection: hidden → vocab logits
    this.output = new Dense(hiddenSize, vocabSize, 'linear');
  }
  
  forward(input) {
    // input: [batch, seqLen * vocabSize]
    const hidden = this.lstm.forward(input);
    const logits = this.output.forward(hidden);
    return logits;
  }
  
  backward(dLogits) {
    const dHidden = this.output.backward(dLogits);
    this.lstm.backward(dHidden);
  }
  
  update(lr) {
    this.lstm.update(lr);
    this.output.update(lr, 0, 'sgd');
  }
  
  train(text, epochs = 100, lr = 0.01, batchSize = 16) {
    const encoder = new CharEncoder(text);
    const { sequences, targets } = prepareData(text, encoder, this.seqLen);
    
    console.log(`Vocab: ${encoder.vocabSize} chars, ${sequences.length} training examples`);
    
    const history = [];
    
    for (let epoch = 0; epoch < epochs; epoch++) {
      let totalLoss = 0;
      let numBatches = 0;
      
      // Mini-batch training
      for (let i = 0; i < sequences.length; i += batchSize) {
        const end = Math.min(i + batchSize, sequences.length);
        const batchSeqs = sequences.slice(i, end);
        const batchTargets = targets.slice(i, end);
        const bs = batchSeqs.length;
        
        // Create input matrix
        const input = new Matrix(bs, this.seqLen * this.vocabSize);
        for (let b = 0; b < bs; b++) {
          for (let j = 0; j < batchSeqs[b].length; j++) {
            input.set(b, j, batchSeqs[b][j]);
          }
        }
        
        // Forward
        const logits = this.forward(input);
        
        // Compute loss and gradients
        let batchLoss = 0;
        const dLogits = new Matrix(bs, this.vocabSize);
        
        for (let b = 0; b < bs; b++) {
          const logitRow = [];
          for (let v = 0; v < this.vocabSize; v++) {
            logitRow.push(logits.get(b, v));
          }
          const probs = softmax(logitRow);
          batchLoss += crossEntropyLoss(probs, batchTargets[b]);
          
          // Cross-entropy gradient: probs - one_hot(target)
          for (let v = 0; v < this.vocabSize; v++) {
            const grad = probs[v] - (v === batchTargets[b] ? 1 : 0);
            dLogits.set(b, v, grad / bs);
          }
        }
        
        // Backward + update
        this.backward(dLogits);
        this.update(lr);
        
        totalLoss += batchLoss;
        numBatches++;
      }
      
      const avgLoss = totalLoss / sequences.length;
      history.push(avgLoss);
      
      if (epoch % 10 === 0 || epoch === epochs - 1) {
        console.log(`Epoch ${epoch}: loss = ${avgLoss.toFixed(4)}`);
      }
    }
    
    return { encoder, history };
  }
  
  generate(encoder, seed, length = 50, temperature = 0.8) {
    let text = seed;
    
    for (let i = 0; i < length; i++) {
      // Take last seqLen characters
      const context = text.slice(-this.seqLen);
      if (context.length < this.seqLen) break;
      
      // Encode
      const input = new Matrix(1, this.seqLen * this.vocabSize);
      for (let j = 0; j < this.seqLen; j++) {
        const oneHot = encoder.oneHot(context[j]);
        for (let v = 0; v < this.vocabSize; v++) {
          input.set(0, j * this.vocabSize + v, oneHot[v]);
        }
      }
      
      // Forward
      const logits = this.forward(input);
      
      // Temperature-scaled sampling
      const logitRow = [];
      for (let v = 0; v < this.vocabSize; v++) {
        logitRow.push(logits.get(0, v) / temperature);
      }
      const probs = softmax(logitRow);
      
      // Sample from distribution
      const r = Math.random();
      let cumulative = 0;
      let chosen = 0;
      for (let v = 0; v < this.vocabSize; v++) {
        cumulative += probs[v];
        if (r < cumulative) { chosen = v; break; }
      }
      
      text += encoder.decode(chosen);
    }
    
    return text;
  }
}

// ============================================================
// Demo
// ============================================================

const corpus = `The quick brown fox jumps over the lazy dog. The dog barked at the fox. The fox ran away quickly. A lazy dog sleeps all day. Quick foxes are clever animals. Brown dogs chase the fox. The lazy fox sleeps under the tree.`;

console.log('=== Character-Level Language Model ===');
console.log(`Corpus: ${corpus.length} chars`);

const seqLen = 8;
const hiddenSize = 32;
const model = new CharLM(0, hiddenSize, seqLen);

// Pre-count vocab
const encoder = new CharEncoder(corpus);
const realModel = new CharLM(encoder.vocabSize, hiddenSize, seqLen);

const { encoder: enc, history } = realModel.train(corpus, 50, 0.05, 8);

console.log('\n=== Generated Text ===');
const generated = realModel.generate(enc, 'The quic', 40);
console.log(generated);

const generated2 = realModel.generate(enc, 'A lazy d', 40);
console.log(generated2);

// Test
const firstLoss = history[0];
const lastLoss = history[history.length - 1];
console.log(`\nLoss: ${firstLoss.toFixed(4)} → ${lastLoss.toFixed(4)} (${((1 - lastLoss/firstLoss) * 100).toFixed(1)}% reduction)`);

if (lastLoss < firstLoss) {
  console.log('✓ Training converged!');
} else {
  console.log('✗ Training did not converge');
  process.exit(1);
}

export { CharLM, CharEncoder };
