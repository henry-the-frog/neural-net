// ntm.js — Neural Turing Machine
// External memory bank with attention-based read/write heads
// Based on "Neural Turing Machines" (Graves, Wayne, Danihelka, 2014)

// ===== Memory Bank =====
export class MemoryBank {
  constructor(numSlots, slotSize) {
    this.numSlots = numSlots;
    this.slotSize = slotSize;
    // Initialize with small values
    this.memory = Array.from({ length: numSlots }, () =>
      new Array(slotSize).fill(0.001)
    );
  }

  read(weights) {
    // Weighted sum of memory rows
    const result = new Array(this.slotSize).fill(0);
    for (let i = 0; i < this.numSlots; i++) {
      for (let j = 0; j < this.slotSize; j++) {
        result[j] += weights[i] * this.memory[i][j];
      }
    }
    return result;
  }

  write(weights, eraseVector, addVector) {
    for (let i = 0; i < this.numSlots; i++) {
      for (let j = 0; j < this.slotSize; j++) {
        // Erase
        this.memory[i][j] *= (1 - weights[i] * eraseVector[j]);
        // Add
        this.memory[i][j] += weights[i] * addVector[j];
      }
    }
  }

  reset() {
    for (let i = 0; i < this.numSlots; i++) {
      this.memory[i].fill(0.001);
    }
  }
}

// ===== Addressing Mechanisms =====

// Content-based addressing: softmax(β * cosine_similarity(key, memory))
export function contentAddressing(key, memory, beta = 1) {
  const numSlots = memory.length;
  const similarities = new Array(numSlots);

  for (let i = 0; i < numSlots; i++) {
    similarities[i] = beta * cosineSimilarity(key, memory[i]);
  }

  return softmax(similarities);
}

// Location-based addressing: interpolation + shift + sharpening
export function locationAddressing(contentWeights, prevWeights, {
  interpolationGate = 0.5, // g ∈ [0, 1] — blend content and previous
  shiftWeights = [0, 1, 0], // Circular shift distribution
  sharpening = 1, // γ ≥ 1 — sharpen after shift
} = {}) {
  const N = contentWeights.length;

  // 1. Interpolation: g * content + (1-g) * previous
  const interpolated = contentWeights.map((c, i) =>
    interpolationGate * c + (1 - interpolationGate) * prevWeights[i]
  );

  // 2. Circular convolution with shift weights
  const shifted = new Array(N).fill(0);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < shiftWeights.length; j++) {
      const offset = j - Math.floor(shiftWeights.length / 2);
      const idx = ((i + offset) % N + N) % N;
      shifted[i] += interpolated[idx] * shiftWeights[j];
    }
  }

  // 3. Sharpening: w_i^γ / sum(w_j^γ)
  const sharpened = shifted.map(w => Math.pow(Math.max(w, 1e-10), sharpening));
  const sum = sharpened.reduce((a, b) => a + b, 0);
  return sharpened.map(w => w / sum);
}

// ===== Read Head =====
export class ReadHead {
  constructor(slotSize) {
    this.slotSize = slotSize;
    this.prevWeights = null;
    // Learnable parameters for addressing
    this.keyWeights = Array.from({ length: slotSize }, () => (Math.random() - 0.5) * 0.3);
    this.beta = 1;
    this.gate = 0.5;
    this.shiftWeights = [0.1, 0.8, 0.1];
    this.gamma = 1;
  }

  address(memory, controllerOutput) {
    // Generate addressing parameters from controller
    const key = controllerOutput.slice(0, this.slotSize);
    const beta = Math.exp(controllerOutput[this.slotSize] || 0) + 0.1;
    const gate = sigmoid(controllerOutput[this.slotSize + 1] || 0);

    // Content-based
    const contentW = contentAddressing(key, memory, beta);

    // Location-based
    if (!this.prevWeights) {
      this.prevWeights = new Array(memory.length).fill(1 / memory.length);
    }

    const weights = locationAddressing(contentW, this.prevWeights, {
      interpolationGate: gate,
      shiftWeights: this.shiftWeights,
      sharpening: 1 + Math.abs(controllerOutput[this.slotSize + 2] || 0),
    });

    this.prevWeights = weights;
    return weights;
  }
}

// ===== Write Head =====
export class WriteHead {
  constructor(slotSize) {
    this.slotSize = slotSize;
    this.prevWeights = null;
  }

  address(memory, controllerOutput) {
    const key = controllerOutput.slice(0, this.slotSize);
    const beta = Math.exp(controllerOutput[this.slotSize] || 0) + 0.1;
    const gate = sigmoid(controllerOutput[this.slotSize + 1] || 0);

    const contentW = contentAddressing(key, memory, beta);

    if (!this.prevWeights) {
      this.prevWeights = new Array(memory.length).fill(1 / memory.length);
    }

    const weights = locationAddressing(contentW, this.prevWeights, {
      interpolationGate: gate,
      shiftWeights: [0.1, 0.8, 0.1],
      sharpening: 1,
    });

    this.prevWeights = weights;
    return weights;
  }

  generateEraseAdd(controllerOutput) {
    const offset = this.slotSize + 3;
    const erase = Array.from({ length: this.slotSize }, (_, i) =>
      sigmoid(controllerOutput[offset + i] || 0)
    );
    const add = Array.from({ length: this.slotSize }, (_, i) =>
      Math.tanh(controllerOutput[offset + this.slotSize + i] || 0)
    );
    return { erase, add };
  }
}

// ===== Neural Turing Machine =====
export class NTM {
  constructor(inputSize, outputSize, memorySlots = 128, slotSize = 20, controllerSize = 100) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.memorySlots = memorySlots;
    this.slotSize = slotSize;
    this.controllerSize = controllerSize;

    this.memory = new MemoryBank(memorySlots, slotSize);
    this.readHead = new ReadHead(slotSize);
    this.writeHead = new WriteHead(slotSize);

    // Controller: simple feedforward (input + read → hidden → output + head params)
    const controllerInput = inputSize + slotSize; // input + last read
    const headParamSize = slotSize + 3 + slotSize * 2; // key + beta + gate + gamma + erase + add

    this.controllerW1 = Array.from({ length: controllerSize }, () =>
      Array.from({ length: controllerInput }, () => (Math.random() - 0.5) * Math.sqrt(2 / controllerInput))
    );
    this.controllerB1 = new Array(controllerSize).fill(0);

    this.outputW = Array.from({ length: outputSize }, () =>
      Array.from({ length: controllerSize }, () => (Math.random() - 0.5) * Math.sqrt(2 / controllerSize))
    );

    this.headW = Array.from({ length: headParamSize }, () =>
      Array.from({ length: controllerSize }, () => (Math.random() - 0.5) * 0.1)
    );

    this.lastRead = new Array(slotSize).fill(0);
  }

  step(input) {
    // Concatenate input with last read
    const controllerInput = [...input, ...this.lastRead];

    // Controller forward
    const hidden = new Array(this.controllerSize);
    for (let h = 0; h < this.controllerSize; h++) {
      let sum = this.controllerB1[h];
      for (let i = 0; i < controllerInput.length; i++) {
        sum += this.controllerW1[h][i] * controllerInput[i];
      }
      hidden[h] = Math.tanh(sum);
    }

    // Generate output
    const output = new Array(this.outputSize);
    for (let o = 0; o < this.outputSize; o++) {
      let sum = 0;
      for (let h = 0; h < this.controllerSize; h++) {
        sum += this.outputW[o][h] * hidden[h];
      }
      output[o] = sum;
    }

    // Generate head parameters
    const headParams = new Array(this.headW.length);
    for (let p = 0; p < this.headW.length; p++) {
      let sum = 0;
      for (let h = 0; h < this.controllerSize; h++) {
        sum += this.headW[p][h] * hidden[h];
      }
      headParams[p] = sum;
    }

    // Write
    const writeWeights = this.writeHead.address(this.memory.memory, headParams);
    const { erase, add } = this.writeHead.generateEraseAdd(headParams);
    this.memory.write(writeWeights, erase, add);

    // Read
    const readWeights = this.readHead.address(this.memory.memory, headParams);
    this.lastRead = this.memory.read(readWeights);

    return output;
  }

  // Process a sequence
  processSequence(inputs) {
    const outputs = [];
    for (const input of inputs) {
      outputs.push(this.step(input));
    }
    return outputs;
  }

  reset() {
    this.memory.reset();
    this.lastRead = new Array(this.slotSize).fill(0);
    this.readHead.prevWeights = null;
    this.writeHead.prevWeights = null;
  }
}

// ===== Utility Functions =====
function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}
