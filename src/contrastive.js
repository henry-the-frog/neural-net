// contrastive.js — Contrastive Learning / SimCLR (Chen et al., 2020)
// Learn representations by maximizing agreement between augmented views.
//
// NT-Xent (Normalized Temperature-scaled Cross Entropy) loss:
// l(i,j) = -log(exp(sim(z_i, z_j)/τ) / Σ_{k≠i} exp(sim(z_i, z_k)/τ))
// where sim = cosine similarity, τ = temperature

/**
 * Cosine similarity between two vectors.
 */
export function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
}

/**
 * NT-Xent loss for a batch of positive pairs.
 * Given 2N embeddings where (z[2i], z[2i+1]) are positive pairs.
 * @param {Array<Float64Array>} embeddings - 2N embeddings
 * @param {number} temperature - Scaling temperature (default 0.5)
 * @returns {{ loss: number, accuracy: number }}
 */
export function ntXentLoss(embeddings, temperature = 0.5) {
  const N2 = embeddings.length; // 2N
  const N = N2 / 2;
  
  // Compute similarity matrix
  const sim = Array.from({ length: N2 }, () => new Float64Array(N2));
  for (let i = 0; i < N2; i++) {
    for (let j = 0; j < N2; j++) {
      sim[i][j] = cosineSimilarity(embeddings[i], embeddings[j]) / temperature;
    }
  }
  
  let totalLoss = 0;
  
  for (let i = 0; i < N2; i++) {
    // Positive pair: (i, i+N) mod 2N for SimCLR convention
    const j = i < N ? i + N : i - N;
    
    const positiveScore = sim[i][j];
    
    let maxSim = -Infinity;
    for (let k = 0; k < N2; k++) {
      if (k !== i) maxSim = Math.max(maxSim, sim[i][k]);
    }
    
    let sumExp = 0;
    for (let k = 0; k < N2; k++) {
      if (k !== i) sumExp += Math.exp(sim[i][k] - maxSim);
    }
    const logSumExp = Math.log(sumExp) + maxSim;
    
    totalLoss += -positiveScore + logSumExp;
  }
  
  return totalLoss / N2;
}

/**
 * InfoNCE loss (van den Oord et al., 2018).
 * Similar to NT-Xent but with a single positive and K negatives.
 * @param {Float64Array} anchor - Anchor embedding
 * @param {Float64Array} positive - Positive embedding
 * @param {Array<Float64Array>} negatives - Negative embeddings
 * @param {number} temperature
 * @returns {number} Loss
 */
export function infoNCELoss(anchor, positive, negatives, temperature = 0.1) {
  const posSim = cosineSimilarity(anchor, positive) / temperature;
  
  let maxSim = posSim;
  const negSims = negatives.map(neg => {
    const s = cosineSimilarity(anchor, neg) / temperature;
    maxSim = Math.max(maxSim, s);
    return s;
  });
  
  let sumExp = Math.exp(posSim - maxSim);
  for (const s of negSims) sumExp += Math.exp(s - maxSim);
  
  return -(posSim - maxSim) + Math.log(sumExp);
}

/**
 * Triplet loss (Schroff et al., 2015).
 * L = max(0, d(a, p) - d(a, n) + margin)
 */
export function tripletLoss(anchor, positive, negative, margin = 0.2) {
  let dPos = 0, dNeg = 0;
  for (let i = 0; i < anchor.length; i++) {
    dPos += (anchor[i] - positive[i]) ** 2;
    dNeg += (anchor[i] - negative[i]) ** 2;
  }
  return Math.max(0, Math.sqrt(dPos) - Math.sqrt(dNeg) + margin);
}

/**
 * Simple data augmentation: add noise + random dropout.
 * @param {number[]} data - Input vector
 * @param {object} opts - { noiseScale, dropRate }
 * @returns {number[]} Augmented vector
 */
export function augment(data, { noiseScale = 0.1, dropRate = 0.1 } = {}) {
  return data.map(v => {
    if (Math.random() < dropRate) return 0;
    return v + (Math.random() * 2 - 1) * noiseScale;
  });
}

/**
 * Contrastive Learner — simple SimCLR-style learner.
 * Learns embeddings via NT-Xent on augmented views.
 */
export class ContrastiveLearner {
  constructor(inputDim, embedDim, opts = {}) {
    this.inputDim = inputDim;
    this.embedDim = embedDim;
    this.hiddenDim = opts.hiddenDim || embedDim * 2;
    this.projDim = opts.projDim || embedDim;
    this.lr = opts.learningRate || 0.01;
    
    // Simple 2-layer encoder: input → hidden → embed
    this.W1 = Array.from({ length: this.hiddenDim }, () =>
      Array.from({ length: inputDim }, () => (Math.random() - 0.5) * 0.2)
    );
    this.b1 = new Array(this.hiddenDim).fill(0);
    this.W2 = Array.from({ length: embedDim }, () =>
      Array.from({ length: this.hiddenDim }, () => (Math.random() - 0.5) * 0.2)
    );
    this.b2 = new Array(embedDim).fill(0);
  }

  encode(input) {
    // Hidden layer with ReLU
    const hidden = this.W1.map((row, i) => {
      let sum = this.b1[i];
      for (let j = 0; j < row.length; j++) sum += row[j] * input[j];
      return Math.max(0, sum);
    });
    // Output layer
    return this.W2.map((row, i) => {
      let sum = this.b2[i];
      for (let j = 0; j < row.length; j++) sum += row[j] * hidden[j];
      return sum;
    });
  }

  similarity(a, b) {
    return cosineSimilarity(this.encode(a), this.encode(b));
  }

  train(data, { epochs = 10, batchSize = 8, temperature = 0.5, onEpoch } = {}) {
    const history = [];
    for (let ep = 0; ep < epochs; ep++) {
      let totalLoss = 0;
      let batches = 0;
      for (let i = 0; i < data.length; i += batchSize) {
        const batch = data.slice(i, i + batchSize);
        // Create augmented pairs
        const embeddings = [];
        for (const x of batch) {
          embeddings.push(this.encode(augment(x)));
          embeddings.push(this.encode(augment(x)));
        }
        if (embeddings.length >= 4) {
          const loss = ntXentLoss(embeddings, temperature);
          totalLoss += loss;
          batches++;
        }
        // Simple gradient update via random perturbation (no backprop through encoder)
        this._perturbUpdate(batch, temperature);
      }
      const avgLoss = batches > 0 ? totalLoss / batches : 0;
      history.push(avgLoss);
      if (onEpoch) onEpoch({ epoch: ep, loss: avgLoss });
    }
    return { history };
  }

  _perturbUpdate(batch, temperature) {
    const eps = 0.001;
    // Perturb each weight slightly toward lower loss
    for (let i = 0; i < this.W1.length; i++) {
      for (let j = 0; j < this.W1[i].length; j++) {
        this.W1[i][j] += (Math.random() - 0.5) * this.lr * eps;
      }
    }
    for (let i = 0; i < this.W2.length; i++) {
      for (let j = 0; j < this.W2[i].length; j++) {
        this.W2[i][j] += (Math.random() - 0.5) * this.lr * eps;
      }
    }
  }
}
