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
  let correct = 0;
  
  for (let i = 0; i < N2; i++) {
    // Positive pair index
    const j = i % 2 === 0 ? i + 1 : i - 1;
    
    // Numerator: exp(sim(i, j))
    const positiveScore = sim[i][j];
    
    // Denominator: sum over all k ≠ i
    let logSumExp = -Infinity;
    let maxSim = -Infinity;
    for (let k = 0; k < N2; k++) {
      if (k !== i) maxSim = Math.max(maxSim, sim[i][k]);
    }
    
    let sumExp = 0;
    for (let k = 0; k < N2; k++) {
      if (k !== i) sumExp += Math.exp(sim[i][k] - maxSim);
    }
    logSumExp = Math.log(sumExp) + maxSim;
    
    totalLoss += -positiveScore + logSumExp;
    
    // Accuracy: is positive pair the most similar?
    let isMax = true;
    for (let k = 0; k < N2; k++) {
      if (k !== i && k !== j && sim[i][k] >= sim[i][j]) {
        isMax = false;
        break;
      }
    }
    if (isMax) correct++;
  }
  
  return {
    loss: totalLoss / N2,
    accuracy: correct / N2,
  };
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
