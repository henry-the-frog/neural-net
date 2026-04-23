// embedding-similarity.js — Word embedding operations
export function analogySolve(embeddings, a, b, c, topK = 5) {
  // king - man + woman = queen
  // Find d such that d ≈ b - a + c
  const target = a.map((v, i) => b[i] - v + c[i]);
  
  return embeddings.map((emb, idx) => {
    let dot = 0, normT = 0, normE = 0;
    for (let i = 0; i < emb.length; i++) {
      dot += target[i] * emb[i];
      normT += target[i] ** 2;
      normE += emb[i] ** 2;
    }
    return { idx, score: dot / (Math.sqrt(normT * normE) + 1e-8) };
  }).sort((a, b) => b.score - a.score).slice(0, topK);
}

export function wordSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]; normA += a[i] ** 2; normB += b[i] ** 2;
  }
  return dot / (Math.sqrt(normA * normB) + 1e-8);
}
