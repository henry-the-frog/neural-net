// cosine-sim-search.js — Cosine similarity search for embedding retrieval
export function buildIndex(embeddings) {
  // Normalize all embeddings
  return embeddings.map(emb => {
    const norm = Math.sqrt(emb.reduce((s, v) => s + v * v, 0));
    return emb.map(v => v / (norm + 1e-8));
  });
}

export function search(queryEmb, index, topK = 5) {
  const queryNorm = Math.sqrt(queryEmb.reduce((s, v) => s + v * v, 0));
  const normalizedQuery = queryEmb.map(v => v / (queryNorm + 1e-8));
  
  const scores = index.map((emb, idx) => {
    let dot = 0;
    for (let i = 0; i < emb.length; i++) dot += normalizedQuery[i] * emb[i];
    return { idx, score: dot };
  });
  
  scores.sort((a, b) => b.score - a.score);
  return scores.slice(0, topK);
}
