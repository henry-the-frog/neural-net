// rag.js — Retrieval Augmented Generation components
export function retrieveTopK(query, documents, scoreFn, k = 3) {
  return documents.map((doc, idx) => ({ idx, doc, score: scoreFn(query, doc) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

export function buildRAGPrompt(query, retrievedDocs, template = null) {
  const context = retrievedDocs.map((d, i) => `[${i+1}] ${d.doc}`).join('\n');
  if (template) return template.replace('{context}', context).replace('{query}', query);
  return `Context:\n${context}\n\nQuestion: ${query}\nAnswer:`;
}

export function chunkText(text, chunkSize = 256, overlap = 32) {
  const chunks = [];
  for (let i = 0; i < text.length; i += chunkSize - overlap) {
    chunks.push(text.slice(i, i + chunkSize));
    if (i + chunkSize >= text.length) break;
  }
  return chunks;
}
