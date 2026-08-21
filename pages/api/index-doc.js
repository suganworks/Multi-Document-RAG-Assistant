/**
 * pages/api/index-doc.js
 * This endpoint is superseded by /api/parse-pdf + client-side BM25.
 * Kept as a stub to avoid 404 errors.
 */
export default function handler(req, res) {
  res.status(410).json({ error: 'This endpoint is deprecated. Use /api/parse-pdf instead.' });
}
