/**
 * pages/api/parse-pdf.js
 * ========================
 * Server-side PDF text extraction.
 * Receives a base64-encoded PDF, returns plain text.
 * Used because PDF parsing requires Node.js (not available in the browser).
 */

import pdfParse from 'pdf-parse';

export const config = {
  api: { bodyParser: { sizeLimit: '20mb' } },
};

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).end();
  try {
    const { fileData } = req.body;
    if (!fileData) return res.status(400).json({ error: 'fileData (base64) required' });

    const buffer = Buffer.from(fileData, 'base64');
    const { text, numpages } = await pdfParse(buffer);

    res.status(200).json({ text, pages: numpages });
  } catch (e) {
    console.error('parse-pdf error:', e);
    res.status(500).json({ error: e.message || 'PDF parsing failed' });
  }
}
