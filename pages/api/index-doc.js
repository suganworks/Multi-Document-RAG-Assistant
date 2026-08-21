/**
 * pages/api/index-doc.js — Lab 1 + Lab 3 Integration
 * ====================================================
 * Document indexing pipeline:
 *   1. Receive base64-encoded file (PDF / CSV / TXT)
 *   2. Extract text with appropriate parser
 *   3. Split into overlapping chunks (Lab 1)
 *   4. Generate OpenAI embeddings (Lab 1)
 *   5. Store in Pinecone with rich metadata (Lab 1 / Lab 3)
 */

import OpenAI from 'openai';
import { Pinecone } from '@pinecone-database/pinecone';
import pdfParse from 'pdf-parse';
import Papa from 'papaparse';

export const config = {
  api: { bodyParser: { sizeLimit: '20mb' } },
};

// ─── Text chunking (Lab 1) ────────────────────────────────────────────────────
function chunkText(text, chunkSize = 1000, overlap = 200) {
  const chunks = [];
  // First split on paragraphs for natural boundaries
  const paragraphs = text.split(/\n\n+/).map(p => p.trim()).filter(Boolean);

  let current = '';
  for (const para of paragraphs) {
    if ((current + '\n\n' + para).length > chunkSize && current) {
      chunks.push(current.trim());
      // Overlap: keep last overlap chars
      current = current.slice(-overlap) + '\n\n' + para;
    } else {
      current = current ? current + '\n\n' + para : para;
    }
  }
  if (current.trim()) chunks.push(current.trim());

  // If any chunk is still too long, hard-split it
  const result = [];
  for (const chunk of chunks) {
    if (chunk.length <= chunkSize) {
      result.push(chunk);
    } else {
      for (let i = 0; i < chunk.length; i += chunkSize - overlap) {
        const slice = chunk.slice(i, i + chunkSize);
        if (slice.trim()) result.push(slice.trim());
      }
    }
  }
  return result;
}

// ─── Deduplication check ─────────────────────────────────────────────────────
async function isAlreadyIndexed(index, filename) {
  try {
    const res = await index.query({
      vector: Array(1536).fill(0),
      topK: 1,
      filter: { filename: { $eq: filename } },
      includeMetadata: true,
    });
    return res.matches && res.matches.length > 0;
  } catch {
    return false;
  }
}

// ─── OpenAI embeddings in batches ─────────────────────────────────────────────
async function embedTexts(texts, openai) {
  const BATCH = 96;
  const all = [];
  for (let i = 0; i < texts.length; i += BATCH) {
    const batch = texts.slice(i, i + BATCH);
    const res = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: batch,
    });
    all.push(...res.data.map(d => d.embedding));
  }
  return all;
}

// ─── Main handler ─────────────────────────────────────────────────────────────
export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  const {
    fileData,      // base64-encoded file content
    filename,
    fileType,      // 'pdf' | 'csv' | 'txt'
    productCategory = 'General',
    openaiKey,
    groqKey,
    pineconeKey,
    pineconeIndexName = 'multi-doc-rag',
  } = req.body;

  if (!fileData || !filename || !fileType) {
    return res.status(400).json({ error: 'Missing required fields: fileData, filename, fileType' });
  }
  if (!openaiKey) return res.status(400).json({ error: 'OpenAI API key required for embeddings' });
  if (!pineconeKey) return res.status(400).json({ error: 'Pinecone API key required' });

  try {
    // ── Init clients ──────────────────────────────────────────────────────────
    const openai = new OpenAI({ apiKey: openaiKey });
    const pinecone = new Pinecone({ apiKey: pineconeKey });
    const index = pinecone.index(pineconeIndexName);

    // ── Deduplication ─────────────────────────────────────────────────────────
    if (await isAlreadyIndexed(index, filename)) {
      return res.status(200).json({ chunksAdded: 0, message: `'${filename}' is already indexed. Skipping.` });
    }

    // ── Extract text ──────────────────────────────────────────────────────────
    const buffer = Buffer.from(fileData, 'base64');
    let chunks = [];
    let docPages = [];  // For PDF page tracking

    if (fileType === 'pdf') {
      const data = await pdfParse(buffer);
      const text = data.text;
      chunks = chunkText(text);
      docPages = chunks.map((_, i) => String(Math.floor(i * data.numpages / chunks.length) + 1));
    } else if (fileType === 'csv') {
      const csvText = buffer.toString('utf-8');
      const { data: rows } = Papa.parse(csvText, { header: true, skipEmptyLines: true });
      // Each CSV row = one chunk (natural unit of data)
      chunks = rows.map(row => Object.entries(row).map(([k, v]) => `${k}: ${v}`).join('\n'));
      docPages = rows.map((_, i) => String(i + 1));
    } else if (fileType === 'txt') {
      const text = buffer.toString('utf-8');
      chunks = chunkText(text);
      docPages = chunks.map(() => 'N/A');
    } else {
      return res.status(400).json({ error: `Unsupported file type: ${fileType}` });
    }

    if (chunks.length === 0) {
      return res.status(200).json({ chunksAdded: 0, message: `No text extracted from '${filename}'.` });
    }

    // ── Generate embeddings ───────────────────────────────────────────────────
    const vectors = await embedTexts(chunks, openai);

    // ── Build Pinecone records ─────────────────────────────────────────────────
    const records = chunks.map((text, i) => ({
      id: `${filename}__chunk_${i}`,
      values: vectors[i],
      metadata: {
        text,
        filename,
        file_type: fileType,
        page: docPages[i] || 'N/A',
        row: fileType === 'csv' ? String(i + 1) : 'N/A',
        product_category: productCategory,
        source: filename,
      },
    }));

    // ── Upsert in batches of 100 ──────────────────────────────────────────────
    const PINECONE_BATCH = 100;
    for (let i = 0; i < records.length; i += PINECONE_BATCH) {
      await index.upsert(records.slice(i, i + PINECONE_BATCH));
    }

    return res.status(200).json({
      chunksAdded: chunks.length,
      message: `Indexed '${filename}' — ${chunks.length} chunks added.`,
    });

  } catch (err) {
    console.error('index-doc error:', err);
    return res.status(500).json({ error: err.message || 'Indexing failed' });
  }
}
