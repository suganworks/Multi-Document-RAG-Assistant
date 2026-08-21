/**
 * pages/api/ask.js — Lab 2 Integration
 * =======================================
 * RAG answer generation using Groq LLM.
 * Receives pre-retrieved chunks (BM25 done client-side in Lab 3),
 * builds a grounded prompt, calls Groq, and returns the answer.
 *
 * The LLM is strictly instructed to answer ONLY from the provided context.
 */

import Groq from 'groq-sdk';

export const config = {
  api: { bodyParser: { sizeLimit: '4mb' } },
};

const SYSTEM_PROMPT = `You are a helpful document assistant.

Rules:
1. Answer ONLY using the provided context. Do not use outside knowledge.
2. If the answer is not in the context, reply exactly: "I couldn't find this information in the uploaded documents."
3. Be concise, accurate, and cite the source document name naturally.
4. Never guess or hallucinate.`;

export default async function handler(req, res) {
  if (req.method !== 'POST') return res.status(405).end();

  const { question, chunks, groqKey, model = 'llama-3.1-8b-instant' } = req.body;

  if (!groqKey) return res.status(400).json({ error: 'Groq API key is required.' });
  if (!question) return res.status(400).json({ error: 'Question is required.' });
  if (!chunks || chunks.length === 0) return res.status(400).json({ error: 'No context chunks provided.' });

  try {
    // Build context string from retrieved chunks (Lab 2 prompt construction)
    const context = chunks.map((c, i) => {
      const loc = c.row !== 'N/A' ? `Row ${c.row}`
        : c.page !== 'N/A' ? `Page ${c.page}`
        : 'Section 1';
      return `[Source ${i + 1}: ${c.filename} — ${loc}]\n${c.text}`;
    }).join('\n\n---\n\n');

    const groq = new Groq({ apiKey: groqKey });

    const response = await groq.chat.completions.create({
      model,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: `Context:\n${context}\n\n---\n\nQuestion: ${question}` },
      ],
      temperature: 0,
      max_tokens: 800,
    });

    const answer = response.choices[0].message.content.trim();
    res.status(200).json({ answer });
  } catch (e) {
    console.error('ask error:', e);
    res.status(500).json({ error: e.message || 'Groq API error' });
  }
}
