/**
 * pages/index.js — Multi-Document RAG Assistant
 * ================================================
 * Full-stack RAG demo integrating Labs 1–4:
 *
 *  Lab 1 — Document indexing & chunking (client-side)
 *  Lab 2 — RAG Q&A via Groq API (/api/ask)
 *  Lab 3 — BM25 keyword retrieval (client-side, no API needed)
 *  Lab 4 — Multi-document support (PDF / CSV / TXT)
 *
 * Architecture (Groq-only, deploys on Vercel):
 *  - Documents are parsed → chunked → stored in React state (browser memory)
 *  - BM25 search runs entirely in the browser (no vector DB needed)
 *  - Only 2 API calls: /api/parse-pdf (PDF text) + /api/ask (Groq LLM)
 */

import Head from 'next/head';
import { useState, useRef } from 'react';
import Papa from 'papaparse';

// ─────────────────────────────────────────────────────────────────────────────
// Lab 3 — BM25 keyword search (runs in browser, no API key)
// ─────────────────────────────────────────────────────────────────────────────
function tokenize(text) {
  return text.toLowerCase().replace(/[^\w\s]/g, ' ').split(/\s+/).filter(t => t.length > 1);
}

function bm25Search(query, chunks, topK = 5, k1 = 1.5, b = 0.75) {
  if (!chunks.length) return [];
  const queryTerms = tokenize(query);
  if (!queryTerms.length) return chunks.slice(0, topK);

  const tokenizedDocs = chunks.map(c => tokenize(c.text));
  const avgLen = tokenizedDocs.reduce((s, d) => s + d.length, 0) / tokenizedDocs.length;
  const N = chunks.length;

  // Precompute document frequencies
  const df = {};
  for (const tokens of tokenizedDocs) {
    for (const t of new Set(tokens)) df[t] = (df[t] || 0) + 1;
  }

  const scored = chunks.map((chunk, i) => {
    const docTokens = tokenizedDocs[i];
    const dl = docTokens.length;
    let score = 0;
    for (const term of queryTerms) {
      const tf = docTokens.filter(t => t === term).length;
      if (!tf) continue;
      const n = df[term] || 0;
      const idf = Math.log((N - n + 0.5) / (n + 0.5) + 1);
      score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgLen));
    }
    return { ...chunk, score };
  });

  return scored.sort((a, b) => b.score - a.score).slice(0, topK);
}

// ─────────────────────────────────────────────────────────────────────────────
// Lab 1 — Text chunker (client-side)
// ─────────────────────────────────────────────────────────────────────────────
function chunkText(text, chunkSize = 800, overlap = 150) {
  const paragraphs = text.split(/\n\n+/).map(p => p.trim()).filter(Boolean);
  const chunks = [];
  let cur = '';
  for (const para of paragraphs) {
    const merged = cur ? cur + '\n\n' + para : para;
    if (cur && merged.length > chunkSize) {
      chunks.push(cur.trim());
      cur = cur.slice(-overlap) + '\n\n' + para;
    } else {
      cur = merged;
    }
  }
  if (cur.trim()) chunks.push(cur.trim());

  const result = [];
  for (const chunk of chunks) {
    if (chunk.length <= chunkSize) { result.push(chunk); continue; }
    for (let i = 0; i < chunk.length; i += chunkSize - overlap) {
      const s = chunk.slice(i, i + chunkSize).trim();
      if (s) result.push(s);
    }
  }
  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
const FILE_ICONS = { pdf: '📄', csv: '📊', txt: '📝' };
// Groq model IDs — matches what's shown in console.groq.com/docs/models
const GROQ_MODELS = [
  // ── Featured (most likely on your account) ─────────────
  'compound-beta',               // Groq Compound (~450 tps)
  'compound-beta-mini',          // Groq Compound Mini
  // ── OpenAI GPT-OSS 120B on Groq ────────────────────────
  'openai/gpt-4o',               // OpenAI GPT-OSS 120B via Groq
  // ── Production LLaMA ───────────────────────────────────
  'llama3-8b-8192',
  'llama3-70b-8192',
  'llama-3.3-70b-versatile',
  'llama-3.1-8b-instant',
  // ── Other production models ─────────────────────────────
  'mixtral-8x7b-32768',
  'gemma2-9b-it',
  'deepseek-r1-distill-llama-70b',
  // ── Custom ─────────────────────────────────────────────
  'custom — type below',         // Paste any model ID from your console
];


// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────
export default function Home() {
  // API / model config
  const [groqKey, setGroqKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [model, setModel] = useState('compound-beta');
  const [customModel, setCustomModel] = useState('');

  // Document store: [{filename, fileType, category, chunks:[{text,page,row}]}]
  const [docStore, setDocStore] = useState([]);
  const [pendingFiles, setPendingFiles] = useState([]);
  const [category, setCategory] = useState('General');
  const [isProcessing, setIsProcessing] = useState(false);

  // Retrieval
  const [topK, setTopK] = useState(5);
  const [filterDoc, setFilterDoc] = useState('All');

  // Chat
  const [chatHistory, setChatHistory] = useState([]);
  const [question, setQuestion] = useState('');
  const [isAsking, setIsAsking] = useState(false);

  // Alerts
  const [alerts, setAlerts] = useState([]);
  const [dragging, setDragging] = useState(false);

  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);

  // ── Derived state ──────────────────────────────────────────────────────────
  const allChunks = docStore.flatMap(doc =>
    doc.chunks.map(c => ({
      ...c, filename: doc.filename, fileType: doc.fileType, category: doc.category,
    }))
  );
  const filteredChunks = filterDoc === 'All'
    ? allChunks
    : allChunks.filter(c => c.filename === filterDoc);
  const docNames = docStore.map(d => d.filename);

  // ── Alerts ─────────────────────────────────────────────────────────────────
  function addAlert(type, msg) {
    const id = Date.now() + Math.random();
    setAlerts(prev => [...prev, { id, type, msg }]);
    setTimeout(() => setAlerts(prev => prev.filter(a => a.id !== id)), 6000);
  }

  // ── File selection ─────────────────────────────────────────────────────────
  function onFilesSelected(files) {
    const valid = Array.from(files).filter(f => {
      const ext = f.name.split('.').pop().toLowerCase();
      return ['pdf', 'csv', 'txt'].includes(ext);
    });
    setPendingFiles(prev => {
      const names = new Set(prev.map(f => f.name));
      return [...prev, ...valid.filter(f => !names.has(f.name))];
    });
  }

  // ── Lab 1: Process & index files ──────────────────────────────────────────
  async function processFiles() {
    if (!pendingFiles.length) { addAlert('warning', 'Select at least one file first.'); return; }
    setIsProcessing(true);

    for (const file of pendingFiles) {
      const ext = file.name.split('.').pop().toLowerCase();
      if (docStore.some(d => d.filename === file.name)) {
        addAlert('warning', `'${file.name}' is already indexed.`); continue;
      }

      try {
        let chunks = [];

        if (ext === 'txt') {
          // TXT: read directly in browser
          const text = await file.text();
          chunks = chunkText(text).map(text => ({ text, page: 'N/A', row: 'N/A' }));

        } else if (ext === 'csv') {
          // CSV: parse with PapaParse — each row = one chunk
          const text = await file.text();
          const { data } = Papa.parse(text, { header: true, skipEmptyLines: true });
          chunks = data.map((row, i) => ({
            text: Object.entries(row).map(([k, v]) => `${k}: ${v}`).join(' | '),
            page: 'N/A',
            row: String(i + 1),
          }));

        } else if (ext === 'pdf') {
          // PDF: send to server-side API to extract text
          const buffer = await file.arrayBuffer();
          const bytes = new Uint8Array(buffer);
          let binary = '';
          for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
          const base64 = btoa(binary);

          const res = await fetch('/api/parse-pdf', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fileData: base64 }),
          });
          if (!res.ok) throw new Error('PDF parsing failed');
          const { text } = await res.json();
          chunks = chunkText(text).map(text => ({ text, page: 'N/A', row: 'N/A' }));
        }

        if (!chunks.length) { addAlert('warning', `No text extracted from '${file.name}'.`); continue; }

        setDocStore(prev => [...prev, { filename: file.name, fileType: ext, category, chunks }]);
        addAlert('success', `'${file.name}' indexed — ${chunks.length} chunks.`);
      } catch (e) {
        addAlert('error', `'${file.name}': ${e.message}`);
      }
    }

    setPendingFiles([]);
    setIsProcessing(false);
  }

  // ── Lab 2 + 3: Ask question ────────────────────────────────────────────────
  async function askQuestion() {
    const q = question.trim();
    if (!q) return;
    if (!groqKey) { addAlert('error', 'Enter your Groq API key in the sidebar.'); return; }
    if (!allChunks.length) { addAlert('warning', 'No documents indexed yet.'); return; }

    setQuestion('');
    setIsAsking(true);
    setChatHistory(prev => [...prev, { role: 'user', content: q }]);
    setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 50);

    // Lab 3 — BM25 keyword retrieval (runs in browser)
    const topChunks = bm25Search(q, filteredChunks, topK);

    // Lab 2 — Groq RAG answer
    const effectiveModel = model === 'custom — type below' ? customModel.trim() : model;
    if (!effectiveModel) { addAlert('error', 'Enter a custom model name.'); setIsAsking(false); return; }
    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, chunks: topChunks, groqKey, model: effectiveModel }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'API error');
      setChatHistory(prev => [...prev, { role: 'assistant', content: data.answer, sources: topChunks }]);
    } catch (e) {
      setChatHistory(prev => [...prev, { role: 'assistant', content: `❌ Error: ${e.message}`, sources: [] }]);
    }

    setIsAsking(false);
    setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 100);
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────────────────────
  return (
    <>
      <Head>
        <title>Multi-Document RAG Assistant</title>
        <meta name="description" content="Ask questions across your documents using Groq AI and BM25 search" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet" />
      </Head>

      <div className="layout">

        {/* ━━━━━━━━━━━━━━━━━━ SIDEBAR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
        <aside className="sidebar">
          <div className="sidebar-logo">📚 RAG Assistant</div>

          {/* ── API Config ──────────────────────────────────────────────────── */}
          <div>
            <div className="section-label">⚙️ Configuration</div>
            <div className="card">
              <label>Groq API Key</label>
              <div style={{ position: 'relative' }}>
                <input
                  type={showKey ? 'text' : 'password'}
                  value={groqKey}
                  onChange={e => setGroqKey(e.target.value)}
                  placeholder="gsk_..."
                  style={{ paddingRight: '2.5rem' }}
                />
                <button
                  onClick={() => setShowKey(s => !s)}
                  style={{
                    position: 'absolute', right: '0.5rem', top: '50%',
                    transform: 'translateY(-50%)', background: 'none',
                    border: 'none', cursor: 'pointer', color: 'var(--muted)',
                    fontSize: '0.9rem', paddingTop: '0.35rem'
                  }}
                >{showKey ? '🙈' : '👁'}</button>
              </div>
              <div style={{ fontSize: '0.68rem', color: 'var(--muted)', marginTop: '0.35rem' }}>
                Get free key: <a href="https://console.groq.com" target="_blank" rel="noreferrer"
                  style={{ color: 'var(--accent)' }}>console.groq.com</a>
              </div>

              <label style={{ marginTop: '0.85rem' }}>LLM Model</label>
              <select value={model} onChange={e => setModel(e.target.value)}>
                {GROQ_MODELS.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
              {model === 'custom — type below' && (
                <input
                  type="text"
                  value={customModel}
                  onChange={e => setCustomModel(e.target.value)}
                  placeholder="e.g. llama-3.2-1b-preview"
                  style={{ marginTop: '0.4rem' }}
                />
              )}
              <div style={{ fontSize: '0.68rem', color: 'var(--muted)', marginTop: '0.3rem' }}>
                Check available models: <a href="https://console.groq.com/docs/models" target="_blank" rel="noreferrer" style={{ color: 'var(--accent)' }}>console.groq.com/docs/models</a>
              </div>
            </div>
          </div>

          {/* ── Upload ──────────────────────────────────────────────────────── */}
          <div>
            <div className="section-label">📂 Upload Documents</div>
            <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
              {/* Drop zone */}
              <div
                className={`upload-zone${dragging ? ' drag' : ''}`}
                onDragOver={e => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={e => { e.preventDefault(); setDragging(false); onFilesSelected(e.dataTransfer.files); }}
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="upload-zone-icon">📁</div>
                <div className="upload-zone-text">
                  Drop PDF / CSV / TXT files<br />or click to browse
                </div>
                <input
                  ref={fileInputRef} type="file" multiple accept=".pdf,.csv,.txt"
                  style={{ display: 'none' }}
                  onChange={e => onFilesSelected(e.target.files)}
                />
              </div>

              {/* Pending file chips */}
              {pendingFiles.map(f => (
                <div key={f.name} className="file-chip">
                  <span>{FILE_ICONS[f.name.split('.').pop().toLowerCase()] || '📄'}</span>
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
                    {f.name}
                  </span>
                  <span
                    className="file-chip-remove"
                    onClick={() => setPendingFiles(p => p.filter(x => x.name !== f.name))}
                  >✕</span>
                </div>
              ))}

              <label>Product Category</label>
              <input
                type="text" value={category}
                onChange={e => setCategory(e.target.value)}
                placeholder="e.g. Electronics, HR, Finance"
              />

              <div className="row" style={{ marginTop: '0.25rem' }}>
                <button
                  className="btn btn-primary"
                  disabled={isProcessing || !pendingFiles.length}
                  onClick={processFiles}
                >
                  {isProcessing
                    ? <><span className="spinner" style={{ width: 12, height: 12 }} /> Indexing…</>
                    : '📥 Index Files'}
                </button>
                <button
                  className="btn btn-danger"
                  onClick={() => { setDocStore([]); setChatHistory([]); addAlert('success', 'Database cleared.'); }}
                >🗑️</button>
              </div>
            </div>
          </div>

          {/* ── Stats ───────────────────────────────────────────────────────── */}
          <div>
            <div className="section-label">🗄️ Index Stats</div>
            <div className="card">
              <div className="stat-big">{allChunks.length}</div>
              <div className="stat-label">Indexed Chunks from {docNames.length} document(s)</div>
              {docNames.length > 0
                ? <div style={{ marginTop: '0.75rem' }}>
                    {docStore.map(doc => (
                      <div key={doc.filename} className="file-chip" style={{ marginTop: '0.35rem' }}>
                        <span>{FILE_ICONS[doc.fileType] || '📄'}</span>
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1, fontSize: '0.72rem' }}>
                          {doc.filename}
                        </span>
                        <span style={{ fontSize: '0.68rem', color: 'var(--muted)', marginLeft: 'auto' }}>
                          {doc.chunks.length}c
                        </span>
                      </div>
                    ))}
                  </div>
                : <div style={{ fontSize: '0.75rem', color: 'var(--muted)', marginTop: '0.5rem' }}>
                    No documents indexed yet.
                  </div>
              }
            </div>
          </div>

          {/* ── Retrieval Settings ───────────────────────────────────────────── */}
          <div>
            <div className="section-label">🔍 Retrieval Settings</div>
            <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
              <label>Document Filter</label>
              <select value={filterDoc} onChange={e => setFilterDoc(e.target.value)}>
                <option value="All">All Documents</option>
                {docNames.map(n => <option key={n} value={n}>{n}</option>)}
              </select>
              <label>
                Top-K Chunks: <strong style={{ color: 'var(--accent)' }}>{topK}</strong>
              </label>
              <input
                type="range" min={1} max={15} value={topK}
                onChange={e => setTopK(Number(e.target.value))}
              />
            </div>
          </div>
        </aside>

        {/* ━━━━━━━━━━━━━━━━━━ MAIN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
        <div className="main">
          <div className="main-scroll">

            {/* Alerts */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', marginBottom: alerts.length ? '1rem' : 0 }}>
              {alerts.map(a => (
                <div key={a.id} className={`alert alert-${a.type}`}>{a.msg}</div>
              ))}
            </div>

            {/* ── Hero ──────────────────────────────────────────────────────── */}
            <div className="hero-title">Multi-Document RAG Assistant</div>
            <div className="hero-sub">
              Upload documents, ask questions — answers grounded in your data, powered by Groq AI
            </div>
            <div className="badges">
              <span className="badge">📦 Lab 1 · Chunking & Indexing</span>
              <span className="badge">🤖 Lab 2 · RAG Q&A (Groq)</span>
              <span className="badge">🔍 Lab 3 · BM25 Hybrid Search</span>
              <span className="badge">📚 Lab 4 · Multi-Document</span>
            </div>

            {/* ── Pipeline explainer ─────────────────────────────────────────── */}
            <details style={{ marginTop: '1.25rem' }}>
              <summary>📊 How the RAG pipeline works</summary>
              <div className="card pipeline-grid" style={{ marginTop: '0.6rem' }}>
                <div className="pipeline-step">
                  <strong>Indexing (Lab 1 + Lab 4)</strong><br />
                  1. Upload PDF / CSV / TXT files<br />
                  2. Extract text (pdf-parse / PapaParse / plain)<br />
                  3. Split into overlapping chunks (800 chars, 150 overlap)<br />
                  4. Store in browser with metadata (filename, type, category)
                </div>
                <div className="pipeline-step">
                  <strong>Retrieval &amp; Answer (Lab 2 + Lab 3)</strong><br />
                  1. Apply document filter (Lab 3)<br />
                  2. BM25 keyword ranking on all chunks (Lab 3, client-side)<br />
                  3. Select Top-K most relevant chunks<br />
                  4. Build grounded prompt → Groq LLM (Lab 2)<br />
                  5. Return answer + cited sources
                </div>
              </div>
            </details>

            <div className="divider" />

            {/* ── Chat history ──────────────────────────────────────────────── */}
            {chatHistory.map((msg, i) => (
              <div key={i} className={`chat-msg ${msg.role}`}>
                <div className="bubble-label">
                  {msg.role === 'user' ? '🧑 You' : '🤖 Assistant (Groq)'}
                </div>
                <div className={`bubble ${msg.role}`} style={{ whiteSpace: 'pre-wrap' }}>
                  {msg.content}
                </div>

                {/* Source cards */}
                {msg.role === 'assistant' && msg.sources?.length > 0 && (
                  <div style={{ maxWidth: '80%', width: '100%', marginTop: '0.25rem' }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--muted)', marginBottom: '0.4rem' }}>
                      📎 Retrieved {msg.sources.length} chunk(s) via BM25:
                    </div>
                    <div className="sources-grid">
                      {msg.sources.slice(0, 4).map((src, j) => (
                        <div key={j} className="source-card">
                          <div className="source-card-title">
                            {FILE_ICONS[src.fileType] || '📄'} {src.filename}
                          </div>
                          <div className="source-card-meta">
                            {src.fileType?.toUpperCase()}&nbsp;·&nbsp;
                            {src.category}&nbsp;·&nbsp;
                            {src.row !== 'N/A' ? `Row ${src.row}` : src.page !== 'N/A' ? `Page ${src.page}` : '—'}
                            &nbsp;·&nbsp;score: {src.score?.toFixed(3)}
                          </div>
                          <div className="source-card-preview">
                            &ldquo;{src.text.slice(0, 160)}{src.text.length > 160 ? '…' : ''}&rdquo;
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}

            {/* Thinking indicator */}
            {isAsking && (
              <div className="chat-msg assistant">
                <div className="bubble-label">🤖 Assistant (Groq)</div>
                <div className="bubble assistant" style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                  <span className="spinner" style={{ borderColor: 'rgba(165,180,252,0.3)', borderTopColor: '#a5b4fc' }} />
                  Searching {filteredChunks.length} chunks with BM25 → asking Groq…
                </div>
              </div>
            )}

            {/* Empty state */}
            {!chatHistory.length && !isAsking && (
              <div className="empty-state">
                <div className="empty-icon">💬</div>
                <div style={{ fontWeight: 600, color: '#94a3b8', fontSize: '1rem', marginBottom: '0.5rem' }}>
                  No conversation yet
                </div>
                <div style={{ fontSize: '0.82rem', lineHeight: 1.6 }}>
                  1. Enter your <strong>Groq API key</strong> above<br />
                  2. Upload a <strong>CSV, PDF, or TXT</strong> file<br />
                  3. Click <strong>Index Files</strong><br />
                  4. Ask anything below ↓
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>

          {/* ── Question form ────────────────────────────────────────────────── */}
          <div className="qa-form">
            <textarea
              value={question}
              onChange={e => setQuestion(e.target.value)}
              placeholder="Ask anything about your documents… (Shift+Enter for new line, Enter to send)"
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  if (!isAsking && question.trim()) askQuestion();
                }
              }}
            />
            <button
              className="btn btn-primary"
              onClick={askQuestion}
              disabled={isAsking || !question.trim()}
              style={{ minWidth: 90 }}
            >
              {isAsking ? <span className="spinner" /> : '🚀 Ask'}
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
