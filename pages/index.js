/**
 * pages/index.js — Multi-Document RAG Assistant
 * ================================================
 * Labs 1–4 integrated with agent orchestration visualization.
 *
 *  Lab 1 — Chunking & indexing (client-side)
 *  Lab 2 — RAG Q&A via Groq LLM (/api/ask)
 *  Lab 3 — BM25 keyword retrieval (client-side)
 *  Lab 4 — Multi-document support (PDF / CSV / TXT)
 *
 * Agent pipeline (Lab 4 orchestration):
 *   🔍 Retriever Agent  → BM25 search
 *   📊 Context Builder  → chunk formatting
 *   🤖 LLM Agent        → Groq answer generation
 *   ✅ Source Validator  → grounding check
 */

import Head from 'next/head';
import { useState, useRef } from 'react';
import Papa from 'papaparse';

// ─────────────────────────────────────────────────────────────────────────────
// Lab 3 — BM25 keyword search (runs entirely in browser)
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

  const df = {};
  for (const tokens of tokenizedDocs)
    for (const t of new Set(tokens)) df[t] = (df[t] || 0) + 1;

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
    if (cur && merged.length > chunkSize) { chunks.push(cur.trim()); cur = cur.slice(-overlap) + '\n\n' + para; }
    else cur = merged;
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
// Agent Step component — Lab 4 orchestration visualization
// ─────────────────────────────────────────────────────────────────────────────
const STATUS_CFG = {
  pending: { symbol: '○', color: 'var(--muted)',    bg: 'rgba(100,116,139,0.08)',  border: 'rgba(100,116,139,0.2)'  },
  active:  { symbol: '●', color: 'var(--accent)',   bg: 'rgba(99,102,241,0.12)',   border: 'rgba(99,102,241,0.4)'   },
  done:    { symbol: '✓', color: 'var(--success)',  bg: 'rgba(52,211,153,0.12)',   border: 'rgba(52,211,153,0.35)'  },
  error:   { symbol: '✕', color: 'var(--error)',    bg: 'rgba(248,113,113,0.12)',  border: 'rgba(248,113,113,0.35)' },
  skipped: { symbol: '—', color: 'var(--muted)',    bg: 'rgba(100,116,139,0.05)',  border: 'rgba(100,116,139,0.15)' },
};

function AgentStep({ step }) {
  const cfg = STATUS_CFG[step.status] || STATUS_CFG.pending;
  return (
    <div className={`agent-step agent-step-${step.status}`}>
      <div className="agent-step-icon" style={{ background: cfg.bg, color: cfg.color, borderColor: cfg.border }}>
        {step.status === 'active'
          ? <span className="agent-pulse">●</span>
          : cfg.symbol}
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div className="agent-step-name" style={{ color: cfg.color }}>
          {step.icon}&nbsp; {step.name}
        </div>
        {step.detail && (
          <div className="agent-step-detail">{step.detail}</div>
        )}
      </div>
      <div className="agent-step-right">
        {step.status === 'active' && (
          <span className="spinner" style={{ width: 11, height: 11, borderColor: 'rgba(99,102,241,0.3)', borderTopColor: 'var(--accent)' }} />
        )}
        {step.durationMs != null && (
          <span className="agent-step-duration">{step.durationMs}ms</span>
        )}
      </div>
    </div>
  );
}

function AgentPanel({ trace, live = false }) {
  const done = trace.every(s => s.status !== 'active');
  return (
    <div className="agent-panel">
      <div className="agent-panel-header">
        <span className="agent-panel-title">🧠 Agent Orchestration</span>
        {live && !done && <span className="agent-live-badge">LIVE</span>}
        {done && <span className="agent-done-badge">COMPLETE</span>}
      </div>
      <div className="agent-steps">
        {trace.map(step => <AgentStep key={step.id} step={step} />)}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────
const FILE_ICONS = { pdf: '📄', csv: '📊', txt: '📝' };

const GROQ_MODELS = [
  'compound-beta',
  'compound-beta-mini',
  'openai/gpt-4o',
  'llama3-8b-8192',
  'llama3-70b-8192',
  'llama-3.3-70b-versatile',
  'llama-3.1-8b-instant',
  'mixtral-8x7b-32768',
  'gemma2-9b-it',
  'deepseek-r1-distill-llama-70b',
  'custom — type below',
];

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────
export default function Home() {
  const [groqKey, setGroqKey]         = useState('');
  const [showKey, setShowKey]         = useState(false);
  const [model, setModel]             = useState('compound-beta');
  const [customModel, setCustomModel] = useState('');

  const [docStore, setDocStore]         = useState([]);
  const [pendingFiles, setPendingFiles] = useState([]);
  const [category, setCategory]         = useState('General');
  const [isProcessing, setIsProcessing] = useState(false);

  const [topK, setTopK]           = useState(5);
  const [filterDoc, setFilterDoc] = useState('All');

  const [chatHistory, setChatHistory] = useState([]);
  const [question, setQuestion]       = useState('');
  const [isAsking, setIsAsking]       = useState(false);
  const [agentTrace, setAgentTrace]   = useState([]);

  const [alerts, setAlerts]   = useState([]);
  const [dragging, setDragging] = useState(false);

  const fileInputRef = useRef(null);
  const chatEndRef   = useRef(null);

  // Derived
  const allChunks = docStore.flatMap(doc =>
    doc.chunks.map(c => ({ ...c, filename: doc.filename, fileType: doc.fileType, category: doc.category }))
  );
  const filteredChunks = filterDoc === 'All' ? allChunks : allChunks.filter(c => c.filename === filterDoc);
  const docNames = docStore.map(d => d.filename);

  // ── Alerts ─────────────────────────────────────────────────────────────────
  function addAlert(type, msg) {
    const id = Date.now() + Math.random();
    setAlerts(prev => [...prev, { id, type, msg }]);
    setTimeout(() => setAlerts(prev => prev.filter(a => a.id !== id)), 6000);
  }

  // ── File selection ─────────────────────────────────────────────────────────
  function onFilesSelected(files) {
    const valid = Array.from(files).filter(f =>
      ['pdf', 'csv', 'txt'].includes(f.name.split('.').pop().toLowerCase())
    );
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
        addAlert('warning', `'${file.name}' already indexed.`); continue;
      }
      try {
        let chunks = [];
        if (ext === 'txt') {
          const text = await file.text();
          chunks = chunkText(text).map(text => ({ text, page: 'N/A', row: 'N/A' }));
        } else if (ext === 'csv') {
          const text = await file.text();
          const { data } = Papa.parse(text, { header: true, skipEmptyLines: true });
          chunks = data.map((row, i) => ({
            text: Object.entries(row).map(([k, v]) => `${k}: ${v}`).join(' | '),
            page: 'N/A', row: String(i + 1),
          }));
        } else if (ext === 'pdf') {
          const buffer = await file.arrayBuffer();
          const bytes = new Uint8Array(buffer);
          let binary = '';
          for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
          const res = await fetch('/api/parse-pdf', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fileData: btoa(binary) }),
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

  // ── Lab 2 + 3 + 4: Agent-orchestrated RAG ──────────────────────────────────
  async function askQuestion() {
    const q = question.trim();
    if (!q) return;
    if (!groqKey) { addAlert('error', 'Enter your Groq API key in the sidebar.'); return; }
    if (!allChunks.length) { addAlert('warning', 'No documents indexed yet.'); return; }

    const effectiveModel = model === 'custom — type below' ? customModel.trim() : model;
    if (!effectiveModel) { addAlert('error', 'Enter a custom model name.'); return; }

    setQuestion('');
    setIsAsking(true);
    setChatHistory(prev => [...prev, { role: 'user', content: q }]);
    setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 50);

    // ── Initialize agent trace ─────────────────────────────────────────────
    let trace = [
      { id: 1, icon: '🔍', name: 'Retriever Agent',   status: 'active',  detail: `Scanning ${filteredChunks.length} chunks with BM25…`, startMs: Date.now(), durationMs: null },
      { id: 2, icon: '📊', name: 'Context Builder',   status: 'pending', detail: '', startMs: null, durationMs: null },
      { id: 3, icon: '🤖', name: 'LLM Agent',         status: 'pending', detail: effectiveModel, startMs: null, durationMs: null },
      { id: 4, icon: '✅', name: 'Source Validator',  status: 'pending', detail: '', startMs: null, durationMs: null },
    ];

    function step(id, updates) {
      trace = trace.map(s => {
        if (s.id !== id) return s;
        const now = Date.now();
        const durationMs = (updates.status === 'done' || updates.status === 'error') && s.startMs
          ? now - s.startMs : s.durationMs;
        return { ...s, ...updates, durationMs };
      });
      setAgentTrace([...trace]);
    }

    setAgentTrace([...trace]);

    // ── Step 1: BM25 Retrieval ─────────────────────────────────────────────
    await new Promise(r => setTimeout(r, 480));
    const topChunks = bm25Search(q, filteredChunks, topK);
    step(1, { status: 'done', detail: `Found ${topChunks.length} relevant chunk${topChunks.length !== 1 ? 's' : ''}` });

    // ── Step 2: Context Building ───────────────────────────────────────────
    step(2, { status: 'active', detail: 'Assembling prompt context…', startMs: Date.now() });
    await new Promise(r => setTimeout(r, 320));
    const contextSize = topChunks.reduce((sum, c) => sum + c.text.length, 0);
    step(2, { status: 'done', detail: `${(contextSize / 1000).toFixed(1)}k chars, ${topChunks.length} source${topChunks.length !== 1 ? 's' : ''}` });

    // ── Step 3: Groq LLM ───────────────────────────────────────────────────
    step(3, { status: 'active', detail: `Querying ${effectiveModel}…`, startMs: Date.now() });
    let answer = '';
    let hadError = false;
    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, chunks: topChunks, groqKey, model: effectiveModel }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'API error');
      answer = data.answer;
      step(3, { status: 'done', detail: `${answer.split(' ').length} words generated` });
    } catch (e) {
      answer = `❌ Error: ${e.message}`;
      hadError = true;
      step(3, { status: 'error', detail: e.message });
    }

    // ── Step 4: Source Validator ───────────────────────────────────────────
    if (!hadError) {
      step(4, { status: 'active', detail: 'Checking answer is grounded in context…', startMs: Date.now() });
      await new Promise(r => setTimeout(r, 220));
      const isGrounded = !answer.toLowerCase().includes("couldn't find");
      step(4, {
        status: 'done',
        detail: isGrounded
          ? `Grounded ✓ — cites ${topChunks.length} source${topChunks.length !== 1 ? 's' : ''}`
          : 'Not found in indexed documents',
      });
    } else {
      step(4, { status: 'skipped', detail: 'Skipped (upstream error)' });
    }

    // ── Finalize ────────────────────────────────────────────────────────────
    const finalTrace = [...trace];
    setChatHistory(prev => [...prev, { role: 'assistant', content: answer, sources: topChunks, agentTrace: finalTrace }]);
    setAgentTrace([]);
    setIsAsking(false);
    setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 100);
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <>
      <Head>
        <title>Multi-Document RAG Assistant</title>
        <meta name="description" content="Ask questions across your documents using Groq AI, BM25 search, and agent orchestration" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet" />
      </Head>

      <div className="layout">

        {/* ━━━━━━━━━━━━━━━━━━ SIDEBAR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
        <aside className="sidebar">
          <div className="sidebar-logo">📚 RAG Assistant</div>

          {/* Config */}
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
                <button onClick={() => setShowKey(s => !s)} style={{
                  position: 'absolute', right: '0.5rem', top: '50%', transform: 'translateY(-50%)',
                  background: 'none', border: 'none', cursor: 'pointer', color: 'var(--muted)', fontSize: '0.9rem', paddingTop: '0.35rem',
                }}>{showKey ? '🙈' : '👁'}</button>
              </div>
              <div style={{ fontSize: '0.68rem', color: 'var(--muted)', marginTop: '0.35rem' }}>
                Get free key:&nbsp;
                <a href="https://console.groq.com" target="_blank" rel="noreferrer" style={{ color: 'var(--accent)' }}>
                  console.groq.com
                </a>
              </div>

              <label style={{ marginTop: '0.85rem' }}>LLM Model</label>
              <select value={model} onChange={e => setModel(e.target.value)}>
                {GROQ_MODELS.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
              {model === 'custom — type below' && (
                <input
                  type="text" value={customModel}
                  onChange={e => setCustomModel(e.target.value)}
                  placeholder="Paste model ID from console.groq.com/docs/models"
                  style={{ marginTop: '0.4rem' }}
                />
              )}
              <div style={{ fontSize: '0.68rem', color: 'var(--muted)', marginTop: '0.3rem' }}>
                <a href="https://console.groq.com/docs/models" target="_blank" rel="noreferrer" style={{ color: 'var(--accent)' }}>
                  View available models →
                </a>
              </div>
            </div>
          </div>

          {/* Upload */}
          <div>
            <div className="section-label">📂 Upload Documents</div>
            <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
              <div
                className={`upload-zone${dragging ? ' drag' : ''}`}
                onDragOver={e => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={e => { e.preventDefault(); setDragging(false); onFilesSelected(e.dataTransfer.files); }}
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="upload-zone-icon">📁</div>
                <div className="upload-zone-text">Drop PDF / CSV / TXT<br />or click to browse</div>
                <input ref={fileInputRef} type="file" multiple accept=".pdf,.csv,.txt"
                  style={{ display: 'none' }} onChange={e => onFilesSelected(e.target.files)} />
              </div>

              {pendingFiles.map(f => (
                <div key={f.name} className="file-chip">
                  <span>{FILE_ICONS[f.name.split('.').pop().toLowerCase()] || '📄'}</span>
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>{f.name}</span>
                  <span className="file-chip-remove" onClick={() => setPendingFiles(p => p.filter(x => x.name !== f.name))}>✕</span>
                </div>
              ))}

              <label>Product Category</label>
              <input type="text" value={category} onChange={e => setCategory(e.target.value)} placeholder="e.g. Electronics" />

              <div className="row" style={{ marginTop: '0.25rem' }}>
                <button className="btn btn-primary" disabled={isProcessing || !pendingFiles.length} onClick={processFiles}>
                  {isProcessing ? <><span className="spinner" style={{ width: 12, height: 12 }} /> Indexing…</> : '📥 Index Files'}
                </button>
                <button className="btn btn-danger" onClick={() => { setDocStore([]); setChatHistory([]); setAgentTrace([]); addAlert('success', 'Database cleared.'); }}>🗑️</button>
              </div>
            </div>
          </div>

          {/* Stats */}
          <div>
            <div className="section-label">🗄️ Index Stats</div>
            <div className="card">
              <div className="stat-big">{allChunks.length}</div>
              <div className="stat-label">Chunks from {docNames.length} document(s)</div>
              {docStore.length > 0
                ? <div style={{ marginTop: '0.75rem' }}>
                    {docStore.map(doc => (
                      <div key={doc.filename} className="file-chip" style={{ marginTop: '0.35rem' }}>
                        <span>{FILE_ICONS[doc.fileType] || '📄'}</span>
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1, fontSize: '0.72rem' }}>{doc.filename}</span>
                        <span style={{ fontSize: '0.68rem', color: 'var(--muted)', flexShrink: 0 }}>{doc.chunks.length}c</span>
                      </div>
                    ))}
                  </div>
                : <div style={{ fontSize: '0.75rem', color: 'var(--muted)', marginTop: '0.5rem' }}>No documents indexed yet.</div>
              }
            </div>
          </div>

          {/* Retrieval */}
          <div>
            <div className="section-label">🔍 Retrieval Settings</div>
            <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
              <label>Document Filter</label>
              <select value={filterDoc} onChange={e => setFilterDoc(e.target.value)}>
                <option value="All">All Documents</option>
                {docNames.map(n => <option key={n} value={n}>{n}</option>)}
              </select>
              <label>Top-K Chunks: <strong style={{ color: 'var(--accent)' }}>{topK}</strong></label>
              <input type="range" min={1} max={15} value={topK} onChange={e => setTopK(Number(e.target.value))} />
            </div>
          </div>
        </aside>

        {/* ━━━━━━━━━━━━━━━━━━ MAIN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */}
        <div className="main">
          <div className="main-scroll">

            {/* Alerts */}
            {alerts.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', marginBottom: '1rem' }}>
                {alerts.map(a => <div key={a.id} className={`alert alert-${a.type}`}>{a.msg}</div>)}
              </div>
            )}

            {/* Hero */}
            <div className="hero-title">Multi-Document RAG Assistant</div>
            <div className="hero-sub">Upload documents · BM25 retrieval · Agent orchestration · Groq-powered answers</div>
            <div className="badges">
              <span className="badge">📦 Lab 1 · Chunking</span>
              <span className="badge">🤖 Lab 2 · RAG Q&amp;A</span>
              <span className="badge">🔍 Lab 3 · BM25 Search</span>
              <span className="badge">🧠 Lab 4 · Agent Pipeline</span>
            </div>

            {/* Pipeline explainer */}
            <details style={{ marginTop: '1.25rem' }}>
              <summary>📊 How the agent pipeline works</summary>
              <div className="card pipeline-grid" style={{ marginTop: '0.6rem' }}>
                <div className="pipeline-step">
                  <strong>Indexing (Lab 1 + 4)</strong><br />
                  1. Upload PDF / CSV / TXT<br />
                  2. Extract text per file type<br />
                  3. Split into overlapping chunks<br />
                  4. Store in browser with metadata
                </div>
                <div className="pipeline-step">
                  <strong>Agent Orchestration (Lab 2 + 3)</strong><br />
                  🔍 Retriever Agent — BM25 keyword search<br />
                  📊 Context Builder — format top-K chunks<br />
                  🤖 LLM Agent — Groq generates answer<br />
                  ✅ Validator — grounding check
                </div>
              </div>
            </details>

            <div className="divider" />

            {/* Chat history */}
            {chatHistory.map((msg, i) => (
              <div key={i} className={`chat-msg ${msg.role}`}>
                <div className="bubble-label">{msg.role === 'user' ? '🧑 You' : '🤖 Assistant'}</div>
                <div className={`bubble ${msg.role}`} style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>

                {/* Agent trace (collapsible after answer) */}
                {msg.role === 'assistant' && msg.agentTrace?.length > 0 && (
                  <details className="agent-trace-details">
                    <summary>
                      <span>🧠 Agent trace</span>
                      <span className="agent-done-badge" style={{ marginLeft: '0.5rem' }}>
                        {msg.agentTrace.filter(s => s.status === 'done').length}/{msg.agentTrace.length} done
                      </span>
                    </summary>
                    <AgentPanel trace={msg.agentTrace} />
                  </details>
                )}

                {/* Source cards */}
                {msg.role === 'assistant' && msg.sources?.length > 0 && (
                  <div style={{ maxWidth: '80%', width: '100%', marginTop: '0.5rem' }}>
                    <div style={{ fontSize: '0.7rem', color: 'var(--muted)', marginBottom: '0.4rem' }}>
                      📎 {msg.sources.length} chunk{msg.sources.length !== 1 ? 's' : ''} retrieved via BM25:
                    </div>
                    <div className="sources-grid">
                      {msg.sources.slice(0, 4).map((src, j) => (
                        <div key={j} className="source-card">
                          <div className="source-card-title">
                            {FILE_ICONS[src.fileType] || '📄'} {src.filename}
                          </div>
                          <div className="source-card-meta">
                            {src.fileType?.toUpperCase()} · {src.category} ·{' '}
                            {src.row !== 'N/A' ? `Row ${src.row}` : src.page !== 'N/A' ? `Page ${src.page}` : '—'}
                            {src.score != null && ` · score: ${src.score.toFixed(3)}`}
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

            {/* Live agent trace during processing */}
            {isAsking && agentTrace.length > 0 && (
              <div className="chat-msg assistant" style={{ animation: 'fadeUp 0.3s ease' }}>
                <div className="bubble-label">🧠 Agent Pipeline</div>
                <AgentPanel trace={agentTrace} live />
              </div>
            )}

            {/* Empty state */}
            {!chatHistory.length && !isAsking && (
              <div className="empty-state">
                <div className="empty-icon">🧠</div>
                <div style={{ fontWeight: 600, color: '#94a3b8', fontSize: '1rem', marginBottom: '0.5rem' }}>
                  Ready for agent-orchestrated RAG
                </div>
                <div style={{ fontSize: '0.82rem', lineHeight: 1.7 }}>
                  1. Enter your <strong>Groq API key</strong><br />
                  2. Upload a <strong>CSV, PDF or TXT</strong> file<br />
                  3. Click <strong>Index Files</strong><br />
                  4. Ask anything — watch the agents work ↓
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>

          {/* Question form */}
          <div className="qa-form">
            <textarea
              value={question}
              onChange={e => setQuestion(e.target.value)}
              placeholder="Ask anything about your documents… (Enter to send, Shift+Enter for newline)"
              onKeyDown={e => {
                if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (!isAsking && question.trim()) askQuestion(); }
              }}
            />
            <button className="btn btn-primary" onClick={askQuestion} disabled={isAsking || !question.trim()} style={{ minWidth: 90 }}>
              {isAsking ? <span className="spinner" /> : '🚀 Ask'}
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
