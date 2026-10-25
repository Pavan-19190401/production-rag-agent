'use client';

import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Upload,
  ThumbsUp,
  ThumbsDown,
  FileText,
  CheckCircle2,
  AlertCircle,
  ExternalLink,
  Sparkles,
  HelpCircle,
  RotateCcw,
  Cpu,
  Layers,
  Copy,
  Check
} from 'lucide-react';

interface Citation {
  source: string;
  page: number;
  chunk_id: string;
  score: number;
  snippet: string;
}

interface AgentStep {
  type: 'thought' | 'observation';
  iteration: number;
  content: string;
  tool?: string;
  tool_input?: string;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  steps?: AgentStep[];
  feedback?: number; // 1 for up, -1 for down
  isStreaming?: boolean;
}

const EXAMPLE_QUESTIONS = [
  "What is FlashAttention and how does tiling help?",
  "Calculate the ratio of parameters in LoRA vs GPT-3",
  "What are the dimensions of d_model and heads in the base Transformer?",
  "What is the mathematical formulation of LoRA weight adaptation?",
  "What is the return and refund policy for annual enterprise subscription plans?"
];

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: "Hello! I am your production research assistant grounded strictly on arXiv papers and technical documentation (Attention, RAG, LoRA, FlashAttention). Ask me a technical question, calculate a parameter ratio, or select an example below!"
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedMode, setSelectedMode] = useState<string>('hybrid');
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [copiedSnippet, setCopiedSnippet] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  const API_KEY = process.env.NEXT_PUBLIC_API_KEY || 'rag-prod-secret-key-change-me';

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async (questionText?: string) => {
    const q = (questionText || input).trim();
    if (!q || loading) return;

    setInput('');
    const userMsgId = `usr-${Date.now()}`;
    const botMsgId = `bot-${Date.now()}`;

    setMessages(prev => [
      ...prev,
      { id: userMsgId, role: 'user', content: q },
      { id: botMsgId, role: 'assistant', content: '', citations: [], steps: [], isStreaming: true }
    ]);
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': API_KEY,
        },
        body: JSON.stringify({ question: q, top_k: 5, mode: selectedMode }),
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      if (!response.body) {
        throw new Error('ReadableStream not supported.');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let accumulatedAnswer = '';
      let incomingCitations: Citation[] = [];
      let incomingSteps: AgentStep[] = [];

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunkText = decoder.decode(value, { stream: true });
        const lines = chunkText.split('\n');

        for (let i = 0; i < lines.length; i++) {
          const line = lines[i].trim();

          if (line.startsWith('event: thought')) {
            if (i + 1 < lines.length && lines[i + 1].startsWith('data:')) {
              try {
                const thoughtPayload = JSON.parse(lines[i + 1].replace('data:', '').trim());
                incomingSteps.push({
                  type: 'thought',
                  iteration: thoughtPayload.iteration || incomingSteps.length + 1,
                  content: thoughtPayload.thought || thoughtPayload.content || '',
                  tool: thoughtPayload.tool,
                  tool_input: thoughtPayload.tool_input
                });
                setMessages(prev =>
                  prev.map(m => m.id === botMsgId ? { ...m, steps: [...incomingSteps] } : m)
                );
              } catch (e) {
                console.error("Error parsing thought step", e);
              }
              i++;
            }
          } else if (line.startsWith('event: observation')) {
            if (i + 1 < lines.length && lines[i + 1].startsWith('data:')) {
              try {
                const obsPayload = JSON.parse(lines[i + 1].replace('data:', '').trim());
                incomingSteps.push({
                  type: 'observation',
                  iteration: obsPayload.iteration || incomingSteps.length,
                  content: obsPayload.content || '',
                  tool: obsPayload.tool
                });
                setMessages(prev =>
                  prev.map(m => m.id === botMsgId ? { ...m, steps: [...incomingSteps] } : m)
                );
              } catch (e) {
                console.error("Error parsing observation step", e);
              }
              i++;
            }
          } else if (line.startsWith('event: sources')) {
            if (i + 1 < lines.length && lines[i + 1].startsWith('data:')) {
              try {
                const sourcesJson = lines[i + 1].replace('data:', '').trim();
                incomingCitations = JSON.parse(sourcesJson);
                setMessages(prev =>
                  prev.map(m => m.id === botMsgId ? { ...m, citations: incomingCitations } : m)
                );
              } catch (e) {
                console.error("Error parsing citations", e);
              }
              i++;
            }
          } else if (line.startsWith('data:')) {
            const dataStr = line.replace('data:', '').trim();
            if (dataStr === '[DONE]') continue;

            try {
              const payload = JSON.parse(dataStr);
              if (payload.token) {
                accumulatedAnswer += payload.token;
                setMessages(prev =>
                  prev.map(m => m.id === botMsgId ? { ...m, content: accumulatedAnswer } : m)
                );
              }
            } catch {
              if (dataStr) {
                accumulatedAnswer += dataStr;
                setMessages(prev =>
                  prev.map(m => m.id === botMsgId ? { ...m, content: accumulatedAnswer } : m)
                );
              }
            }
          }
        }
      }

      setMessages(prev =>
        prev.map(m => m.id === botMsgId ? { ...m, isStreaming: false } : m)
      );
    } catch (err: any) {
      setMessages(prev =>
        prev.map(m =>
          m.id === botMsgId
            ? { ...m, content: `Error: Unable to complete query (${err.message}). Verify API service is running.`, isStreaming: false }
            : m
        )
      );
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async (messageId: string, rating: number) => {
    setMessages(prev =>
      prev.map(m => m.id === messageId ? { ...m, feedback: rating } : m)
    );

    try {
      await fetch(`${API_URL}/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': API_KEY,
        },
        body: JSON.stringify({
          request_id: messageId,
          rating: rating,
          comment: rating > 0 ? "Thumbs up" : "Thumbs down"
        }),
      });
    } catch (e) {
      console.error("Feedback error", e);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadStatus("Uploading & indexing document...");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}/ingest`, {
        method: 'POST',
        headers: { 'x-api-key': API_KEY },
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        setUploadStatus(`Indexed successfully! Processed ${data.processed_files} files, ${data.chunks_indexed} chunks.`);
      } else {
        setUploadStatus("Upload failed. Verify server status.");
      }
    } catch (err) {
      setUploadStatus("Upload failed due to connection error.");
    } finally {
      setUploading(false);
      setTimeout(() => setUploadStatus(null), 5000);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedSnippet(true);
    setTimeout(() => setCopiedSnippet(false), 2000);
  };

  return (
    <div className="flex flex-col h-[calc(100vh-8.5rem)]">
      {/* Top Banner & Pipeline Controls */}
      <div className="mb-4 space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2 pb-1">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
              Verified Prompts
            </span>
            <div className="flex items-center gap-1.5 pl-2 border-l border-slate-200">
              <Layers className="w-3.5 h-3.5 text-slate-400" />
              <label htmlFor="mode-select" className="text-xs text-slate-600 font-medium">Pipeline Mode:</label>
              <select
                id="mode-select"
                value={selectedMode}
                onChange={(e) => setSelectedMode(e.target.value)}
                className="text-xs bg-white border border-slate-200 text-slate-800 rounded-md px-2 py-1 focus:ring-2 focus:ring-emerald-500 focus:outline-hidden font-medium shadow-2xs"
              >
                <option value="hybrid">⚡ Hybrid Search (BM25 + Dense RRF)</option>
                <option value="dense">🎯 Dense Vector (BGE-Small)</option>
                <option value="rerank">⚖️ Cross-Encoder Reranker</option>
                <option value="rewrite">🔍 Query Rewriter</option>
                <option value="agent">🤖 ReAct Autonomous Agent</option>
                <option value="auto">🌐 Auto Intent Router</option>
              </select>
            </div>
          </div>

          <label className="cursor-pointer text-xs font-medium text-emerald-700 hover:text-emerald-800 bg-emerald-50 px-2.5 py-1 rounded-md border border-emerald-200 transition-colors flex items-center gap-1 shadow-2xs">
            <Upload className="w-3.5 h-3.5" />
            <span>Upload Document</span>
            <input type="file" className="hidden" accept=".pdf,.md,.html,.txt" onChange={handleFileUpload} disabled={uploading} />
          </label>
        </div>

        {uploadStatus && (
          <div className="p-2 text-xs bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-md flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
            <span>{uploadStatus}</span>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
          {EXAMPLE_QUESTIONS.map((q, idx) => (
            <button
              key={idx}
              onClick={() => handleSend(q)}
              disabled={loading}
              className="text-left text-xs bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 p-2.5 rounded-lg transition-all shadow-2xs hover:border-slate-300 disabled:opacity-50 truncate"
            >
              {idx === 4 ? "🚫 Refusal Test: " : idx === 1 ? "🤖 Agent Test: " : "💡 "}{q}
            </button>
          ))}
        </div>
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto bg-white border border-slate-200 rounded-xl p-4 shadow-sm space-y-4">
        {messages.map(msg => (
          <div
            key={msg.id}
            className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
          >
            <div
              className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                msg.role === 'user'
                  ? 'bg-emerald-600 text-white rounded-br-none'
                  : 'bg-slate-100 text-slate-900 border border-slate-200/60 rounded-bl-none'
              }`}
            >
              {/* ReAct Agent Reasoning Steps Display */}
              {msg.steps && msg.steps.length > 0 && (
                <details className="mb-3 rounded-lg border border-slate-300/80 bg-slate-50/90 p-2.5 text-xs text-slate-700 group" open>
                  <summary className="font-semibold text-emerald-800 cursor-pointer flex items-center gap-1.5 select-none hover:text-emerald-900">
                    <Cpu className="w-3.5 h-3.5 text-emerald-600" />
                    <span>Agent Reasoning Trace ({msg.steps.length} event{msg.steps.length > 1 ? 's' : ''})</span>
                  </summary>
                  <div className="mt-2.5 space-y-2 pl-2 border-l-2 border-emerald-500 text-xs">
                    {msg.steps.map((st, i) => (
                      <div key={i} className="space-y-1">
                        {st.type === 'thought' ? (
                          <div className="text-slate-800">
                            <span className="font-semibold text-indigo-700">Step {st.iteration} Thought:</span> {st.content}
                            {st.tool && (
                              <div className="mt-0.5 text-slate-500 font-mono text-[11px] bg-slate-100 px-1.5 py-0.5 rounded inline-block">
                                &rarr; Action: <strong className="text-slate-800">{st.tool}</strong>({st.tool_input || '""'})
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="bg-white p-2 rounded border border-slate-200 font-mono text-[11px] text-slate-600 max-h-32 overflow-y-auto">
                            <span className="font-semibold text-emerald-700 font-sans block text-xs">Observation ({st.tool}):</span>
                            {st.content}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </details>
              )}

              <div className="whitespace-pre-wrap">{msg.content || (msg.isStreaming ? "Retrieving evidence & generating..." : "")}</div>

              {/* Citations Chips */}
              {msg.citations && msg.citations.length > 0 && (
                <div className="mt-3 pt-3 border-t border-slate-200/80">
                  <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block mb-1.5">
                    Verified Citations:
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {msg.citations.map((c, i) => (
                      <button
                        key={i}
                        onClick={() => setSelectedCitation(c)}
                        className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded-md bg-white hover:bg-emerald-50 text-emerald-800 border border-emerald-300 transition-colors shadow-2xs"
                      >
                        <FileText className="w-3 h-3 text-emerald-600" />
                        <span>{c.source} (p. {c.page})</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Thumbs Feedback for Assistant */}
            {msg.role === 'assistant' && msg.id !== 'welcome' && !msg.isStreaming && (
              <div className="flex items-center gap-2 mt-1.5 text-xs text-slate-400 pl-1">
                <span>Was this response grounded?</span>
                <button
                  onClick={() => handleFeedback(msg.id, 1)}
                  className={`p-1 rounded hover:bg-slate-100 transition-colors ${msg.feedback === 1 ? 'text-emerald-600 font-bold' : ''}`}
                  title="Thumbs up (Grounded & Accurate)"
                >
                  <ThumbsUp className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => handleFeedback(msg.id, -1)}
                  className={`p-1 rounded hover:bg-slate-100 transition-colors ${msg.feedback === -1 ? 'text-rose-600 font-bold' : ''}`}
                  title="Thumbs down (Incorrect or Hallucinated)"
                >
                  <ThumbsDown className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Query Input Bar */}
      <form
        onSubmit={e => {
          e.preventDefault();
          handleSend();
        }}
        className="mt-3 flex items-center gap-2"
      >
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder={`Ask a technical question over research papers using ${selectedMode.toUpperCase()} mode...`}
          disabled={loading}
          className="flex-1 bg-white border border-slate-300 rounded-xl px-4 py-3 text-sm focus:outline-hidden focus:ring-2 focus:ring-emerald-500 focus:border-transparent transition-all shadow-xs disabled:bg-slate-50"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="bg-emerald-600 hover:bg-emerald-700 text-white font-medium px-5 py-3 rounded-xl transition-all shadow-sm flex items-center gap-1.5 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Send className="w-4 h-4" />
          <span>Send</span>
        </button>
      </form>

      {/* Citation Passage Modal / Drawer */}
      {selectedCitation && (
        <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl max-w-xl w-full p-6 shadow-2xl border border-slate-200 animate-in fade-in zoom-in-95 duration-150">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-emerald-600" />
                <h3 className="font-semibold text-slate-900">{selectedCitation.source}</h3>
                <span className="text-xs bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded-full font-medium">
                  Page {selectedCitation.page}
                </span>
              </div>
              <button
                onClick={() => setSelectedCitation(null)}
                className="text-slate-400 hover:text-slate-700 text-sm font-bold"
              >
                ✕
              </button>
            </div>

            <div className="my-4">
              <div className="flex items-center justify-between text-xs text-slate-500 mb-2">
                <span>Chunk ID: <code className="bg-slate-100 px-1 py-0.5 rounded">{selectedCitation.chunk_id}</code></span>
                <span>Retrieval Score: <strong className="text-slate-700">{selectedCitation.score}</strong></span>
              </div>
              <div className="bg-slate-50 p-4 rounded-xl border border-slate-200/80 text-xs leading-relaxed text-slate-800 max-h-60 overflow-y-auto whitespace-pre-wrap">
                {selectedCitation.snippet}
              </div>
            </div>

            <div className="flex items-center justify-between pt-2 border-t border-slate-100">
              <button
                onClick={() => copyToClipboard(selectedCitation.snippet)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 hover:text-slate-900 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors"
              >
                {copiedSnippet ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-600" />
                    <span>Copied!</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" />
                    <span>Copy Snippet</span>
                  </>
                )}
              </button>

              <button
                onClick={() => setSelectedCitation(null)}
                className="px-4 py-2 text-xs font-medium bg-slate-900 text-white rounded-lg hover:bg-slate-800 transition-colors"
              >
                Close Preview
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
