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
  Check,
  Pin,
  PinOff,
  Filter,
  FileCode,
  FileSpreadsheet,
  X,
  ChevronRight,
  BookOpen,
  Hash
} from 'lucide-react';
import { cn } from '@/lib/utils';

export interface Citation {
  source: string;
  page: number;
  chunk_id: string;
  score: number;
  snippet: string;
}

export interface AgentStep {
  type: 'thought' | 'observation';
  iteration: number;
  content: string;
  tool?: string;
  tool_input?: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  steps?: AgentStep[];
  feedback?: number;
  isStreaming?: boolean;
  pinnedSourcesSnapshot?: string[];
}

export interface DocumentItem {
  name: string;
  title: string;
  type: string;
  chunks: number;
  pages: number;
}

const DEFAULT_DOCUMENTS: DocumentItem[] = [
  { name: 'attention_paper.pdf', title: 'Attention Is All You Need', type: 'pdf', chunks: 27, pages: 15 },
  { name: 'flash_attention.md', title: 'FlashAttention IO-Aware Tiling', type: 'md', chunks: 6, pages: 1 },
  { name: 'lora_paper.pdf', title: 'LoRA Low-Rank Adaptation', type: 'pdf', chunks: 66, pages: 26 },
  { name: 'rag_paper.pdf', title: 'Retrieval-Augmented Generation', type: 'pdf', chunks: 47, pages: 19 },
  { name: 'rag_system_architecture.html', title: 'Enterprise RAG Architecture', type: 'html', chunks: 1, pages: 1 }
];

const EXAMPLE_QUESTIONS = [
  { label: 'FlashAttention Tiling', text: 'What is FlashAttention and how does tiling help?' },
  { label: 'LoRA vs GPT-3 (Agent)', text: 'Calculate the ratio of parameters in LoRA vs GPT-3' },
  { label: 'Transformer Heads', text: 'What are the dimensions of d_model and heads in the base Transformer?' },
  { label: 'LoRA Adaptation', text: 'What is the mathematical formulation of LoRA weight adaptation?' },
  { label: 'Refusal Test', text: 'What is the return and refund policy for annual enterprise subscription plans?' }
];

export function AIChat7() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: "Hello! I am your research assistant equipped with **pinned sources & verified in-line citations**.\n\nYou can pin/unpin specific arXiv research papers and technical docs using the toolbar above to ground responses strictly on your selected context."
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedMode, setSelectedMode] = useState<string>('hybrid');
  const [documents, setDocuments] = useState<DocumentItem[]>(DEFAULT_DOCUMENTS);
  const [pinnedSources, setPinnedSources] = useState<string[]>(DEFAULT_DOCUMENTS.map(d => d.name));
  const [showPinnedDrawer, setShowPinnedDrawer] = useState(false);
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [copiedSnippet, setCopiedSnippet] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  const API_KEY = process.env.NEXT_PUBLIC_API_KEY || 'rag-prod-secret-key-change-me';

  // Load available documents from backend
  useEffect(() => {
    async function fetchDocuments() {
      try {
        const res = await fetch(`${API_URL}/documents`, {
          headers: { 'x-api-key': API_KEY }
        });
        if (res.ok) {
          const data = await res.json();
          if (data.documents && data.documents.length > 0) {
            setDocuments(data.documents);
            setPinnedSources(prev => prev.length > 0 ? prev : data.documents.map((d: DocumentItem) => d.name));
          }
        }
      } catch (e) {
        // Fallback to default documents
      }
    }
    fetchDocuments();
  }, [API_URL, API_KEY]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const togglePin = (sourceName: string) => {
    setPinnedSources(prev =>
      prev.includes(sourceName) ? prev.filter(s => s !== sourceName) : [...prev, sourceName]
    );
  };

  const pinAll = () => setPinnedSources(documents.map(d => d.name));
  const clearPins = () => setPinnedSources([]);

  const handleSend = async (questionText?: string) => {
    const q = (questionText || input).trim();
    if (!q || loading) return;

    setInput('');
    const userMsgId = `usr-${Date.now()}`;
    const botMsgId = `bot-${Date.now()}`;

    setMessages(prev => [
      ...prev,
      { id: userMsgId, role: 'user', content: q },
      {
        id: botMsgId,
        role: 'assistant',
        content: '',
        citations: [],
        steps: [],
        isStreaming: true,
        pinnedSourcesSnapshot: [...pinnedSources]
      }
    ]);
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': API_KEY,
        },
        body: JSON.stringify({
          question: q,
          top_k: 5,
          mode: selectedMode,
          filters: pinnedSources.length > 0 && pinnedSources.length < documents.length
            ? { sources: pinnedSources }
            : null
        }),
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
          comment: rating > 0 ? "Thumbs up (Grounded)" : "Thumbs down (Ungrounded)"
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
    setUploadStatus("Uploading & indexing document into Qdrant...");

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
        setUploadStatus(`Indexed ${data.processed_files} file(s), ${data.chunks_indexed} chunk(s) successfully!`);
        // Refresh document list
        const docRes = await fetch(`${API_URL}/documents`, { headers: { 'x-api-key': API_KEY } });
        if (docRes.ok) {
          const docData = await docRes.json();
          setDocuments(docData.documents);
          setPinnedSources(prev => [...prev, file.name]);
        }
      } else {
        setUploadStatus("Upload failed. Verify server status.");
      }
    } catch (err) {
      setUploadStatus("Upload failed due to network error.");
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

  const getDocTypeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'pdf':
        return <FileText className="w-3.5 h-3.5 text-rose-600" />;
      case 'md':
        return <FileCode className="w-3.5 h-3.5 text-sky-600" />;
      case 'html':
        return <BookOpen className="w-3.5 h-3.5 text-amber-600" />;
      default:
        return <FileSpreadsheet className="w-3.5 h-3.5 text-slate-600" />;
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-8.5rem)] max-w-7xl mx-auto w-full">
      {/* AI Chat 7 Top Controls & Pinned Sources Toolbar */}
      <div className="mb-3 space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2 bg-white border border-slate-200/90 rounded-xl px-3 py-2 shadow-2xs">
          {/* Left: Pinned Sources Trigger & Pipeline Mode */}
          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={() => setShowPinnedDrawer(!showPinnedDrawer)}
              className={cn(
                "inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold rounded-lg border transition-all",
                pinnedSources.length > 0
                  ? "bg-emerald-50 text-emerald-800 border-emerald-300 hover:bg-emerald-100"
                  : "bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100"
              )}
            >
              <Pin className={cn("w-3.5 h-3.5", pinnedSources.length > 0 ? "fill-emerald-600 text-emerald-600" : "text-slate-400")} />
              <span>Pinned Sources ({pinnedSources.length}/{documents.length})</span>
            </button>

            <div className="flex items-center gap-1.5 pl-2 border-l border-slate-200">
              <Layers className="w-3.5 h-3.5 text-slate-400" />
              <label htmlFor="pipeline-mode" className="text-xs text-slate-500 font-medium hidden sm:inline">Mode:</label>
              <select
                id="pipeline-mode"
                value={selectedMode}
                onChange={(e) => setSelectedMode(e.target.value)}
                className="text-xs bg-slate-50 hover:bg-slate-100 border border-slate-200 text-slate-800 rounded-lg px-2 py-1 font-medium focus:ring-2 focus:ring-emerald-500 focus:outline-hidden"
              >
                <option value="hybrid">⚡ Hybrid RRF (BM25 + Dense)</option>
                <option value="dense">🎯 Dense Vector (BGE-Small)</option>
                <option value="rerank">⚖️ Cross-Encoder Rerank</option>
                <option value="rewrite">🔍 Query Rewriting</option>
                <option value="agent">🤖 ReAct Autonomous Agent</option>
                <option value="auto">🌐 Auto Intent Router</option>
              </select>
            </div>
          </div>

          {/* Right: Actions */}
          <div className="flex items-center gap-2">
            <label className="cursor-pointer text-xs font-medium text-emerald-700 hover:text-emerald-800 bg-emerald-50 hover:bg-emerald-100 px-2.5 py-1 rounded-lg border border-emerald-200 transition-colors flex items-center gap-1">
              <Upload className="w-3.5 h-3.5" />
              <span>Upload Doc</span>
              <input type="file" className="hidden" accept=".pdf,.md,.html,.txt" onChange={handleFileUpload} disabled={uploading} />
            </label>
            <button
              onClick={() => setMessages([{ id: 'welcome', role: 'assistant', content: "Chat reset. How can I assist you with the pinned sources?" }])}
              className="p-1 text-slate-400 hover:text-slate-600 rounded-lg hover:bg-slate-100 transition-colors"
              title="Reset Chat Session"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Pinned Sources Slide-Out Bar */}
        {showPinnedDrawer && (
          <div className="bg-slate-50 border border-slate-200/90 rounded-xl p-3 shadow-inner animate-in fade-in duration-150">
            <div className="flex items-center justify-between mb-2 pb-1.5 border-b border-slate-200">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-700">
                <Filter className="w-3.5 h-3.5 text-emerald-600" />
                <span>Knowledge Base Filter (Grounded Context Restriction)</span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <button onClick={pinAll} className="text-emerald-700 hover:underline font-medium">Pin All</button>
                <span className="text-slate-300">|</span>
                <button onClick={clearPins} className="text-slate-500 hover:underline">Clear Pins</button>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2">
              {documents.map((doc) => {
                const isPinned = pinnedSources.includes(doc.name);
                return (
                  <button
                    key={doc.name}
                    onClick={() => togglePin(doc.name)}
                    className={cn(
                      "text-left p-2 rounded-lg border text-xs flex items-center justify-between transition-all",
                      isPinned
                        ? "bg-white border-emerald-300 text-slate-900 shadow-2xs ring-1 ring-emerald-200"
                        : "bg-slate-100/60 border-slate-200 text-slate-400 hover:bg-slate-100"
                    )}
                  >
                    <div className="flex items-center gap-2 truncate pr-2">
                      {getDocTypeIcon(doc.type)}
                      <div className="truncate">
                        <p className={cn("font-medium truncate", isPinned ? "text-slate-900" : "text-slate-500")}>
                          {doc.title || doc.name}
                        </p>
                        <p className="text-[10px] text-slate-400 font-mono">
                          {doc.chunks} chunks • {doc.pages} page{doc.pages > 1 ? 's' : ''}
                        </p>
                      </div>
                    </div>
                    {isPinned ? (
                      <Pin className="w-3.5 h-3.5 text-emerald-600 fill-emerald-600 flex-shrink-0" />
                    ) : (
                      <PinOff className="w-3.5 h-3.5 text-slate-300 flex-shrink-0" />
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* Upload Notification Banner */}
        {uploadStatus && (
          <div className="p-2 text-xs bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-lg flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
            <span>{uploadStatus}</span>
          </div>
        )}
      </div>

      {/* Main Chat Canvas with Pinned Sources Badges & Citations */}
      <div className="flex-1 overflow-y-auto bg-white border border-slate-200 rounded-2xl p-4 shadow-sm space-y-4">
        {/* Quick Example Starters (when conversation is short) */}
        {messages.length <= 1 && (
          <div className="my-2 p-3 bg-slate-50/70 border border-slate-200/80 rounded-xl space-y-2">
            <p className="text-xs font-semibold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-emerald-600" />
              Verified Prompt Starters
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
              {EXAMPLE_QUESTIONS.map((ex, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSend(ex.text)}
                  disabled={loading}
                  className="text-left text-xs bg-white hover:bg-emerald-50/60 border border-slate-200 hover:border-emerald-300 text-slate-700 p-2.5 rounded-lg transition-all shadow-2xs disabled:opacity-50"
                >
                  <span className="font-semibold text-emerald-800 block text-[11px] mb-0.5">{ex.label}</span>
                  <span className="text-slate-600 truncate block">{ex.text}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Message Thread */}
        {messages.map(msg => (
          <div
            key={msg.id}
            className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
          >
            <div
              className={cn(
                "max-w-[88%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
                msg.role === 'user'
                  ? 'bg-emerald-600 text-white rounded-br-none shadow-xs'
                  : 'bg-slate-50 text-slate-900 border border-slate-200/80 rounded-bl-none shadow-2xs'
              )}
            >
              {/* Agent Reasoning Trace Display */}
              {msg.steps && msg.steps.length > 0 && (
                <details className="mb-3 rounded-xl border border-indigo-100 bg-indigo-50/40 p-2.5 text-xs text-slate-700 group" open>
                  <summary className="font-semibold text-indigo-900 cursor-pointer flex items-center gap-1.5 select-none hover:text-indigo-950">
                    <Cpu className="w-3.5 h-3.5 text-indigo-600" />
                    <span>ReAct Reasoning Trace ({msg.steps.length} event{msg.steps.length > 1 ? 's' : ''})</span>
                  </summary>
                  <div className="mt-2.5 space-y-2 pl-2 border-l-2 border-indigo-400 text-xs">
                    {msg.steps.map((st, i) => (
                      <div key={i} className="space-y-1">
                        {st.type === 'thought' ? (
                          <div className="text-slate-800">
                            <span className="font-semibold text-indigo-700">Step {st.iteration} Thought:</span> {st.content}
                            {st.tool && (
                              <div className="mt-0.5 text-slate-600 font-mono text-[11px] bg-white px-2 py-0.5 rounded border border-indigo-200 inline-block">
                                &rarr; Action: <strong className="text-indigo-900">{st.tool}</strong>({st.tool_input || '""'})
                              </div>
                            )}
                          </div>
                        ) : (
                          <div className="bg-white p-2 rounded-lg border border-slate-200 font-mono text-[11px] text-slate-600 max-h-32 overflow-y-auto">
                            <span className="font-semibold text-emerald-700 font-sans block text-xs">Observation ({st.tool}):</span>
                            {st.content}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </details>
              )}

              {/* Message Content */}
              <div className="whitespace-pre-wrap">{msg.content || (msg.isStreaming ? "Retrieving evidence & generating answer..." : "")}</div>

              {/* In-line Citations Chips Bar */}
              {msg.citations && msg.citations.length > 0 && (
                <div className="mt-3 pt-3 border-t border-slate-200/80">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1">
                      <BookOpen className="w-3 h-3 text-emerald-600" />
                      Pinned Citations & Source Evidence:
                    </span>
                    <span className="text-[10px] text-slate-400">Click to preview chunk</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {msg.citations.map((c, i) => (
                      <button
                        key={i}
                        onClick={() => setSelectedCitation(c)}
                        className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded-lg bg-white hover:bg-emerald-50 text-emerald-800 border border-emerald-300 transition-all shadow-2xs hover:border-emerald-400 hover:shadow-xs"
                      >
                        <FileText className="w-3 h-3 text-emerald-600" />
                        <span className="font-semibold">{c.source}</span>
                        <span className="text-emerald-600/70">(p. {c.page})</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Assistant Response Feedback Toolbar */}
            {msg.role === 'assistant' && msg.id !== 'welcome' && !msg.isStreaming && (
              <div className="flex items-center gap-2 mt-1.5 text-xs text-slate-400 pl-1">
                <span>Grounded on pinned context?</span>
                <button
                  onClick={() => handleFeedback(msg.id, 1)}
                  className={cn("p-1 rounded hover:bg-slate-100 transition-colors", msg.feedback === 1 && "text-emerald-600 font-bold")}
                  title="Thumbs up (Grounded & Accurate)"
                >
                  <ThumbsUp className="w-3.5 h-3.5" />
                </button>
                <button
                  onClick={() => handleFeedback(msg.id, -1)}
                  className={cn("p-1 rounded hover:bg-slate-100 transition-colors", msg.feedback === -1 && "text-rose-600 font-bold")}
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

      {/* AI Chat 7 Input Canvas */}
      <form
        onSubmit={e => {
          e.preventDefault();
          handleSend();
        }}
        className="mt-3 bg-white border border-slate-300 rounded-2xl p-2 shadow-xs transition-all focus-within:ring-2 focus-within:ring-emerald-500 focus-within:border-transparent"
      >
        <div className="flex items-center justify-between px-2 pb-1 text-[11px] text-slate-400 border-b border-slate-100">
          <div className="flex items-center gap-1.5">
            <Pin className="w-3 h-3 text-emerald-600" />
            <span>Grounded on: <strong className="text-slate-600">{pinnedSources.length} pinned source(s)</strong></span>
          </div>
          <span className="font-mono text-[10px]">Mode: {selectedMode.toUpperCase()}</span>
        </div>

        <div className="flex items-center gap-2 mt-1">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask a technical question over pinned research papers..."
            disabled={loading}
            className="flex-1 bg-transparent px-3 py-2 text-sm focus:outline-hidden disabled:bg-slate-50 text-slate-900 placeholder-slate-400"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="bg-emerald-600 hover:bg-emerald-700 text-white font-medium px-4 py-2 rounded-xl transition-all shadow-xs flex items-center gap-1.5 disabled:opacity-50 disabled:cursor-not-allowed text-xs"
          >
            <Send className="w-3.5 h-3.5" />
            <span>Send</span>
          </button>
        </div>
      </form>

      {/* Citation Slide-Over / Drawer Modal */}
      {selectedCitation && (
        <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl max-w-xl w-full p-6 shadow-2xl border border-slate-200 animate-in fade-in zoom-in-95 duration-150">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100">
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-emerald-600" />
                <h3 className="font-semibold text-slate-900">{selectedCitation.source}</h3>
                <span className="text-xs bg-emerald-100 text-emerald-800 px-2.5 py-0.5 rounded-full font-semibold">
                  Page {selectedCitation.page}
                </span>
              </div>
              <button
                onClick={() => setSelectedCitation(null)}
                className="text-slate-400 hover:text-slate-700 text-sm font-bold p-1 rounded-md"
              >
                ✕
              </button>
            </div>

            <div className="my-4 space-y-3">
              <div className="flex items-center justify-between text-xs text-slate-500">
                <span className="flex items-center gap-1 font-mono">
                  <Hash className="w-3 h-3 text-slate-400" />
                  Chunk ID: <code className="bg-slate-100 px-1 py-0.5 rounded text-slate-700">{selectedCitation.chunk_id}</code>
                </span>
                <span>Retrieval Score: <strong className="text-emerald-700 font-mono">{selectedCitation.score}</strong></span>
              </div>

              <div className="bg-slate-50 p-4 rounded-xl border border-slate-200/80 text-xs leading-relaxed text-slate-800 max-h-64 overflow-y-auto whitespace-pre-wrap font-sans">
                {selectedCitation.snippet}
              </div>
            </div>

            <div className="flex items-center justify-between pt-3 border-t border-slate-100">
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
                className="px-4 py-2 text-xs font-semibold bg-slate-900 text-white rounded-xl hover:bg-slate-800 transition-colors shadow-2xs"
              >
                Close Drawer
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
export default AIChat7;
