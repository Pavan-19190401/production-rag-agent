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
  BookOpen,
  Hash,
  Plus,
  ArrowUp,
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  Download
} from 'lucide-react';
import { cn } from '@/lib/utils';
import LatticeLoader from './LatticeLoader';

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
  status?: 'working' | 'done' | 'error';
  elapsedSeconds?: number;
}

export interface DocumentItem {
  id: number;
  name: string;
  title: string;
  type: string;
  subtitle: string;
  chunks: number;
  pages: number;
}

const DEFAULT_DOCUMENTS: DocumentItem[] = [
  { id: 1, name: 'attention_paper.pdf', title: 'attention_paper.pdf', subtitle: 'PDF 15 pages', type: 'pdf', chunks: 27, pages: 15 },
  { id: 2, name: 'flash_attention.md', title: 'flash_attention.md', subtitle: 'Note 1,400 words', type: 'md', chunks: 6, pages: 1 },
  { id: 3, name: 'lora_paper.pdf', title: 'lora_paper.pdf', subtitle: 'PDF 26 pages', type: 'pdf', chunks: 66, pages: 26 },
  { id: 4, name: 'rag_paper.pdf', title: 'rag_paper.pdf', subtitle: 'PDF 19 pages', type: 'pdf', chunks: 47, pages: 19 },
  { id: 5, name: 'rag_system_architecture.html', title: 'rag_system_architecture.html', subtitle: 'Doc Production Arch', type: 'html', chunks: 1, pages: 1 }
];

const SUGGESTED_PROMPTS = [
  "What is FlashAttention and how does tiling help?",
  "Calculate the ratio of parameters in LoRA vs GPT-3",
  "What are the dimensions of d_model and heads in Transformer?",
  "What is the mathematical formulation of LoRA weight adaptation?",
  "What is the return and refund policy for annual enterprise subscription plans?"
];

export function AIChat7() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: "Welcome! I am your research assistant grounded on the arXiv research papers and architecture specifications pinned on the left. [1] All answers include verified in-line citations mapped to your pinned sources. [2]\n\nAsk a question or select a prompt below to get started.",
      status: 'done',
      elapsedSeconds: 0.3,
      citations: [
        {
          source: 'attention_paper.pdf',
          page: 1,
          chunk_id: 'attention_paper.pdf_p1_Abstract_c0',
          score: 0.033,
          snippet: 'Attention Is All You Need: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.'
        },
        {
          source: 'flash_attention.md',
          page: 1,
          chunk_id: 'flash_attention.md_p1_Abstract_c0',
          score: 0.032,
          snippet: 'FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness using tiling between SRAM and HBM.'
        }
      ]
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedMode, setSelectedMode] = useState<string>('hybrid');
  const [documents, setDocuments] = useState<DocumentItem[]>(DEFAULT_DOCUMENTS);
  const [pinnedSources, setPinnedSources] = useState<string[]>(DEFAULT_DOCUMENTS.map(d => d.name));
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [copiedSnippet, setCopiedSnippet] = useState(false);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [speakingId, setSpeakingId] = useState<string | null>(null);
  const recognitionRef = useRef<any>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  };
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Voice Input (Web Speech Recognition)
  const toggleListening = () => {
    if (typeof window === 'undefined') return;
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Speech recognition is not supported in this browser. Please use Chrome, Edge, or Safari.");
      return;
    }

    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      setIsListening(true);
    };

    recognition.onresult = (event: any) => {
      const transcript = event.results[0]?.[0]?.transcript;
      if (transcript) {
        setInput(prev => (prev ? `${prev} ${transcript}` : transcript));
      }
      setIsListening(false);
    };

    recognition.onerror = () => {
      setIsListening(false);
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognitionRef.current = recognition;
    recognition.start();
  };

  // Text-to-Speech (Read Aloud)
  const toggleSpeech = (id: string, text: string) => {
    if (typeof window === 'undefined' || !window.speechSynthesis) return;

    if (speakingId === id) {
      window.speechSynthesis.cancel();
      setSpeakingId(null);
      return;
    }

    window.speechSynthesis.cancel();
    const cleanText = text.replace(/\[\d+\]/g, '').replace(/\[.*?page \d+\]/g, '');
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.0;
    utterance.onend = () => setSpeakingId(null);
    utterance.onerror = () => setSpeakingId(null);

    setSpeakingId(id);
    window.speechSynthesis.speak(utterance);
  };

  // Export Chat as Markdown
  const exportChat = () => {
    if (messages.length === 0) return;
    let md = `# Research Assistant Session - ${new Date().toLocaleString()}\n\n`;
    messages.forEach((m) => {
      const role = m.role === 'user' ? '### 👤 User' : '### 🤖 Grounded Assistant';
      md += `${role}\n\n${m.content}\n\n`;
      if (m.citations && m.citations.length > 0) {
        md += `**Cited Sources:**\n`;
        m.citations.forEach((c, cIdx) => {
          md += `- [${cIdx + 1}] \`${c.source}\` (Page ${c.page}) - Score: ${c.score}\n`;
        });
        md += '\n';
      }
      md += '---\n\n';
    });

    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `research-session-${Date.now()}.md`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  const API_KEY = process.env.NEXT_PUBLIC_API_KEY || 'rag-prod-secret-key-change-me';

  // Load available documents dynamically from the backend
  useEffect(() => {
    async function fetchDocuments() {
      try {
        const res = await fetch(`${API_URL}/documents`, {
          headers: { 'x-api-key': API_KEY }
        });
        if (res.ok) {
          const data = await res.json();
          if (data.documents && data.documents.length > 0) {
            const formatted: DocumentItem[] = data.documents.map((d: any, idx: number) => ({
              id: idx + 1,
              name: d.name,
              title: d.name,
              subtitle: d.type === 'pdf' ? `PDF ${d.pages} pages` : d.type === 'md' ? 'Note 1,400 words' : `${d.type.toUpperCase()} Doc`,
              type: d.type,
              chunks: d.chunks,
              pages: d.pages
            }));
            setDocuments(formatted);
            setPinnedSources(formatted.map(d => d.name));
          }
        }
      } catch (e) {
        // Fallback to default catalog
      }
    }
    fetchDocuments();
  }, [API_URL, API_KEY]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const togglePin = (sourceName: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    setPinnedSources(prev =>
      prev.includes(sourceName) ? prev.filter(s => s !== sourceName) : [...prev, sourceName]
    );
  };

  const pinAll = () => setPinnedSources(documents.map(d => d.name));
  const clearPins = () => setPinnedSources([]);

  // Source number mapping for citation badges
  const getSourceNumber = (sourceName: string): number => {
    const doc = documents.find(d => d.name.toLowerCase() === sourceName.toLowerCase());
    return doc ? doc.id : 1;
  };

  // Inspect source details when clicked in sidebar
  const handleInspectSource = (doc: DocumentItem) => {
    setSelectedCitation({
      source: doc.name,
      page: 1,
      chunk_id: `${doc.name}_catalog_entry`,
      score: 1.0,
      snippet: `Indexed Document: ${doc.title}\nFormat: ${doc.subtitle}\nTotal Chunks: ${doc.chunks} indexed vectors in Qdrant.\nPages: ${doc.pages}\nStatus: ${pinnedSources.includes(doc.name) ? 'Pinned (Active retrieval context)' : 'Unpinned (Excluded from context)'}`
    });
  };

  const handleSend = async (questionText?: string) => {
    const q = (questionText || input).trim();
    if (!q || loading) return;

    setInput('');
    const userMsgId = `usr-${Date.now()}`;
    const botMsgId = `bot-${Date.now()}`;
    const startTime = performance.now();

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
        status: 'working'
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

      const elapsed = Math.round((performance.now() - startTime) / 100) / 10;
      setMessages(prev =>
        prev.map(m => m.id === botMsgId ? { ...m, isStreaming: false, status: 'done', elapsedSeconds: elapsed } : m)
      );
    } catch (err: any) {
      const elapsed = Math.round((performance.now() - startTime) / 100) / 10;
      setMessages(prev =>
        prev.map(m =>
          m.id === botMsgId
            ? {
                ...m,
                content: `Error: Unable to complete query (${err.message}). Verify API service is running.`,
                isStreaming: false,
                status: 'error',
                elapsedSeconds: elapsed
              }
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

  const copyMessageContent = (msgId: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedMessageId(msgId);
    setTimeout(() => setCopiedMessageId(null), 2000);
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Reset input so re-selecting the same file works
    e.target.value = '';

    if (file.size > 25 * 1024 * 1024) {
      setUploadStatus("File exceeds maximum allowed size (25MB).");
      return;
    }

    setUploading(true);
    setUploadStatus(`Uploading "${file.name}" & computing vector embeddings...`);

    const formData = new FormData();
    formData.append("file", file);

    const tStart = performance.now();
    try {
      const res = await fetch(`${API_URL}/ingest`, {
        method: 'POST',
        headers: { 'x-api-key': API_KEY },
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        const elapsedSec = ((performance.now() - tStart) / 1000).toFixed(1);
        setUploadStatus(`✓ Indexed ${data.chunks_indexed} chunk(s) in ${elapsedSec}s!`);
        setTimeout(() => setShowUploadModal(false), 1400);

        // Refresh document catalog safely
        try {
          const docRes = await fetch(`${API_URL}/documents`, { headers: { 'x-api-key': API_KEY } });
          if (docRes.ok) {
            const docData = await docRes.json();
            if (docData.documents && docData.documents.length > 0) {
              const formatted: DocumentItem[] = docData.documents.map((d: any, idx: number) => ({
                id: idx + 1,
                name: d.name,
                title: d.title || d.name,
                subtitle: d.type === 'pdf' ? `PDF ${d.pages} pages` : d.type === 'md' ? 'Note 1,400 words' : `${d.type.toUpperCase()} Doc`,
                type: d.type,
                chunks: d.chunks,
                pages: d.pages
              }));
              setDocuments(formatted);
            }
          }
        } catch (catErr) {
          console.warn("Catalog refresh postponed:", catErr);
        }
        setPinnedSources(prev => prev.includes(file.name) ? prev : [...prev, file.name]);
      } else {
        let errDetail = "";
        try {
          const errJson = await res.json();
          errDetail = errJson.detail || JSON.stringify(errJson);
        } catch {
          errDetail = await res.text().catch(() => "");
        }
        setUploadStatus(`Upload failed: ${res.status} ${errDetail || res.statusText || 'Error'}`);
      }
    } catch (err: any) {
      console.error("Upload error:", err);
      setUploadStatus(`Upload failed: ${err?.message || 'Server connection error'}`);
    } finally {
      setUploading(false);
      setTimeout(() => setUploadStatus(null), 6000);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedSnippet(true);
    setTimeout(() => setCopiedSnippet(false), 2000);
  };

  // Convert raw citations ([source.pdf, page X] or [1]) into clean numbered pills
  const renderFormattedText = (text: string, citations?: Citation[]) => {
    if (!text) return null;

    const parts = text.split(/(\[[^\]]+\])/g);
    return parts.map((part, index) => {
      if (part.startsWith('[') && part.endsWith(']')) {
        const inner = part.slice(1, -1).trim();

        // 1. Direct numeric format like [1], [2]
        if (/^\d+$/.test(inner)) {
          const docId = parseInt(inner, 10);
          const matchedDoc = documents.find(d => d.id === docId);
          const matchedCitation = citations?.find(c =>
            matchedDoc ? c.source.toLowerCase() === matchedDoc.name.toLowerCase() : false
          ) || citations?.[docId - 1];

          return (
            <button
              key={index}
              onClick={() => {
                if (matchedCitation) {
                  setSelectedCitation(matchedCitation);
                } else if (matchedDoc) {
                  setSelectedCitation({
                    source: matchedDoc.name,
                    page: 1,
                    chunk_id: `${matchedDoc.name}_p1_c0`,
                    score: 0.033,
                    snippet: `Pinned source: ${matchedDoc.title} (${matchedDoc.subtitle})`
                  });
                }
              }}
              className="inline-flex items-center justify-center w-5 h-5 mx-1 text-[11px] font-mono font-semibold rounded bg-[#2b2b30] hover:bg-[#383842] text-slate-200 border border-[#3f3f46] transition-colors cursor-pointer align-baseline"
              title={`View Source #${docId}: ${matchedDoc?.name || ''}`}
            >
              {docId}
            </button>
          );
        }

        // 2. Full citation format like [source.pdf, page X]
        const matchedCitation = citations?.find(c => inner.toLowerCase().includes(c.source.toLowerCase()));
        if (matchedCitation) {
          const docNum = getSourceNumber(matchedCitation.source);
          return (
            <button
              key={index}
              onClick={() => setSelectedCitation(matchedCitation)}
              className="inline-flex items-center justify-center w-5 h-5 mx-1 text-[11px] font-mono font-semibold rounded bg-[#2b2b30] hover:bg-[#383842] text-slate-200 border border-[#3f3f46] transition-colors cursor-pointer align-baseline"
              title={`View Citation: ${matchedCitation.source} (p. ${matchedCitation.page})`}
            >
              {docNum}
            </button>
          );
        }
      }
      return <span key={index}>{part}</span>;
    });
  };

  const totalCitationsCount = messages.reduce((acc, m) => acc + (m.citations?.length || 0), 0);

  return (
    <div className="flex h-[calc(100vh-5.5rem)] w-full bg-[#0c0c0e] text-[#f4f4f5] antialiased select-none font-sans overflow-hidden">
      {/* ------------------------------------------------------------- */}
      {/* LEFT SIDEBAR: Pinned Sources                                  */}
      {/* ------------------------------------------------------------- */}
      <aside className="w-72 lg:w-80 flex-shrink-0 flex flex-col p-4 border-r border-[#1e1e24] bg-[#0c0c0e]">
        {/* Sidebar Header */}
        <div className="flex items-center justify-between pb-3 mb-2">
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-300 uppercase tracking-wider">
            <Pin className="w-3.5 h-3.5 text-slate-400 fill-slate-400 rotate-45" />
            <span>Pinned sources</span>
          </div>
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <button onClick={pinAll} className="hover:text-slate-300 transition-colors">All</button>
            <span>•</span>
            <button onClick={clearPins} className="hover:text-slate-300 transition-colors">Clear</button>
          </div>
        </div>

        {/* Pinned Sources Container Card */}
        <div className="flex-1 overflow-y-auto space-y-1 pr-1 border border-[#1f1f23] bg-[#111114] rounded-2xl p-2 shadow-inner scrollbar-thin scrollbar-thumb-zinc-800">
          {documents.map((doc) => {
            const isPinned = pinnedSources.includes(doc.name);
            return (
              <div
                key={doc.id}
                onClick={() => handleInspectSource(doc)}
                className={cn(
                  "group flex items-start gap-3 p-2.5 rounded-xl border transition-all cursor-pointer",
                  isPinned
                    ? "bg-[#16161a] border-[#27272e] hover:border-[#383842]"
                    : "bg-transparent border-transparent opacity-40 hover:opacity-75"
                )}
              >
                {/* Numbered Square Badge */}
                <div className="w-6 h-6 flex-shrink-0 rounded-md bg-[#222228] border border-[#2d2d35] flex items-center justify-center text-xs font-mono font-medium text-slate-300">
                  {doc.id}
                </div>

                {/* Document Title & Subtitle */}
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium text-slate-200 truncate leading-tight group-hover:text-white">
                    {doc.title}
                  </p>
                  <p className="text-[11px] text-slate-400 font-normal truncate mt-0.5">
                    {doc.subtitle}
                  </p>
                </div>

                {/* Pin Toggle Button */}
                <button
                  type="button"
                  onClick={(e) => togglePin(doc.name, e)}
                  className="flex-shrink-0 pt-0.5 p-1 text-slate-400 hover:text-white"
                  title={isPinned ? "Unpin Source" : "Pin Source"}
                >
                  {isPinned ? (
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 inline-block shadow-[0_0_8px_#10b981]" />
                  ) : (
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-600 inline-block" />
                  )}
                </button>
              </div>
            );
          })}
        </div>

        {/* Upload Trigger Button */}
        <div className="pt-3">
          <button
            onClick={() => setShowUploadModal(true)}
            className="w-full flex items-center justify-center gap-2 py-2 px-3 text-xs font-medium rounded-xl border border-[#27272e] bg-[#141418] hover:bg-[#1a1a20] text-slate-300 transition-colors"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Add / Upload Document</span>
          </button>
        </div>
      </aside>

      {/* ------------------------------------------------------------- */}
      {/* RIGHT MAIN AREA: Chat Thread & Input Canvas                  */}
      {/* ------------------------------------------------------------- */}
      <main className="flex-1 flex flex-col min-w-0 bg-[#0c0c0e]">
        {/* Top Header Bar */}
        <div className="h-14 flex items-center justify-between px-6 border-b border-[#1e1e24] bg-[#0c0c0e]">
          <div className="flex items-center gap-3">
            <h2 className="text-sm font-semibold text-slate-200">
              AI Research & Architecture Review
            </h2>

            {/* Pipeline Mode Selector */}
            <div className="flex items-center gap-1.5 pl-3 border-l border-[#222226]">
              <Layers className="w-3.5 h-3.5 text-slate-500" />
              <select
                value={selectedMode}
                onChange={(e) => setSelectedMode(e.target.value)}
                className="text-xs bg-[#16161a] border border-[#27272e] text-slate-300 rounded-lg px-2 py-1 font-medium focus:ring-1 focus:ring-emerald-500 focus:outline-none"
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

          {/* Right Cited Badge & Reset */}
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-400 font-medium">
              <strong className="text-slate-200 font-mono">{totalCitationsCount}</strong> sources Cited
            </span>

            {/* Export Chat */}
            {messages.length > 0 && (
              <button
                onClick={exportChat}
                className="p-1.5 text-slate-400 hover:text-slate-200 rounded-lg hover:bg-[#1c1c21] transition-colors flex items-center gap-1 text-xs"
                title="Export Chat as Markdown"
              >
                <Download className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Export</span>
              </button>
            )}

            <button
              onClick={() => {
                if (window.speechSynthesis) window.speechSynthesis.cancel();
                setSpeakingId(null);
                setMessages([]);
              }}
              className="p-1.5 text-slate-400 hover:text-rose-400 rounded-lg hover:bg-[#1c1c21] transition-colors"
              title="Clear Thread"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Message Stream */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-6 scrollbar-thin scrollbar-thumb-zinc-800">
          {messages.length === 0 ? (
            /* Clean Empty State */
            <div className="h-full flex flex-col items-center justify-center text-center p-6 space-y-3 opacity-80">
              <div className="w-12 h-12 rounded-2xl bg-[#16161a] border border-[#27272e] flex items-center justify-center text-emerald-400 shadow-sm">
                <Sparkles className="w-6 h-6" />
              </div>
              <div>
                <h3 className="text-base font-semibold text-slate-200">Grounded Research Assistant</h3>
                <p className="text-xs text-slate-500 max-w-sm mt-1">
                  Ask any question over your {pinnedSources.length} pinned arXiv papers and architectural docs.
                </p>
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <div
                key={msg.id}
                className={cn(
                  "flex flex-col",
                  msg.role === 'user' ? "items-end" : "items-start"
                )}
              >
                {msg.role === 'user' ? (
                  /* User Prompt Bubble */
                  <div className="max-w-xl bg-[#1d1d22] text-[#f4f4f5] px-4 py-2.5 rounded-2xl rounded-br-xs text-sm border border-[#2a2a32] shadow-xs">
                    {msg.content}
                  </div>
                ) : (
                  /* Assistant Response */
                  <div className="max-w-3xl space-y-2">
                    {/* LatticeLoader Component while Thinking / Completed */}
                    {msg.status === 'working' ? (
                      <div className="py-1">
                        <LatticeLoader
                          status="working"
                          label="Thinking"
                          pattern="orbit"
                          grid={3}
                          shape="round"
                          doneColor="#22c55e"
                          errorColor="#ef4444"
                          cellSize={6}
                          gap={2}
                          fontSize={13}
                          step={90}
                          color="#a1a1aa"
                          showTimer={true}
                        />
                      </div>
                    ) : msg.elapsedSeconds != null ? (
                      <div className="py-0.5">
                        <LatticeLoader
                          status={msg.status === 'error' ? 'error' : 'done'}
                          label="Thinking"
                          doneLabel="Done in"
                          errorLabel="Failed after"
                          pattern="orbit"
                          grid={3}
                          shape="round"
                          doneColor="#22c55e"
                          errorColor="#ef4444"
                          cellSize={6}
                          gap={2}
                          fontSize={12}
                          color="#71717a"
                          elapsed={msg.elapsedSeconds}
                          showTimer={true}
                        />
                      </div>
                    ) : null}

                    {/* ReAct Agent Reasoning Trace */}
                    {msg.steps && msg.steps.length > 0 && (
                      <details className="rounded-xl border border-[#27272e] bg-[#131317] p-3 text-xs text-slate-300">
                        <summary className="font-semibold text-emerald-400 cursor-pointer flex items-center gap-1.5 select-none hover:text-emerald-300">
                          <Cpu className="w-3.5 h-3.5 text-emerald-500" />
                          <span>ReAct Reasoning Trace ({msg.steps.length} step{msg.steps.length > 1 ? 's' : ''})</span>
                        </summary>
                        <div className="mt-2.5 space-y-2 pl-2 border-l border-emerald-500/60 text-xs">
                          {msg.steps.map((st, i) => (
                            <div key={i} className="space-y-1">
                              {st.type === 'thought' ? (
                                <div>
                                  <span className="font-semibold text-indigo-400">Step {st.iteration} Thought:</span> {st.content}
                                  {st.tool && (
                                    <div className="mt-0.5 text-slate-400 font-mono text-[11px] bg-[#1a1a20] px-2 py-0.5 rounded border border-[#2e2e38] inline-block">
                                      &rarr; Action: <strong className="text-slate-200">{st.tool}</strong>({st.tool_input || '""'})
                                    </div>
                                  )}
                                </div>
                              ) : (
                                <div className="bg-[#18181d] p-2 rounded-lg border border-[#27272e] font-mono text-[11px] text-slate-400 max-h-32 overflow-y-auto">
                                  <span className="font-semibold text-emerald-400 font-sans block text-xs">Observation ({st.tool}):</span>
                                  {st.content}
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </details>
                    )}

                    {/* Answer Text with Clean Numbered Citation Pills */}
                    <div className="text-sm leading-relaxed text-[#d4d4d8] font-normal whitespace-pre-wrap select-text">
                      {renderFormattedText(msg.content, msg.citations)}
                    </div>

                    {/* Assistant Footer: Feedback & Copy */}
                    {!msg.isStreaming && msg.content && (
                      <div className="flex items-center gap-3 pt-1 text-xs text-slate-500">
                        <button
                          onClick={() => copyMessageContent(msg.id, msg.content)}
                          className="flex items-center gap-1 hover:text-slate-300 transition-colors"
                          title="Copy Answer"
                        >
                          {copiedMessageId === msg.id ? (
                            <>
                              <Check className="w-3 h-3 text-emerald-400" />
                              <span className="text-[11px] text-emerald-400">Copied</span>
                            </>
                          ) : (
                            <>
                              <Copy className="w-3 h-3" />
                              <span className="text-[11px]">Copy</span>
                            </>
                          )}
                        </button>

                        <span className="text-slate-700">•</span>

                        <button
                          onClick={() => toggleSpeech(msg.id, msg.content)}
                          className={cn(
                            "flex items-center gap-1 hover:text-slate-300 transition-colors",
                            speakingId === msg.id && "text-emerald-400 font-medium"
                          )}
                          title={speakingId === msg.id ? "Stop Listening" : "Read Aloud"}
                        >
                          {speakingId === msg.id ? (
                            <>
                              <VolumeX className="w-3 h-3 text-emerald-400 animate-pulse" />
                              <span className="text-[11px] text-emerald-400">Stop</span>
                            </>
                          ) : (
                            <>
                              <Volume2 className="w-3 h-3" />
                              <span className="text-[11px]">Listen</span>
                            </>
                          )}
                        </button>

                        <span className="text-slate-700">•</span>

                        <button
                          onClick={() => handleFeedback(msg.id, 1)}
                          className={cn("p-0.5 hover:text-slate-300 transition-colors", msg.feedback === 1 && "text-emerald-400 font-bold")}
                          title="Grounded & Helpful"
                        >
                          <ThumbsUp className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={() => handleFeedback(msg.id, -1)}
                          className={cn("p-0.5 hover:text-slate-300 transition-colors", msg.feedback === -1 && "text-rose-400 font-bold")}
                          title="Ungrounded or Inaccurate"
                        >
                          <ThumbsDown className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* ----------------------------------------------------------- */}
        {/* BOTTOM INPUT AREA: Rounded Card matching the image design   */}
        {/* ----------------------------------------------------------- */}
        <div className="p-6 pt-2">
          {/* Suggested Prompts (when thread is empty) */}
          {messages.length <= 1 && (
            <div className="flex flex-wrap gap-1.5 mb-3">
              {SUGGESTED_PROMPTS.map((q, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSend(q)}
                  disabled={loading}
                  className="text-xs bg-[#141418] hover:bg-[#1c1c22] border border-[#222228] text-slate-400 hover:text-slate-200 px-3 py-1.5 rounded-full transition-colors disabled:opacity-50"
                >
                  {q}
                </button>
              ))}
            </div>
          )}

          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
              if (textareaRef.current) {
                textareaRef.current.style.height = 'auto';
              }
            }}
            className="rounded-2xl border border-[#27272e] bg-[#141418] focus-within:border-[#3f3f4a] focus-within:ring-1 focus-within:ring-[#3f3f4a]/30 p-3 shadow-lg transition-all outline-none"
          >
            {/* Input Field */}
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={handleInputChange}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                  if (textareaRef.current) {
                    textareaRef.current.style.height = 'auto';
                  }
                }
              }}
              placeholder="Ask anything..."
              disabled={loading}
              className="w-full bg-transparent text-sm text-slate-200 placeholder-slate-500 border-none outline-none ring-0 focus:outline-none focus:ring-0 focus:border-none focus-visible:outline-none focus-visible:ring-0 resize-none leading-relaxed overflow-hidden scrollbar-none"
            />

            {/* Bottom Controls Row: '+' Button, Pinned Count Badge, Send Button */}
            <div className="flex items-center justify-between pt-1">
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setShowUploadModal(true)}
                  className="w-7 h-7 rounded-lg bg-[#1f1f26] hover:bg-[#282832] text-slate-400 hover:text-slate-200 flex items-center justify-center transition-colors"
                  title="Upload New Document"
                >
                  <Plus className="w-4 h-4" />
                </button>

                <button
                  type="button"
                  onClick={toggleListening}
                  className={cn(
                    "w-7 h-7 rounded-lg flex items-center justify-center transition-colors",
                    isListening
                      ? "bg-rose-500/20 text-rose-400 border border-rose-500/40 animate-pulse"
                      : "bg-[#1f1f26] hover:bg-[#282832] text-slate-400 hover:text-slate-200"
                  )}
                  title={isListening ? "Listening... Click to Stop" : "Voice Input"}
                >
                  {isListening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
                </button>

                <span className="text-[11px] font-mono text-slate-500">
                  {pinnedSources.length} source(s) pinned
                </span>
              </div>

              <button
                type="submit"
                disabled={loading || !input.trim()}
                className="w-8 h-8 rounded-lg bg-[#272730] hover:bg-[#32323e] text-slate-200 flex items-center justify-center transition-colors disabled:opacity-30 disabled:cursor-not-allowed shadow-xs"
                title="Send Message"
              >
                <ArrowUp className="w-4 h-4" />
              </button>
            </div>
          </form>
        </div>
      </main>

      {/* ------------------------------------------------------------- */}
      {/* CITATION INSPECTOR MODAL / SLIDE-OVER                         */}
      {/* ------------------------------------------------------------- */}
      {selectedCitation && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-[#141418] border border-[#27272e] text-[#f4f4f5] rounded-2xl max-w-lg w-full p-6 shadow-2xl animate-in fade-in zoom-in-95 duration-150">
            <div className="flex items-center justify-between pb-3 border-b border-[#222228]">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 rounded bg-[#222228] border border-[#2d2d35] flex items-center justify-center text-xs font-mono font-medium text-slate-300">
                  {getSourceNumber(selectedCitation.source)}
                </div>
                <h3 className="font-semibold text-sm text-slate-200">{selectedCitation.source}</h3>
                <span className="text-[11px] font-mono bg-emerald-950/80 text-emerald-400 border border-emerald-800 px-2 py-0.5 rounded-full">
                  Page {selectedCitation.page}
                </span>
              </div>
              <button
                onClick={() => setSelectedCitation(null)}
                className="text-slate-400 hover:text-slate-200 text-sm font-bold"
              >
                ✕
              </button>
            </div>

            <div className="my-4 space-y-3">
              <div className="flex items-center justify-between text-xs text-slate-400 font-mono">
                <span>Chunk: <code className="bg-[#1f1f26] px-1 py-0.5 rounded text-slate-300">{selectedCitation.chunk_id}</code></span>
                <span>Score: <strong className="text-emerald-400">{selectedCitation.score}</strong></span>
              </div>

              <div className="bg-[#0e0e11] p-3.5 rounded-xl border border-[#222228] text-xs leading-relaxed text-slate-300 max-h-60 overflow-y-auto whitespace-pre-wrap font-sans">
                {selectedCitation.snippet}
              </div>
            </div>

            <div className="flex items-center justify-between pt-3 border-t border-[#222228]">
              <button
                onClick={() => copyToClipboard(selectedCitation.snippet)}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 bg-[#1e1e24] hover:bg-[#27272f] rounded-lg transition-colors border border-[#2c2c36]"
              >
                {copiedSnippet ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-400" />
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
                className="px-4 py-1.5 text-xs font-semibold bg-slate-200 hover:bg-white text-slate-900 rounded-lg transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* UPLOAD DOCUMENT MODAL                                         */}
      {/* ------------------------------------------------------------- */}
      {showUploadModal && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-[#141418] border border-[#27272e] text-[#f4f4f5] rounded-2xl max-w-md w-full p-6 shadow-2xl animate-in fade-in zoom-in-95 duration-150">
            <div className="flex items-center justify-between pb-3 border-b border-[#222228]">
              <h3 className="font-semibold text-sm text-slate-200 flex items-center gap-2">
                <Upload className="w-4 h-4 text-emerald-400" />
                Upload Document to Catalog
              </h3>
              <button
                onClick={() => setShowUploadModal(false)}
                className="text-slate-400 hover:text-slate-200 text-sm font-bold"
              >
                ✕
              </button>
            </div>

            <div className="my-4 space-y-3">
              <p className="text-xs text-slate-400">
                Upload a research paper or documentation file (.pdf, .md, .html, .txt). The offline pipeline will sanitize, chunk, compute BGE vector embeddings, and pin it to your active session.
              </p>

              <label className={cn(
                "border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center gap-2 transition-colors bg-[#0f0f13]",
                uploading ? "border-emerald-500/50 cursor-wait" : "border-[#2b2b36] hover:border-emerald-500/60 cursor-pointer"
              )}>
                {uploading ? (
                  <div className="py-2 flex flex-col items-center gap-2">
                    <LatticeLoader status="working" label="Indexing" pattern="orbit" cellSize={7} />
                    <span className="text-xs text-emerald-400 font-mono mt-1">Extracting & embedding chunks...</span>
                  </div>
                ) : (
                  <>
                    <Upload className="w-6 h-6 text-slate-400" />
                    <span className="text-xs text-slate-300 font-medium">Select file to ingest</span>
                    <span className="text-[10px] text-slate-500">PDF, Markdown, HTML, TXT</span>
                  </>
                )}
                <input
                  type="file"
                  className="hidden"
                  accept=".pdf,.md,.html,.txt"
                  onChange={handleFileUpload}
                  disabled={uploading}
                />
              </label>

              {uploadStatus && (
                <p className="text-xs text-emerald-400 text-center">{uploadStatus}</p>
              )}
            </div>

            <div className="flex justify-end pt-2 border-t border-[#222228]">
              <button
                onClick={() => setShowUploadModal(false)}
                className="px-4 py-1.5 text-xs text-slate-300 hover:text-white"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default AIChat7;
