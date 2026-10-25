import './globals.css';
import Link from 'next/link';
import { Bot, BarChart3, Github } from 'lucide-react';

export const metadata = {
  title: 'Production RAG & AI Agent System',
  description: 'Enterprise RAG application with hybrid retrieval, cross-encoder reranking, and real-time streaming.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
        <header className="border-b border-slate-200 bg-white sticky top-0 z-30 shadow-sm">
          <div className="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-emerald-600 text-white p-2 rounded-lg shadow-sm">
                <Bot className="w-5 h-5" />
              </div>
              <div>
                <h1 className="font-semibold text-slate-900 leading-tight">Production RAG Assistant</h1>
                <p className="text-xs text-slate-500">Hybrid Search (BM25 + Dense RRF) • Strict Guardrails</p>
              </div>
            </div>

            <nav className="flex items-center space-x-6 text-sm font-medium">
              <Link href="/" className="text-slate-700 hover:text-emerald-600 transition-colors flex items-center gap-1.5">
                <Bot className="w-4 h-4" />
                Chat
              </Link>
              <Link href="/metrics" className="text-slate-700 hover:text-emerald-600 transition-colors flex items-center gap-1.5">
                <BarChart3 className="w-4 h-4" />
                Metrics & Ablations
              </Link>
              <a
                href="https://github.com"
                target="_blank"
                rel="noreferrer"
                className="text-slate-500 hover:text-slate-900 transition-colors"
                title="View GitHub Repository"
              >
                <Github className="w-5 h-5" />
              </a>
            </nav>
          </div>
        </header>

        <main className="flex-1 max-w-6xl w-full mx-auto p-4 md:p-6">
          {children}
        </main>

        <footer className="border-t border-slate-200 py-4 bg-white text-center text-xs text-slate-500">
          Production RAG Architecture • FastEmbed BGE • Qdrant Vector Store • Server-Sent Events Streaming
        </footer>
      </body>
    </html>
  );
}
