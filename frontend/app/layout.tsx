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
    <html lang="en" className="dark">
      <body className="min-h-screen flex flex-col bg-[#0c0c0e] text-[#f4f4f5]">
        <header className="border-b border-[#1e1e24] bg-[#0c0c0e] sticky top-0 z-30">
          <div className="max-w-7xl mx-auto px-4 h-14 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 p-1.5 rounded-lg">
                <Bot className="w-4 h-4" />
              </div>
              <div>
                <h1 className="font-semibold text-xs md:text-sm text-slate-100 leading-tight">
                  Production RAG Assistant
                </h1>
                <p className="text-[11px] text-slate-500">
                  AI Chat 7 with Pinned Sources & Citations
                </p>
              </div>
            </div>

            <nav className="flex items-center space-x-6 text-xs font-medium">
              <Link href="/" className="text-slate-300 hover:text-white transition-colors flex items-center gap-1.5">
                <Bot className="w-3.5 h-3.5" />
                Chat
              </Link>
              <Link href="/metrics" className="text-slate-400 hover:text-slate-200 transition-colors flex items-center gap-1.5">
                <BarChart3 className="w-3.5 h-3.5" />
                Metrics & Ablations
              </Link>
              <a
                href="https://github.com"
                target="_blank"
                rel="noreferrer"
                className="text-slate-500 hover:text-slate-300 transition-colors"
                title="View GitHub Repository"
              >
                <Github className="w-4 h-4" />
              </a>
            </nav>
          </div>
        </header>

        <main className="flex-1 w-full mx-auto">
          {children}
        </main>
      </body>
    </html>
  );
}
