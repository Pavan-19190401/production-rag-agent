'use client';

import React, { useState, useEffect } from 'react';
import {
  BarChart3,
  CheckCircle2,
  Clock,
  Database,
  DollarSign,
  Layers,
  RefreshCw,
  ShieldAlert,
  ThumbsUp,
  Zap,
  TrendingUp,
  Activity,
  Cpu
} from 'lucide-react';

interface MetricsData {
  telemetry: {
    uptime_seconds: number;
    total_queries: number;
    cache_hits: number;
    cache_hit_ratio: number;
    refusals: number;
    refusal_rate: number;
    feedback_positive: number;
    feedback_negative: number;
    average_latency_ms: number;
  };
  budget: {
    date: string;
    daily_spend_usd: number;
    daily_budget_usd: number;
    daily_tokens: number;
    budget_remaining_usd: number;
  };
  settings: {
    embedding_model: string;
    reranker_model: string;
    llm_model: string;
    rate_limit_per_minute: number;
  };
}

const ABLATION_DATA = [
  {
    name: "1. Baseline (Dense Vector Search)",
    recall5: "47.3%",
    mrr: "0.3027",
    ndcg5: "0.3328",
    faithfulness: "88.3%",
    hallucination: "11.7%",
    p50: "8.28 ms",
    p95: "10.11 ms",
    isDefault: false
  },
  {
    name: "2. + Hybrid (BM25 + Dense RRF)",
    recall5: "72.7%",
    mrr: "0.5582",
    ndcg5: "0.5741",
    faithfulness: "90.0%",
    hallucination: "10.0%",
    p50: "9.19 ms",
    p95: "13.15 ms",
    isDefault: true
  },
  {
    name: "3. + Cross-Encoder Reranker",
    recall5: "61.8%",
    mrr: "0.4915",
    ndcg5: "0.5020",
    faithfulness: "90.0%",
    hallucination: "10.0%",
    p50: "2,920.59 ms",
    p95: "3,191.96 ms",
    isDefault: false
  },
  {
    name: "4. + Query Rewriting",
    recall5: "70.9%",
    mrr: "0.5385",
    ndcg5: "0.5492",
    faithfulness: "90.0%",
    hallucination: "10.0%",
    p50: "28.57 ms",
    p95: "42.30 ms",
    isDefault: false
  }
];

export default function MetricsPage() {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  const fetchMetrics = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_URL}/metrics`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setMetrics(data);
      setError(null);
    } catch (err: any) {
      setError(err.message || "Failed to load telemetry");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 10000); // 10s auto-refresh
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 py-6 space-y-6 text-[#f4f4f5]">
      {/* Page Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-2 border-b border-[#1e1e24]">
        <div>
          <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-emerald-400" />
            Evaluation Benchmark & Live Telemetry
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Empirical quality benchmarks across 65 golden test cases, latency percentiles, and runtime telemetry.
          </p>
        </div>

        <button
          onClick={fetchMetrics}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-[#141418] hover:bg-[#1e1e24] border border-[#27272e] rounded-xl text-slate-300 hover:text-white transition-colors shadow-2xs"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-emerald-400' : 'text-slate-400'}`} />
          <span>Refresh Live Stats</span>
        </button>
      </div>

      {/* Live System Telemetry Stat Cards */}
      {metrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3.5">
          {/* Card 1: Total Queries */}
          <div className="bg-[#121215] border border-[#1f1f25] p-4 rounded-2xl shadow-xs space-y-1">
            <div className="flex items-center justify-between text-slate-400 text-xs">
              <span className="font-medium">Total Inferences</span>
              <Zap className="w-4 h-4 text-emerald-400" />
            </div>
            <p className="text-2xl font-bold font-mono text-white tracking-tight">
              {metrics.telemetry.total_queries}
            </p>
            <p className="text-[11px] font-mono text-slate-500">
              Uptime: {Math.round(metrics.telemetry.uptime_seconds)}s
            </p>
          </div>

          {/* Card 2: Cache Hit Ratio */}
          <div className="bg-[#121215] border border-[#1f1f25] p-4 rounded-2xl shadow-xs space-y-1">
            <div className="flex items-center justify-between text-slate-400 text-xs">
              <span className="font-medium">Cache Hit Ratio</span>
              <Database className="w-4 h-4 text-cyan-400" />
            </div>
            <p className="text-2xl font-bold font-mono text-white tracking-tight">
              {(metrics.telemetry.cache_hit_ratio * 100).toFixed(1)}%
            </p>
            <p className="text-[11px] font-mono text-slate-500">
              {metrics.telemetry.cache_hits} served in &lt;2ms
            </p>
          </div>

          {/* Card 3: Average Latency */}
          <div className="bg-[#121215] border border-[#1f1f25] p-4 rounded-2xl shadow-xs space-y-1">
            <div className="flex items-center justify-between text-slate-400 text-xs">
              <span className="font-medium">Average Latency</span>
              <Clock className="w-4 h-4 text-amber-400" />
            </div>
            <p className="text-2xl font-bold font-mono text-white tracking-tight">
              {metrics.telemetry.average_latency_ms.toFixed(1)} ms
            </p>
            <p className="text-[11px] font-mono text-slate-500">
              End-to-end SSE streaming
            </p>
          </div>

          {/* Card 4: Daily Spend */}
          <div className="bg-[#121215] border border-[#1f1f25] p-4 rounded-2xl shadow-xs space-y-1">
            <div className="flex items-center justify-between text-slate-400 text-xs">
              <span className="font-medium">Daily Spend</span>
              <DollarSign className="w-4 h-4 text-purple-400" />
            </div>
            <p className="text-2xl font-bold font-mono text-white tracking-tight">
              ${metrics.budget.daily_spend_usd.toFixed(4)}
            </p>
            <p className="text-[11px] font-mono text-slate-500">
              Cap: ${metrics.budget.daily_budget_usd.toFixed(2)} USD
            </p>
          </div>
        </div>
      )}

      {/* Ablation Benchmark Table */}
      <div className="bg-[#121215] border border-[#1f1f25] rounded-2xl overflow-hidden shadow-xs">
        <div className="px-5 py-3.5 border-b border-[#1f1f25] flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="font-semibold text-sm text-slate-200">
              Empirical Ablation Benchmark (Golden Dataset)
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Measured over 65 gold-standard records (55 answerable queries, 10 unanswerable refusal tests).
            </p>
          </div>
          <span className="text-xs font-semibold bg-emerald-950/80 text-emerald-400 border border-emerald-800/80 px-2.5 py-1 rounded-lg">
            Production Winner: Hybrid (BM25 + Dense RRF)
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-[#16161a] border-b border-[#222228] text-slate-400 font-semibold uppercase tracking-wider text-[11px]">
              <tr>
                <th className="px-5 py-3">Configuration</th>
                <th className="px-4 py-3">Recall@5</th>
                <th className="px-4 py-3">MRR</th>
                <th className="px-4 py-3">nDCG@5</th>
                <th className="px-4 py-3">Faithfulness</th>
                <th className="px-4 py-3">Hallucination</th>
                <th className="px-4 py-3">p50 Latency</th>
                <th className="px-4 py-3">p95 Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#1e1e24]">
              {ABLATION_DATA.map((row, i) => (
                <tr
                  key={i}
                  className={`transition-colors ${
                    row.isDefault
                      ? 'bg-emerald-950/20 border-l-2 border-emerald-500 text-slate-200 font-medium'
                      : 'hover:bg-[#18181d] text-slate-300'
                  }`}
                >
                  <td className="px-5 py-3.5 flex items-center gap-2">
                    <span className="text-slate-200">{row.name}</span>
                    {row.isDefault && (
                      <span className="text-[10px] bg-emerald-600 text-white px-1.5 py-0.5 rounded font-bold uppercase tracking-wider">
                        Default
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3.5 font-bold font-mono text-white">{row.recall5}</td>
                  <td className="px-4 py-3.5 font-mono text-slate-300">{row.mrr}</td>
                  <td className="px-4 py-3.5 font-mono text-slate-300">{row.ndcg5}</td>
                  <td className="px-4 py-3.5 font-mono text-emerald-400 font-bold">{row.faithfulness}</td>
                  <td className="px-4 py-3.5 font-mono text-rose-400/80">{row.hallucination}</td>
                  <td className="px-4 py-3.5 font-mono text-slate-300">{row.p50}</td>
                  <td className="px-4 py-3.5 font-mono text-slate-400">{row.p95}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Engineering Insights Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
        {/* Insight 1 */}
        <div className="bg-[#121215] border border-[#1f1f25] p-5 rounded-2xl shadow-xs space-y-2">
          <h4 className="font-semibold text-slate-200 text-sm flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            Why Hybrid (BM25 + Dense RRF) Outperformed Dense Only
          </h4>
          <p className="text-slate-400 leading-relaxed">
            In dense technical and AI research papers (Attention, LoRA, FlashAttention), pure dense vector search struggles with exact formula syntax (such as <code className="bg-[#1a1a20] border border-[#27272e] text-emerald-300 px-1 py-0.5 rounded font-mono">W_0 + B * A</code>), constant identifiers (<code className="bg-[#1a1a20] border border-[#27272e] text-emerald-300 px-1 py-0.5 rounded font-mono">d_model=512</code>), and exact acronyms. Adding BM25 inverted indexing fused via Reciprocal Rank Fusion ($k=60$) produced a massive <strong className="text-slate-200">+25.4% gain in Recall@5</strong> and an <strong className="text-slate-200">84% improvement in MRR</strong> while keeping median latency under 10ms.
          </p>
        </div>

        {/* Insight 2 */}
        <div className="bg-[#121215] border border-[#1f1f25] p-5 rounded-2xl shadow-xs space-y-2">
          <h4 className="font-semibold text-slate-200 text-sm flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-amber-400" />
            Strict Grounding & Hallucination Defense Guardrail
          </h4>
          <p className="text-slate-400 leading-relaxed">
            The system achieves <strong className="text-emerald-400">90.0% Faithfulness</strong> by enforcing strict grounding prompt directives, requiring verified bracketed citations <code className="bg-[#1a1a20] border border-[#27272e] text-amber-300 px-1 py-0.5 rounded font-mono">[source, p. X]</code>, and running post-generation claim verification. When an out-of-domain or unanswerable query is received, it deterministically refuses with <em className="text-slate-300">"I don't know based on the provided documents."</em>
          </p>
        </div>
      </div>

      {/* Production Configuration Summary Footer */}
      {metrics && (
        <div className="bg-[#121215] border border-[#1f1f25] p-4 rounded-2xl flex flex-wrap items-center justify-between gap-3 text-xs text-slate-400">
          <div className="flex flex-wrap items-center gap-4">
            <span className="flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-slate-500" />
              Embeddings: <strong className="text-slate-300 font-mono">{metrics.settings.embedding_model}</strong>
            </span>
            <span className="flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-slate-500" />
              Reranker: <strong className="text-slate-300 font-mono">{metrics.settings.reranker_model}</strong>
            </span>
            <span className="flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5 text-slate-500" />
              Rate Limit: <strong className="text-slate-300 font-mono">{metrics.settings.rate_limit_per_minute}/min</strong>
            </span>
          </div>

          <span className="text-[11px] text-slate-500 font-mono">
            Qdrant Vector Database • FastEmbed ONNX • FastAPI
          </span>
        </div>
      )}
    </div>
  );
}
