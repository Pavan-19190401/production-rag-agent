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
  Zap
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
      setError(err.message || "Failed to load metrics");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 10000); // 10s auto refresh
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6 pb-12">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-slate-900 flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-emerald-600" />
            Evaluation Benchmark & Production Telemetry
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            Real-time quality scores, ablation results, latency percentiles, and spending caps.
          </p>
        </div>
        <button
          onClick={fetchMetrics}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-white hover:bg-slate-50 border border-slate-200 rounded-lg text-slate-700 shadow-2xs transition-colors"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Live System Telemetry Cards */}
      {metrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
            <div className="flex items-center justify-between text-slate-500 text-xs mb-1">
              <span>Total Queries</span>
              <Zap className="w-4 h-4 text-emerald-600" />
            </div>
            <p className="text-2xl font-bold text-slate-900">{metrics.telemetry.total_queries}</p>
            <p className="text-2xs text-slate-400 mt-0.5">Uptime: {Math.round(metrics.telemetry.uptime_seconds)}s</p>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
            <div className="flex items-center justify-between text-slate-500 text-xs mb-1">
              <span>Cache Hit Ratio</span>
              <Database className="w-4 h-4 text-blue-600" />
            </div>
            <p className="text-2xl font-bold text-slate-900">
              {(metrics.telemetry.cache_hit_ratio * 100).toFixed(1)}%
            </p>
            <p className="text-2xs text-slate-400 mt-0.5">{metrics.telemetry.cache_hits} served in &lt;2ms</p>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
            <div className="flex items-center justify-between text-slate-500 text-xs mb-1">
              <span>Average Latency</span>
              <Clock className="w-4 h-4 text-amber-600" />
            </div>
            <p className="text-2xl font-bold text-slate-900">
              {metrics.telemetry.average_latency_ms.toFixed(1)} ms
            </p>
            <p className="text-2xs text-slate-400 mt-0.5">Retrieval + LLM generation</p>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-xs">
            <div className="flex items-center justify-between text-slate-500 text-xs mb-1">
              <span>Daily Spend</span>
              <DollarSign className="w-4 h-4 text-purple-600" />
            </div>
            <p className="text-2xl font-bold text-slate-900">
              ${metrics.budget.daily_spend_usd.toFixed(4)}
            </p>
            <p className="text-2xs text-slate-400 mt-0.5">Cap: ${metrics.budget.daily_budget_usd.toFixed(2)}</p>
          </div>
        </div>
      )}

      {/* Ablation Benchmark Table */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-xs overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-100 flex items-center justify-between">
          <div>
            <h3 className="font-semibold text-slate-900 text-sm">Empirical Ablation Benchmark (Golden Dataset)</h3>
            <p className="text-xs text-slate-500">65 Curated Gold Records (55 Answerable, 10 Unanswerable Refusal Tests)</p>
          </div>
          <span className="text-xs bg-emerald-100 text-emerald-800 font-medium px-2.5 py-1 rounded-md">
            Production Default: Hybrid RRF
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 text-slate-500 font-medium border-b border-slate-200">
              <tr>
                <th className="px-5 py-3">Configuration</th>
                <th className="px-4 py-3">Recall@5 (Hit Rate)</th>
                <th className="px-4 py-3">MRR</th>
                <th className="px-4 py-3">nDCG@5</th>
                <th className="px-4 py-3">Faithfulness</th>
                <th className="px-4 py-3">Hallucination</th>
                <th className="px-4 py-3">p50 Latency</th>
                <th className="px-4 py-3">p95 Latency</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-slate-700">
              {ABLATION_DATA.map((row, i) => (
                <tr
                  key={i}
                  className={`hover:bg-slate-50/80 transition-colors ${
                    row.isDefault ? 'bg-emerald-50/50 font-medium' : ''
                  }`}
                >
                  <td className="px-5 py-3.5 flex items-center gap-2">
                    <span>{row.name}</span>
                    {row.isDefault && (
                      <span className="text-2xs bg-emerald-600 text-white px-1.5 py-0.5 rounded font-semibold uppercase">
                        Default
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3.5 font-bold text-slate-900">{row.recall5}</td>
                  <td className="px-4 py-3.5">{row.mrr}</td>
                  <td className="px-4 py-3.5">{row.ndcg5}</td>
                  <td className="px-4 py-3.5 text-emerald-700 font-semibold">{row.faithfulness}</td>
                  <td className="px-4 py-3.5 text-slate-500">{row.hallucination}</td>
                  <td className="px-4 py-3.5">{row.p50}</td>
                  <td className="px-4 py-3.5">{row.p95}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Engineering Insights Card */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs">
          <h4 className="font-semibold text-slate-900 text-sm mb-2 flex items-center gap-1.5">
            <CheckCircle2 className="w-4 h-4 text-emerald-600" />
            Why Hybrid (BM25 + Dense RRF) Won
          </h4>
          <p className="text-slate-600 leading-relaxed">
            In dense technical and scientific papers (e.g., Attention, LoRA, FlashAttention), pure dense vector search struggles with exact formulas (such as <code>W_0 + B * A</code>), constant identifiers (<code>d_model=512</code>), and acronyms. Adding BM25 inverted indexing and Reciprocal Rank Fusion ($k=60$) yielded a massive <strong>+25.4% boost in Recall@5</strong> and an <strong>84% improvement in MRR</strong> while keeping median latency under 10ms.
          </p>
        </div>

        <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-xs">
          <h4 className="font-semibold text-slate-900 text-sm mb-2 flex items-center gap-1.5">
            <ShieldAlert className="w-4 h-4 text-amber-600" />
            Strict Grounding & Hallucination Guardrails
          </h4>
          <p className="text-slate-600 leading-relaxed">
            The system achieves <strong>90.0% Faithfulness</strong> by enforcing grounding prompt directives, requiring bracketed page citations <code>[source, p. X]</code>, and performing post-generation claim verification. When an out-of-domain or unanswerable query is submitted, it deterministically refuses with <em>"I don't know based on the provided documents"</em>.
          </p>
        </div>
      </div>
    </div>
  );
}
