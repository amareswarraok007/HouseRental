import React, { useEffect, useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const fallbackData = {
  incident_name: "Silent Checkout Collapse After Payment Deploy",
  stability_score: 58,
  run_number: 2,
  anomalies: [],
  diagnosis: {
    root_cause:
      "Payment service connection pool exhaustion after v3.2.1 deployment",
    confidence: 0.9,
    evidence: [],
  },
  remediation: [],
  execution_log: [],
  timeline: [
    {
      agent: "Observer",
      status: "completed",
      summary: "Detected 4 anomalies from metrics CSVs.",
      timestamp: "2024-07-01T10:03:10Z",
    },
    {
      agent: "Diagnostician",
      status: "completed",
      summary: "Correlated anomalies with RAG memory and deployments.",
      timestamp: "2024-07-01T10:03:40Z",
    },
    {
      agent: "Planner",
      status: "completed",
      summary: "Generated remediation plan with risk flags.",
      timestamp: "2024-07-01T10:04:10Z",
    },
    {
      agent: "Executor",
      status: "completed",
      summary: "Simulated safe execution sequence.",
      timestamp: "2024-07-01T10:04:40Z",
    },
    {
      agent: "Evaluator",
      status: "completed",
      summary: "Updated thresholds for faster detection next run.",
      timestamp: "2024-07-01T10:05:10Z",
    },
  ],
  learning: {
    run_1: { mttr_minutes: 18, detection_minutes: 5, confidence: 0.76 },
    run_2: { mttr_minutes: 12, detection_minutes: 3, confidence: 0.9 },
    improvement:
      "Tuned latency/error thresholds and tightened prompt to spot pool exhaustion earlier.",
  },
  metadata: { deploy_time: "09:58", latency_spike: "10:03", orders_failed: 3 },
};

const fetchDemo = async () => {
  const response = await fetch("http://localhost:8000/api/demo");
  if (!response.ok) {
    throw new Error("Failed to load demo");
  }
  return response.json();
};

const Gauge = ({ score }) => {
  const degrees = useMemo(() => `${score * 3.6}deg`, [score]);
  return (
    <div className="flex flex-col items-center gap-4">
      <div
        className="gauge-ring relative flex h-56 w-56 items-center justify-center rounded-full border border-slate-800 shadow-xl"
        style={{ "--score": degrees }}
      >
        <div className="flex h-40 w-40 flex-col items-center justify-center rounded-full bg-engine-card shadow-inner">
          <span className="text-5xl font-semibold text-white">{score}</span>
          <span className="text-sm text-slate-400">Stability Score</span>
        </div>
        <div className="absolute -bottom-6 flex items-center gap-2 rounded-full bg-slate-900 px-3 py-1 text-xs text-slate-300">
          <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
          Incident stability index
        </div>
      </div>
    </div>
  );
};

const TimelineItem = ({ item }) => (
  <div className="flex items-start gap-3 rounded-xl border border-slate-800 bg-engine-card p-4">
    <div className="mt-1 h-2 w-2 rounded-full bg-blue-400" />
    <div>
      <p className="text-sm font-semibold text-slate-100">{item.agent}</p>
      <p className="text-xs text-slate-400">{item.summary}</p>
      <p className="text-[10px] text-slate-500">{item.timestamp}</p>
    </div>
  </div>
);

const StatPill = ({ label, value }) => (
  <div className="rounded-lg bg-slate-900 px-3 py-2">
    <p className="text-xs text-slate-400">{label}</p>
    <p className="text-sm font-semibold text-white">{value}</p>
  </div>
);

export default function App() {
  const [demo, setDemo] = useState(fallbackData);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchDemo()
      .then(setDemo)
      .catch((err) => {
        setError(err.message);
      });
  }, []);

  const chartData = [
    {
      run: "Run 1",
      confidence: demo.learning.run_1.confidence * 100,
      mttr: demo.learning.run_1.mttr_minutes,
      detection: demo.learning.run_1.detection_minutes,
    },
    {
      run: "Run 2",
      confidence: demo.learning.run_2.confidence * 100,
      mttr: demo.learning.run_2.mttr_minutes,
      detection: demo.learning.run_2.detection_minutes,
    },
  ];

  return (
    <div className="min-h-screen bg-engine-dark px-6 py-10 text-slate-100">
      <header className="mb-8 flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-sm uppercase tracking-[0.3em] text-slate-500">
            Incident AI Engine
          </p>
          <h1 className="text-3xl font-semibold text-white">
            E-commerce Incident Management Dashboard
          </h1>
        </div>
        <div className="flex items-center gap-3 rounded-full border border-slate-800 bg-slate-900 px-4 py-2 text-xs text-slate-300">
          <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
          Live simulation • Run #{demo.run_number}
        </div>
      </header>

      <section className="grid gap-6 lg:grid-cols-[1.1fr_1.4fr_1.1fr]">
        <div className="flex flex-col gap-4">
          <h2 className="text-sm font-semibold text-slate-300">Agent Activity Timeline</h2>
          <div className="flex flex-col gap-3">
            {(demo.timeline.length ? demo.timeline : fallbackData.timeline).map((item, index) => (
              <TimelineItem key={`${item.agent}-${index}`} item={item} />
            ))}
            {!demo.timeline.length && (
              <p className="text-xs text-slate-500">
                Timeline will populate once the backend is running.
              </p>
            )}
          </div>
        </div>

        <div className="flex flex-col items-center justify-center gap-6 rounded-3xl border border-slate-800 bg-engine-card p-8 shadow-2xl">
          <Gauge score={demo.stability_score} />
          <div className="grid w-full grid-cols-3 gap-3 text-center">
            <StatPill label="Deploy" value={demo.metadata.deploy_time} />
            <StatPill label="Latency spike" value={demo.metadata.latency_spike} />
            <StatPill label="Failed orders" value={demo.metadata.orders_failed} />
          </div>
          {error && (
            <p className="rounded-xl border border-amber-400 bg-amber-950/40 px-3 py-2 text-xs text-amber-200">
              Backend unavailable: {error}
            </p>
          )}
        </div>

        <div className="flex flex-col gap-4">
          <h2 className="text-sm font-semibold text-slate-300">Active Incident</h2>
          <div className="rounded-2xl border border-slate-800 bg-engine-card p-5 shadow-lg">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">
              Incident
            </p>
            <p className="text-lg font-semibold text-white">{demo.incident_name}</p>
            <div className="mt-4 space-y-3 text-sm text-slate-300">
              <div>
                <p className="text-xs text-slate-500">Root Cause</p>
                <p className="font-medium text-white">{demo.diagnosis.root_cause}</p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Confidence</p>
                <div className="mt-1 flex items-center gap-2">
                  <div className="h-2 flex-1 rounded-full bg-slate-800">
                    <div
                      className="h-2 rounded-full bg-emerald-400"
                      style={{ width: `${demo.diagnosis.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-slate-400">
                    {(demo.diagnosis.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <div>
                <p className="text-xs text-slate-500">Suggested Remediation</p>
                <ul className="mt-2 space-y-2 text-xs text-slate-300">
                  {demo.remediation.map((step) => (
                    <li key={step.action} className="flex items-start gap-2">
                      <span
                        className={`mt-1 h-2 w-2 rounded-full ${
                          step.risky ? "bg-rose-400" : "bg-emerald-400"
                        }`}
                      />
                      <span>
                        {step.action}
                        <span className="block text-[10px] text-slate-500">
                          {step.reason}
                        </span>
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-engine-card p-5">
            <p className="text-xs uppercase tracking-[0.2em] text-slate-500">Execution Log</p>
            <ul className="mt-3 space-y-2 text-xs text-slate-300">
              {demo.execution_log.map((log) => (
                <li key={log} className="rounded-lg bg-slate-900 px-3 py-2">
                  {log}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <section className="mt-10 grid gap-6 lg:grid-cols-[1.2fr_2fr]">
        <div className="rounded-2xl border border-slate-800 bg-engine-card p-6">
          <h3 className="text-sm font-semibold text-slate-300">Learning Summary</h3>
          <p className="mt-2 text-sm text-slate-400">{demo.learning.improvement}</p>
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-3">
              <p className="text-xs text-slate-500">Run #1 MTTR</p>
              <p className="text-lg font-semibold text-white">
                {demo.learning.run_1.mttr_minutes} min
              </p>
              <p className="text-xs text-slate-500">
                Detection: {demo.learning.run_1.detection_minutes} min
              </p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-3">
              <p className="text-xs text-slate-500">Run #2 MTTR</p>
              <p className="text-lg font-semibold text-emerald-400">
                {demo.learning.run_2.mttr_minutes} min
              </p>
              <p className="text-xs text-slate-500">
                Detection: {demo.learning.run_2.detection_minutes} min
              </p>
            </div>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-800 bg-engine-card p-6">
          <h3 className="text-sm font-semibold text-slate-300">Before vs After Metrics</h3>
          <div className="mt-4 h-60">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                <XAxis dataKey="run" stroke="#64748b" />
                <YAxis stroke="#64748b" />
                <Tooltip
                  contentStyle={{ background: "#0f172a", border: "1px solid #1f2937" }}
                  labelStyle={{ color: "#94a3b8" }}
                />
                <Line type="monotone" dataKey="confidence" stroke="#22c55e" strokeWidth={2} />
                <Line type="monotone" dataKey="mttr" stroke="#38bdf8" strokeWidth={2} />
                <Line type="monotone" dataKey="detection" stroke="#f97316" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>
    </div>
  );
}
