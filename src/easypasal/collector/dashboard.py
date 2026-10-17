"""HTML Dashboard for EasyPasal VPS Log Collector (Plan §9 / Task T7.3).

Provides a single-page interactive dashboard serving latency percentiles,
request volume, abstain rates, retrieval score distributions, theme breakdowns,
and user feedback directly from SQLite telemetry.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

logger = logging.getLogger("easypasal.collector.dashboard")

dashboard_router = APIRouter(tags=["dashboard"])


def get_dashboard_html() -> str:
    """Return the self-contained HTML/CSS/JavaScript telemetry dashboard."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>EasyPasal — Monitoring & Telemetry Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg-primary: #0a0e17;
      --bg-secondary: #131b2e;
      --bg-card: #1a233a;
      --bg-card-hover: #222d4a;
      --border: #263353;
      --text-primary: #f1f5f9;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
      --accent-blue: #38bdf8;
      --accent-teal: #2dd4bf;
      --accent-green: #34d399;
      --accent-amber: #fbbf24;
      --accent-rose: #f43f5e;
      --accent-purple: #a855f7;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg-primary);
      color: var(--text-primary);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }

    header {
      background-color: var(--bg-secondary);
      border-bottom: 1px solid var(--border);
      padding: 1rem 2rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 1rem;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .brand-badge {
      background: linear-gradient(135deg, #0284c7, #2563eb);
      color: white;
      font-weight: 700;
      font-size: 0.85rem;
      padding: 0.35rem 0.65rem;
      border-radius: 6px;
      letter-spacing: 0.05em;
    }
    .brand-title {
      font-size: 1.25rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }
    .brand-sub {
      color: var(--text-secondary);
      font-size: 0.85rem;
    }

    .controls {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      flex-wrap: wrap;
    }

    input, select, button {
      background-color: var(--bg-card);
      border: 1px solid var(--border);
      color: var(--text-primary);
      padding: 0.45rem 0.75rem;
      border-radius: 6px;
      font-size: 0.85rem;
      font-family: inherit;
      outline: none;
      transition: all 0.2s;
    }
    input:focus, select:focus {
      border-color: var(--accent-blue);
      box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2);
    }
    button {
      cursor: pointer;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
    }
    button.btn-primary {
      background: #0284c7;
      border-color: #0284c7;
      color: white;
    }
    button.btn-primary:hover {
      background: #0369a1;
    }

    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      font-size: 0.8rem;
      font-weight: 600;
      padding: 0.3rem 0.6rem;
      border-radius: 20px;
      background: rgba(52, 211, 153, 0.15);
      color: var(--accent-green);
      border: 1px solid rgba(52, 211, 153, 0.3);
    }
    .status-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background-color: currentColor;
    }

    main {
      padding: 1.5rem 2rem;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 1.5rem;
    }

    /* KPI metric cards */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 1rem;
    }
    .kpi-card {
      background-color: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.1rem;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
      transition: transform 0.2s, border-color 0.2s;
    }
    .kpi-card:hover {
      border-color: var(--accent-blue);
      transform: translateY(-2px);
    }
    .kpi-label {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-secondary);
      font-weight: 600;
    }
    .kpi-value {
      font-size: 1.65rem;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
      letter-spacing: -0.03em;
    }
    .kpi-meta {
      font-size: 0.75rem;
      color: var(--text-muted);
    }

    /* Charts grid */
    .charts-grid {
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 1.25rem;
    }
    .chart-card {
      background-color: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }
    .col-8 { grid-column: span 8; }
    .col-6 { grid-column: span 6; }
    .col-4 { grid-column: span 4; }
    .col-12 { grid-column: span 12; }

    @media (max-width: 1024px) {
      .col-8, .col-6, .col-4 { grid-column: span 12; }
    }

    .chart-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .chart-title {
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text-primary);
    }
    .chart-subtitle {
      font-size: 0.8rem;
      color: var(--text-secondary);
    }
    .chart-container {
      position: relative;
      flex: 1;
      min-height: 240px;
    }

    /* Table Section */
    .table-card {
      background-color: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    .table-wrapper {
      overflow-x: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      text-align: left;
    }
    th {
      background-color: var(--bg-secondary);
      color: var(--text-secondary);
      font-weight: 600;
      padding: 0.7rem 0.9rem;
      border-bottom: 1px solid var(--border);
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    td {
      padding: 0.75rem 0.9rem;
      border-bottom: 1px solid var(--border);
      color: var(--text-primary);
    }
    tr:hover td {
      background-color: var(--bg-card-hover);
    }
    .pill {
      display: inline-block;
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 600;
      font-family: 'JetBrains Mono', monospace;
    }
    .pill-variant { background: rgba(56, 189, 248, 0.15); color: var(--accent-blue); }
    .pill-theme { background: rgba(168, 85, 247, 0.15); color: var(--accent-purple); }
    .pill-abstain { background: rgba(244, 63, 94, 0.15); color: var(--accent-rose); }
    .pill-answered { background: rgba(52, 211, 153, 0.15); color: var(--accent-green); }

    footer {
      background-color: var(--bg-secondary);
      border-top: 1px solid var(--border);
      padding: 0.9rem 2rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      color: var(--text-muted);
      font-size: 0.8rem;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span class="brand-badge">TELEMETRY</span>
      <div>
        <div class="brand-title">EasyPasal Log Collector Dashboard</div>
        <div class="brand-sub">Real-Time VPS Ingestion & Observability (Plan §9 / Task T7.3)</div>
      </div>
    </div>
    <div class="controls">
      <div id="healthBadge" class="status-badge">
        <span class="status-dot"></span>
        <span id="healthText">Collector Healthy</span>
      </div>
      <input type="password" id="tokenInput" placeholder="Collector Auth Token" style="width: 170px;" />
      <select id="autoRefreshSelect">
        <option value="0">Auto-refresh: Off</option>
        <option value="5000">Every 5s</option>
        <option value="10000" selected>Every 10s</option>
        <option value="30000">Every 30s</option>
      </select>
      <button class="btn-primary" onclick="loadAllTelemetry()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.85.99 6.57 2.57L21 8"/><path d="M21 3v5h-5"/></svg>
        Refresh
      </button>
    </div>
  </header>

  <main>
    <!-- KPI Summary Row -->
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Total Requests</div>
        <div class="kpi-value" id="kpiTotalRequests" style="color: var(--accent-blue);">0</div>
        <div class="kpi-meta" id="kpiRequestsMeta">0 feedback logged</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Abstention Rate</div>
        <div class="kpi-value" id="kpiAbstainRate" style="color: var(--accent-amber);">0.0%</div>
        <div class="kpi-meta" id="kpiAbstainCount">0 explicit refusals</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Latency p50</div>
        <div class="kpi-value" id="kpiP50Latency" style="color: var(--accent-teal);">0 ms</div>
        <div class="kpi-meta" id="kpiMeanLatency">Mean: 0 ms</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Latency p95</div>
        <div class="kpi-value" id="kpiP95Latency" style="color: var(--accent-purple);">0 ms</div>
        <div class="kpi-meta" id="kpiP99Latency">p99: 0 ms</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Top-1 Score</div>
        <div class="kpi-value" id="kpiTop1Score" style="color: var(--accent-green);">-</div>
        <div class="kpi-meta">Avg retrieval score</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Memory (RSS)</div>
        <div class="kpi-value" id="kpiMemoryRss" style="color: var(--accent-blue);">0 MB</div>
        <div class="kpi-meta">Budget: &le; 90 MB</div>
      </div>
    </div>

    <!-- Charts Row 1: Latency & Volume -->
    <div class="charts-grid">
      <div class="chart-card col-8">
        <div class="chart-header">
          <div>
            <div class="chart-title">Latency Percentiles & Distribution</div>
            <div class="chart-subtitle">Request end-to-end response duration (ms)</div>
          </div>
        </div>
        <div class="chart-container">
          <canvas id="latencyChart"></canvas>
        </div>
      </div>
      <div class="chart-card col-4">
        <div class="chart-header">
          <div>
            <div class="chart-title">Pipeline Variant Split</div>
            <div class="chart-subtitle">A/B serving volume distribution</div>
          </div>
        </div>
        <div class="chart-container">
          <canvas id="variantChart"></canvas>
        </div>
      </div>
    </div>

    <!-- Charts Row 2: Retrieval Scores & Theme Distribution -->
    <div class="charts-grid">
      <div class="chart-card col-6">
        <div class="chart-header">
          <div>
            <div class="chart-title">Retrieval Top-1 Score Distribution</div>
            <div class="chart-subtitle">Quality anchor: low-evidence alerts (&lt; 0.60 score)</div>
          </div>
        </div>
        <div class="chart-container">
          <canvas id="retrievalChart"></canvas>
        </div>
      </div>
      <div class="chart-card col-6">
        <div class="chart-header">
          <div>
            <div class="chart-title">Regulatory Theme Frequency</div>
            <div class="chart-subtitle">Classified Indonesian financial legal domains</div>
          </div>
        </div>
        <div class="chart-container">
          <canvas id="themeChart"></canvas>
        </div>
      </div>
    </div>

    <!-- Table: Recent Requests -->
    <div class="table-card">
      <div class="chart-header">
        <div>
          <div class="chart-title">Recent Ingested Requests</div>
          <div class="chart-subtitle">Live audit stream with PII redaction and citation anchors</div>
        </div>
      </div>
      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Request ID</th>
              <th>Time (UTC)</th>
              <th>Variant</th>
              <th>Theme</th>
              <th>Score (Top-1)</th>
              <th>Latency (ms)</th>
              <th>Status</th>
              <th>Redacted Question</th>
            </tr>
          </thead>
          <tbody id="requestsTableBody">
            <tr><td colspan="8" style="text-align:center; color: var(--text-muted);">Loading logs...</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </main>

  <footer>
    <div>EasyPasal v0.1 &bull; SQLite WAL Persistent Log Sink</div>
    <div id="dbStatsText">DB: 0 KB &bull; Page Cache Capped at 4 MB</div>
  </footer>

  <script>
    // State & Chart instances
    let latencyChart = null;
    let variantChart = null;
    let retrievalChart = null;
    let themeChart = null;
    let refreshIntervalId = null;

    // Load auth token from localStorage if present
    const savedToken = localStorage.getItem("easypasal_collector_token") || "easypasal-vps-secret-token-v1";
    document.getElementById("tokenInput").value = savedToken;
    document.getElementById("tokenInput").addEventListener("change", (e) => {
      localStorage.setItem("easypasal_collector_token", e.target.value.trim());
      loadAllTelemetry();
    });

    document.getElementById("autoRefreshSelect").addEventListener("change", (e) => {
      setupAutoRefresh(parseInt(e.target.value, 10));
    });

    function getAuthHeaders() {
      const token = document.getElementById("tokenInput").value.trim();
      const headers = { "Content-Type": "application/json" };
      if (token) {
        headers["Authorization"] = `Bearer ${token}`;
      }
      return headers;
    }

    async function loadHealth() {
      try {
        const res = await fetch("/health");
        if (!res.ok) throw new Error("Health check failed");
        const data = await res.json();
        const mem = data.memory || {};
        document.getElementById("kpiMemoryRss").innerText = `${mem.current_rss_mb || 0} MB`;
        document.getElementById("healthText").innerText = mem.within_budget ? "VPS Budget OK" : "Memory High";

        const db = data.database || {};
        document.getElementById("dbStatsText").innerText = `DB: ${db.size_mb || 0} MB (${db.page_count || 0} pages) • Cache: 4 MB`;
      } catch (err) {
        document.getElementById("healthText").innerText = "Collector Unreachable";
        document.getElementById("healthBadge").style.color = "var(--accent-rose)";
      }
    }

    async function loadStatsSummary() {
      try {
        const res = await fetch("/v1/stats/summary", { headers: getAuthHeaders() });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // Update KPIs
        document.getElementById("kpiTotalRequests").innerText = data.total_requests || 0;
        document.getElementById("kpiRequestsMeta").innerText = `${data.total_feedback || 0} user feedback entries`;
        document.getElementById("kpiAbstainRate").innerText = `${((data.abstain_rate || 0) * 100).toFixed(1)}%`;
        document.getElementById("kpiAbstainCount").innerText = `${data.abstain_count || 0} explicit refusals`;
        document.getElementById("kpiP50Latency").innerText = `${Math.round(data.p50_latency_ms || 0)} ms`;
        document.getElementById("kpiMeanLatency").innerText = `Mean: ${Math.round(data.mean_latency_ms || 0)} ms`;
        document.getElementById("kpiP95Latency").innerText = `${Math.round(data.p95_latency_ms || 0)} ms`;
        document.getElementById("kpiP99Latency").innerText = `p99: ${Math.round(data.p99_latency_ms || 0)} ms`;
        document.getElementById("kpiTop1Score").innerText = data.mean_retrieval_score_top1 != null ? data.mean_retrieval_score_top1.toFixed(3) : "-";

        updateLatencyChart(data);
        updateVariantChart(data.variant_distribution || {});
        updateThemeChart(data.theme_distribution || {});
      } catch (err) {
        console.warn("Failed to fetch stats summary:", err);
      }
    }

    async function loadRecentRequests() {
      try {
        const res = await fetch("/v1/logs/requests?limit=15", { headers: getAuthHeaders() });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const items = data.items || [];
        const tbody = document.getElementById("requestsTableBody");

        if (items.length === 0) {
          tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; color: var(--text-muted);">No request logs ingested yet.</td></tr>';
          updateRetrievalScoreChart([]);
          return;
        }

        let html = "";
        const scores = [];
        for (const req of items) {
          const reqId = req.request_id.length > 14 ? req.request_id.substring(0, 12) + "..." : req.request_id;
          const timeStr = req.timestamp ? req.timestamp.replace("T", " ").substring(0, 19) : "-";
          const score = req.retrieval_score_top1 != null ? req.retrieval_score_top1.toFixed(3) : "-";
          if (req.retrieval_score_top1 != null) scores.push(req.retrieval_score_top1);

          const statusHtml = req.abstained
            ? '<span class="pill pill-abstain">Abstained</span>'
            : '<span class="pill pill-answered">Answered</span>';

          html += `<tr>
            <td><code style="color: var(--accent-blue);">${reqId}</code></td>
            <td style="color: var(--text-secondary);">${timeStr}</td>
            <td><span class="pill pill-variant">${req.variant}</span></td>
            <td><span class="pill pill-theme">${req.theme || "Unknown"}</span></td>
            <td style="font-family: monospace;">${score}</td>
            <td style="font-family: monospace;">${Math.round(req.total_latency_ms)}</td>
            <td>${statusHtml}</td>
            <td style="max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${req.question_redacted}</td>
          </tr>`;
        }
        tbody.innerHTML = html;
        updateRetrievalScoreChart(scores);
      } catch (err) {
        console.warn("Failed to fetch recent requests:", err);
      }
    }

    function updateLatencyChart(data) {
      const ctx = document.getElementById("latencyChart").getContext("2d");
      const labels = ["p50 (Median)", "Mean", "p95", "p99"];
      const values = [
        data.p50_latency_ms || 0,
        data.mean_latency_ms || 0,
        data.p95_latency_ms || 0,
        data.p99_latency_ms || 0,
      ];

      if (latencyChart) {
        latencyChart.data.datasets[0].data = values;
        latencyChart.update();
        return;
      }

      latencyChart = new Chart(ctx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [{
            label: "Latency (ms)",
            data: values,
            backgroundColor: ["#2dd4bf", "#38bdf8", "#a855f7", "#f43f5e"],
            borderRadius: 6,
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { grid: { color: "#263353" }, ticks: { color: "#94a3b8" } },
            x: { grid: { display: false }, ticks: { color: "#94a3b8" } }
          }
        }
      });
    }

    function updateVariantChart(varDist) {
      const ctx = document.getElementById("variantChart").getContext("2d");
      const labels = Object.keys(varDist);
      const values = Object.values(varDist);

      if (variantChart) {
        variantChart.data.labels = labels;
        variantChart.data.datasets[0].data = values;
        variantChart.update();
        return;
      }

      variantChart = new Chart(ctx, {
        type: "doughnut",
        data: {
          labels: labels.length ? labels : ["None"],
          datasets: [{
            data: values.length ? values : [1],
            backgroundColor: ["#38bdf8", "#a855f7", "#2dd4bf", "#fbbf24"],
            borderWidth: 0
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { position: "bottom", labels: { color: "#94a3b8" } } }
        }
      });
    }

    function updateRetrievalScoreChart(scores) {
      const ctx = document.getElementById("retrievalChart").getContext("2d");
      // Bin into 5 buckets: 0-0.5, 0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-1.0
      const bins = [0, 0, 0, 0, 0];
      for (const s of scores) {
        if (s < 0.5) bins[0]++;
        else if (s < 0.6) bins[1]++;
        else if (s < 0.7) bins[2]++;
        else if (s < 0.8) bins[3]++;
        else bins[4]++;
      }

      if (retrievalChart) {
        retrievalChart.data.datasets[0].data = bins;
        retrievalChart.update();
        return;
      }

      retrievalChart = new Chart(ctx, {
        type: "bar",
        data: {
          labels: ["< 0.50 (Low)", "0.50 - 0.59", "0.60 - 0.69", "0.70 - 0.79", ">= 0.80 (High)"],
          datasets: [{
            label: "Request Count",
            data: bins,
            backgroundColor: ["#f43f5e", "#fbbf24", "#38bdf8", "#2dd4bf", "#34d399"],
            borderRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { grid: { color: "#263353" }, ticks: { color: "#94a3b8", stepSize: 1 } },
            x: { grid: { display: false }, ticks: { color: "#94a3b8" } }
          }
        }
      });
    }

    function updateThemeChart(themeDist) {
      const ctx = document.getElementById("themeChart").getContext("2d");
      const labels = Object.keys(themeDist);
      const values = Object.values(themeDist);

      if (themeChart) {
        themeChart.data.labels = labels;
        themeChart.data.datasets[0].data = values;
        themeChart.update();
        return;
      }

      themeChart = new Chart(ctx, {
        type: "polarArea",
        data: {
          labels: labels.length ? labels : ["None"],
          datasets: [{
            data: values.length ? values : [1],
            backgroundColor: [
              "rgba(56, 189, 248, 0.6)",
              "rgba(168, 85, 247, 0.6)",
              "rgba(45, 212, 191, 0.6)",
              "rgba(251, 191, 36, 0.6)",
              "rgba(244, 63, 94, 0.6)"
            ],
            borderWidth: 0
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { position: "right", labels: { color: "#94a3b8" } } },
          scales: { r: { grid: { color: "#263353" }, ticks: { display: false } } }
        }
      });
    }

    function loadAllTelemetry() {
      loadHealth();
      loadStatsSummary();
      loadRecentRequests();
    }

    function setupAutoRefresh(ms) {
      if (refreshIntervalId) clearInterval(refreshIntervalId);
      if (ms > 0) {
        refreshIntervalId = setInterval(loadAllTelemetry, ms);
      }
    }

    // Initial fetch and auto-refresh setup
    loadAllTelemetry();
    setupAutoRefresh(10000);
  </script>
</body>
</html>
"""


@dashboard_router.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="Interactive Telemetry & Monitoring Dashboard",
)
def render_dashboard() -> HTMLResponse:
    """Serve the self-contained interactive web monitoring dashboard."""
    return HTMLResponse(content=get_dashboard_html())
