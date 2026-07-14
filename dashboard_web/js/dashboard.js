const state = { chart: null };
const dataPath = "data/results.json";

const escapeHtml = (value) =>
  String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
const percent = (value) => `${((Number(value) || 0) * 100).toFixed(1)}%`;
const statCard = (label, value) => `<article class="stat"><div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(value)}</div></article>`;
const emptyMatrix = () => Array.from({ length: 5 }, () => Array(5).fill(0));

function setStatus(message, isError = false) {
  const banner = document.getElementById("status-banner");
  banner.textContent = message;
  banner.classList.toggle("hidden", !message);
  banner.style.background = isError ? "#fff3f5" : "#effaf5";
  banner.style.borderColor = isError ? "#f1ccd5" : "#d2e9db";
  banner.style.color = isError ? "#8e2435" : "#1c7a48";
}

async function loadDashboardData() {
  const response = await fetch(dataPath, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load dashboard data (${response.status})`);
  }
  return response.json();
}

function updateHeader(data) {
  document.getElementById("project-pill").textContent = `Project: ${data.project || "MediTriageAI"}`;
  document.getElementById("updated-pill").textContent = `Last updated: ${data.last_updated || "unknown"}`;
  document.getElementById("model-pill").textContent = `Models: ${(data.models || []).length}`;
}

function renderStats(stats) {
  const items = [
    ["Total rows", stats.total_rows ?? 0],
    ["Train / Val / Test", `${stats.train_rows ?? 0} / ${stats.val_rows ?? 0} / ${stats.test_rows ?? 0}`],
    ["Departments", stats.departments ?? 0],
    ["Severity levels", stats.severity_levels ?? 0],
  ];
  document.getElementById("stats-grid").innerHTML = items.map(([label, value]) => statCard(label, value)).join("");
}

function renderLeaderboard(models) {
  const rows = [...models].sort((left, right) => (Number(right.specialist_f1) || 0) - (Number(left.specialist_f1) || 0));
  document.getElementById("leaderboard-body").innerHTML = rows.map((model) => `
    <tr>
      <td>${escapeHtml(model.name)}</td>
      <td>${percent(model.specialist_f1)}</td>
      <td>${percent(model.severity_f1)}</td>
      <td>${model.is_novel ? "Novel" : "Baseline"}</td>
    </tr>
  `).join("");
}

function renderSummary(summary) {
  document.getElementById("novelty-summary").textContent = summary || "No model results have been exported yet.";
}

function renderHeatmap(models) {
  const selected = [...models].find((model) => model.is_novel) || models[0];
  const matrix = selected?.severity_confusion_matrix?.length ? selected.severity_confusion_matrix : emptyMatrix();
  document.getElementById("heatmap-label").textContent = selected ? `Showing ${selected.name}` : "No heatmap data available yet.";
  const maxValue = Math.max(1, ...matrix.flat());
  let markup = '<div class="heatmap-row"><div class="heatmap-head"></div>' +
    ["S1", "S2", "S3", "S4", "S5"].map((label) => `<div class="heatmap-head">${label}</div>`).join("") +
    "</div>";
  matrix.forEach((row, rowIndex) => {
    markup += '<div class="heatmap-row">';
    markup += `<div class="heatmap-cell heatmap-label">S${rowIndex + 1}</div>`;
    row.forEach((value) => {
      const intensity = value / maxValue;
      const red = Math.round(238 - intensity * 120);
      const green = Math.round(246 - intensity * 70);
      markup += `<div class="heatmap-cell heatmap-value" style="background: rgb(${red}, ${green}, 255)">${value}</div>`;
    });
    markup += "</div>";
  });
  document.getElementById("heatmap").innerHTML = markup;
}

function renderChart(models) {
  const canvas = document.getElementById("comparison-chart");
  const fallback = document.getElementById("chart-fallback");
  const labels = models.map((model) => model.name);
  const specialist = models.map((model) => Number(model.specialist_f1) || 0);
  const severity = models.map((model) => Number(model.severity_f1) || 0);

  if (!window.Chart) {
    fallback.classList.remove("hidden");
    fallback.innerHTML = labels.map((label, index) => `
      <div class="fallback-row">
        <div>${escapeHtml(label)}</div>
        <div class="fallback-bar"><span style="width:${specialist[index] * 100}%"></span></div>
        <div class="fallback-bar"><span style="width:${severity[index] * 100}%"></span></div>
        <div>${percent(Math.max(specialist[index], severity[index]))}</div>
      </div>
    `).join("");
    canvas.classList.add("hidden");
    return;
  }

  fallback.classList.add("hidden");
  canvas.classList.remove("hidden");
  if (state.chart) state.chart.destroy();
  state.chart = new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [
        { label: "Specialist F1", data: specialist, backgroundColor: "#1566b3" },
        { label: "Severity F1", data: severity, backgroundColor: "#0d8d8f" },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "bottom" } },
      scales: {
        y: { beginAtZero: true, max: 1, ticks: { callback: (value) => `${Math.round(value * 100)}%` } },
      },
    },
  });
}

async function init() {
  try {
    const data = await loadDashboardData();
    updateHeader(data);
    renderStats(data.dataset_stats || {});
    renderLeaderboard(data.models || []);
    renderSummary(data.novelty_summary || "");
    renderHeatmap(data.models || []);
    renderChart(data.models || []);
  } catch (error) {
    setStatus(`Dashboard failed to load: ${error.message}`, true);
    document.getElementById("novelty-summary").textContent = "Dashboard data is unavailable.";
  }
}

init();
