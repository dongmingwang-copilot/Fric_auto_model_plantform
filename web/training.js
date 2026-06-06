const api = {
  async get(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
};

const state = {
  jobs: [],
  selectedJobId: null,
  timer: null,
  nextCycleJobs: new Set(),
};

const pageParams = new URLSearchParams(window.location.search);

const el = {
  monitorHealth: document.getElementById("monitorHealth"),
  backHome: document.getElementById("backHome"),
  refreshJobs: document.getElementById("refreshJobs"),
  jobList: document.getElementById("jobList"),
  jobStatus: document.getElementById("jobStatus"),
  progressBar: document.getElementById("trainProgressBar"),
  metricEpoch: document.getElementById("metricEpoch"),
  metricBatch: document.getElementById("metricBatch"),
  metricLoss: document.getElementById("metricLoss"),
  metricDiceRecall: document.getElementById("metricDiceRecall"),
  metricPhase: document.getElementById("metricPhase"),
  metricSpeed: document.getElementById("metricSpeed"),
  metricGpu: document.getElementById("metricGpu"),
  metricCheckpoint: document.getElementById("metricCheckpoint"),
  metricChart: document.getElementById("metricChart"),
  visualizationRows: document.getElementById("visualizationRows"),
};

function returnUrl() {
  return pageParams.get("return") || "/";
}

function jobMatchesContext(job) {
  const datasetId = pageParams.get("dataset_id");
  const labelId = pageParams.get("label_id");
  const workspace = pageParams.get("workspace");
  if (datasetId && job.dataset_id !== datasetId) return false;
  if (labelId && job.label_id !== labelId) return false;
  if (workspace && job.project_type && job.project_type !== workspace) return false;
  return true;
}

function hasContextFilter() {
  return Boolean(pageParams.get("dataset_id") || pageParams.get("label_id") || pageParams.get("workspace"));
}

function statusText(status) {
  const map = {
    queued: "排队中",
    running: "训练中",
    completed: "已完成",
    failed: "失败",
  };
  return map[status] || status || "-";
}

function phaseText(phase) {
  const map = {
    prepare: "加载数据",
    ready: "准备完成",
    batch: "训练 batch",
    evaluate: "验证评估",
    epoch_end: "轮次完成",
  };
  return map[phase] || phase || "-";
}

function fmt(value, digits = 4) {
  return Number.isFinite(value) ? value.toFixed(digits) : "-";
}

function checkpointIdForJob(job) {
  return `run-${job.id}-best`;
}

function trainingScopeText(scope) {
  if (scope === "all_reviewed_annotations") return "当前数据集全部已 Review 样本";
  return scope || "未记录";
}

function progressPercent(progress, job) {
  if (job.status === "completed") return 100;
  if (!progress) return 0;
  const epochBase = Math.max((Number(progress.epoch || 1) - 1) / Math.max(Number(progress.epochs || 1), 1), 0);
  const batchPart = Number(progress.batches || 0) > 0
    ? Number(progress.batch || 0) / Number(progress.batches)
    : progress.phase === "epoch_end" ? 1 : 0;
  return Math.max(0, Math.min(99.5, (epochBase + batchPart / Math.max(Number(progress.epochs || 1), 1)) * 100));
}

async function loadJobs() {
  const rows = await api.get("/api/training/jobs");
  const contextual = rows.filter(jobMatchesContext);
  state.jobs = hasContextFilter() ? contextual : rows;
  renderJobs();
  if (!state.selectedJobId && state.jobs.length) {
    await selectJob(state.jobs[0].id);
  } else if (state.selectedJobId && !state.jobs.some((job) => job.id === state.selectedJobId)) {
    state.selectedJobId = state.jobs.length ? state.jobs[0].id : null;
    if (state.selectedJobId) await selectJob(state.selectedJobId);
  } else if (!state.jobs.length) {
    renderEmpty();
  }
  el.monitorHealth.textContent = `已同步 · ${state.jobs.length} 个任务`;
}

function renderJobs() {
  el.jobList.innerHTML = "";
  if (!state.jobs.length) {
    el.jobList.innerHTML = '<div class="muted">暂无训练任务</div>';
    return;
  }
  for (const job of state.jobs) {
    const row = document.createElement("div");
    row.className = `job-row ${job.id === state.selectedJobId ? "active" : ""}`;
    row.onclick = () => selectJob(job.id);
    const progress = job.progress;
    const detail = progress
      ? `${phaseText(progress.phase)} · ${progress.epoch || 0}/${progress.epochs || job.params?.epochs || "-"}`
      : `${job.label_id || "spall"} · ${job.params?.epochs || "-"} 轮`;
    row.innerHTML = `
      <strong>${job.id}</strong>
      <div class="muted">${statusText(job.status)} · ${detail}</div>
    `;
    el.jobList.appendChild(row);
  }
}

function renderEmpty() {
  el.jobStatus.textContent = "暂无训练任务";
  el.metricEpoch.textContent = "-";
  el.metricBatch.textContent = "-";
  el.metricLoss.textContent = "-";
  el.metricDiceRecall.textContent = "-";
  el.metricPhase.textContent = "-";
  el.metricSpeed.textContent = "-";
  el.metricGpu.textContent = "-";
  el.metricCheckpoint.textContent = "-";
  el.progressBar.style.width = "0%";
  el.visualizationRows.innerHTML = "";
  drawChart([]);
}

function drawChart(history) {
  const canvas = el.metricChart;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#cbd3dc";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = 28 + i * 46;
    ctx.beginPath();
    ctx.moveTo(48, y);
    ctx.lineTo(canvas.width - 18, y);
    ctx.stroke();
  }
  if (!history.length) {
    ctx.fillStyle = "#697785";
    ctx.fillText("等待第一个 epoch 完成；batch 进度在上方实时刷新", 54, 132);
    return;
  }
  const plot = (key, color) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    history.forEach((row, idx) => {
      const value = key === "loss" ? row.train_loss : row.val?.[key] ?? 0;
      const norm = key === "loss" ? Math.max(0, Math.min(1, value / 3)) : Math.max(0, Math.min(1, value));
      const x = 54 + (idx / Math.max(history.length - 1, 1)) * (canvas.width - 88);
      const y = canvas.height - 30 - norm * (canvas.height - 64);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  };
  plot("loss", "#6d5bd0");
  plot("dice", "#147d64");
  plot("recall", "#c73532");
  ctx.fillStyle = "#6d5bd0";
  ctx.fillText("loss", 62, 20);
  ctx.fillStyle = "#147d64";
  ctx.fillText("dice", 112, 20);
  ctx.fillStyle = "#c73532";
  ctx.fillText("recall", 162, 20);
}

async function selectJob(jobId) {
  state.selectedJobId = jobId;
  renderJobs();
  if (state.timer) clearInterval(state.timer);
  await refreshSelectedJob();
  state.timer = setInterval(() => refreshSelectedJob().catch(console.error), 1000);
}

function historyWithLiveProgress(metrics) {
  const history = [...(metrics.history || [])];
  const progress = metrics.progress;
  if (progress && progress.phase !== "epoch_end" && Number.isFinite(progress.train_loss)) {
    history.push({
      epoch: progress.epoch,
      train_loss: progress.train_loss,
      val: progress.val || null,
      live: true,
    });
  }
  return history;
}

async function refreshSelectedJob() {
  if (!state.selectedJobId) return;
  const [job, metrics] = await Promise.all([
    api.get(`/api/training/jobs/${state.selectedJobId}`),
    api.get(`/api/training/jobs/${state.selectedJobId}/metrics`),
  ]);
  const progress = metrics.progress || job.progress || null;
  const history = historyWithLiveProgress(metrics);
  const last = history[history.length - 1] || progress || null;
  const percent = progressPercent(progress, job);
  const batchText = progress?.batches ? `${progress.batch || 0}/${progress.batches}` : "-";
  const scope = job.training_scope || job.metrics?.training_scope || metrics.training_scope;
  const queueText = job.active_batch_id ? ` · 队列 ${job.active_batch_id}` : "";
  const checkpointText = metrics.checkpoint_available ? `可用：${checkpointIdForJob(job)}` : "训练完成前不可用";
  el.jobStatus.textContent = `${statusText(job.status)} · ${job.id} · ${trainingScopeText(scope)}${queueText}${job.error ? ` · ${job.error}` : ""}`;
  el.metricEpoch.textContent = progress ? `${progress.epoch || 0}/${progress.epochs || job.params?.epochs || "-"}` : last ? `${last.epoch}` : "-";
  el.metricBatch.textContent = batchText;
  el.metricLoss.textContent = last ? fmt(last.train_loss) : "-";
  el.metricDiceRecall.textContent = last?.val ? `${fmt(last.val.dice)} / ${fmt(last.val.recall)}` : "-";
  el.metricPhase.textContent = phaseText(progress?.phase || (job.status === "completed" ? "completed" : job.status));
  el.metricSpeed.textContent = progress?.batches_per_second ? `${fmt(progress.batches_per_second, 2)} batch/s` : "-";
  el.metricGpu.textContent = progress?.gpu_memory_reserved_mb
    ? `${progress.gpu_memory_allocated_mb || 0}/${progress.gpu_memory_reserved_mb} MB`
    : "-";
  el.metricCheckpoint.textContent = checkpointText;
  el.progressBar.style.width = `${percent}%`;
  drawChart(history);
  if (job.status === "completed") {
    await renderVisualizations(job.id);
    await advanceActiveLearningCycle(job);
  } else {
    el.visualizationRows.innerHTML = '<div class="muted">训练完成后生成测试图像。</div>';
  }
}

async function advanceActiveLearningCycle(job) {
  if (!job.dataset_id) return;
  const cycleKey = `next-cycle-view:${job.dataset_id}:${job.id}`;
  if (state.nextCycleJobs.has(cycleKey)) return;
  state.nextCycleJobs.add(cycleKey);
  for (let attempt = 0; attempt < 20; attempt += 1) {
    const current = await api.get(`/api/training/jobs/${job.id}`);
    if (current.next_cycle) {
      const next = current.next_cycle;
      if (next.status === "created") {
        el.monitorHealth.textContent = `后端已生成下一轮队列：${next.batch_id}`;
      } else if (next.status === "no_candidates") {
        el.monitorHealth.textContent = "训练完成，没有未 Review 候选";
      } else {
        el.monitorHealth.textContent = `下一轮状态：${next.status}`;
      }
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  el.monitorHealth.textContent = "训练完成，等待后端生成下一轮状态";
}

async function renderVisualizations(jobId) {
  const rows = await api.get(`/api/training/jobs/${jobId}/visualizations`);
  el.visualizationRows.innerHTML = "";
  if (!rows.length) {
    el.visualizationRows.innerHTML = '<div class="muted">暂无测试图像。</div>';
    return;
  }
  for (const row of rows) {
    const div = document.createElement("div");
    div.className = "viz-row";
    div.innerHTML = `
      <div class="viz-cell"><span>原图</span><img src="/api/training/jobs/${jobId}/visualizations/${row.original}" /></div>
      <div class="viz-cell"><span>人工 GT</span><img src="/api/training/jobs/${jobId}/visualizations/${row.gt}" /></div>
      <div class="viz-cell"><span>模型预测 · D ${fmt(row.dice)} R ${fmt(row.recall)}</span><img src="/api/training/jobs/${jobId}/visualizations/${row.pred}" /></div>
    `;
    el.visualizationRows.appendChild(div);
  }
}

el.backHome.onclick = () => {
  window.location.href = returnUrl();
};

el.refreshJobs.onclick = () => {
  loadJobs().catch(console.error);
};

loadJobs().catch((err) => {
  el.monitorHealth.textContent = "同步失败";
  console.error(err);
});
