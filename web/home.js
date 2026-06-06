const homeApi = {
  async get(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  async post(url, body = {}) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
  async delete(url) {
    const res = await fetch(url, { method: "DELETE" });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  },
};

const homeEl = {
  health: document.getElementById("homeHealth"),
  metricDatasets: document.getElementById("metricDatasets"),
  metricDatasetsHint: document.getElementById("metricDatasetsHint"),
  metricModels: document.getElementById("metricModels"),
  metricModelsHint: document.getElementById("metricModelsHint"),
  metricJobs: document.getElementById("metricJobs"),
  metricJobsHint: document.getElementById("metricJobsHint"),
  metricArchives: document.getElementById("metricArchives"),
  metricArchivesHint: document.getElementById("metricArchivesHint"),
  modelCenter: document.getElementById("modelCenter"),
  datasetGovernance: document.getElementById("datasetGovernance"),
  qualityGates: document.getElementById("qualityGates"),
  auditEvents: document.getElementById("auditEvents"),
  operationRunbook: document.getElementById("operationRunbook"),
};

let homeState = {
  health: null,
  models: [],
  catalog: [],
  jobs: [],
  archives: [],
  deployments: [],
  auditEvents: [],
};

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function modelStageText(model) {
  const map = {
    curated_baseline: "当前基线",
    generated_baseline: "生成基线",
    generation_run: "生成训练",
    optimization_run: "升级模型",
  };
  if (model.role === "foundation") return "初始权重";
  return map[model.model_stage] || model.role || "模型";
}

function projectTypeText(value) {
  if (value === "generation") return "模型生成";
  if (value === "optimization") return "模型升级";
  return value || "-";
}

function datasetStatusText(status) {
  const map = {
    empty: "空项目",
    imported: "已导入",
    predicted: "已预测",
    reviewing: "Review 中",
    baseline_ready: "基线完成",
    review_complete: "Review 完成",
  };
  return map[status] || status || "-";
}

function auditActionText(action) {
  const map = {
    "deployment.create": "生成部署包",
    "deployment.delete": "删除部署包",
  };
  return map[action] || action || "操作";
}

function formatTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("zh-CN", { hour12: false });
}

function modelActionUrl(model) {
  const params = new URLSearchParams();
  if (model.label_id) params.set("label_id", model.label_id);
  if (model.id) params.set("checkpoint_id", model.id);
  if (model.model_stage === "generated_baseline" || model.role === "baseline") {
    return `/optimize?${params.toString()}`;
  }
  if (model.role === "foundation") return "/generate";
  return `/testing?${params.toString()}`;
}

function deploymentFor(model) {
  return homeState.deployments.find((pkg) => pkg.checkpoint_id === model.id) || null;
}

function renderModelCenter(models) {
  if (!models.length) {
    homeEl.modelCenter.innerHTML = '<div class="muted">暂无可用模型，请检查 checkpoints 目录。</div>';
    return;
  }
  const sorted = [...models].sort((a, b) => {
    const priority = (model) => {
      if (model.id === "baseline-spall-unet-recall-v1") return 0;
      if (model.role === "foundation") return 2;
      return 1;
    };
    return priority(a) - priority(b);
  });
  homeEl.modelCenter.innerHTML = sorted.map((model) => {
    const pkg = deploymentFor(model);
    const label = model.label_name || model.label_id || "未绑定";
    return `
      <div class="model-asset ${model.id === "baseline-spall-unet-recall-v1" ? "primary" : ""}">
        <div class="model-asset-head">
          <span class="badge ${escapeHtml(model.model_stage || model.role)}">${modelStageText(model)}</span>
          <small>${escapeHtml(model.model_type || "unet")} · 阈值 ${Number(model.threshold_default ?? 0.5).toFixed(2)}</small>
        </div>
        <strong>${escapeHtml(model.name || model.id)}</strong>
        <p>${escapeHtml(model.description || "训练模型资产，可进入主动学习闭环或模型测试。")}</p>
        <dl>
          <div><dt>模型 ID</dt><dd>${escapeHtml(model.id)}</dd></div>
          <div><dt>缺陷类别</dt><dd>${escapeHtml(label)}</dd></div>
          <div><dt>阶段</dt><dd>${projectTypeText(model.project_type)}</dd></div>
          <div><dt>部署包</dt><dd>${pkg ? escapeHtml(pkg.id) : "未生成"}</dd></div>
          <div><dt>文件</dt><dd title="${escapeHtml(model.path || "")}">${escapeHtml((model.path || "").split(/[\\/]/).pop() || "-")}</dd></div>
          <div><dt>交付状态</dt><dd>${pkg ? escapeHtml(pkg.status) : "待准备"}</dd></div>
        </dl>
        <div class="model-asset-actions">
          <a class="button-link primary" href="${modelActionUrl(model)}">${model.role === "foundation" ? "创建新类别" : "进入闭环"}</a>
          <a class="button-link" href="/testing?checkpoint_id=${encodeURIComponent(model.id)}${model.label_id ? `&label_id=${encodeURIComponent(model.label_id)}` : ""}">验证</a>
          ${model.role !== "foundation" && !pkg ? `<button class="deploy-button" data-checkpoint-id="${escapeHtml(model.id)}">生成部署包</button>` : ""}
          ${pkg ? `<button class="delete-deploy-button" data-deployment-id="${escapeHtml(pkg.id)}">删除部署包</button>` : ""}
        </div>
      </div>
    `;
  }).join("");
}

function renderDatasets(rows) {
  if (!rows.length) {
    homeEl.datasetGovernance.innerHTML = `
      <div class="empty-governance">
        <strong>暂无活动数据集</strong>
        <span>可以从数据集管理平台导入小样本；系统会建立图像、GT、预测掩码和事件记录。</span>
        <a class="button-link primary" href="/datasets">创建数据集</a>
      </div>
    `;
    return;
  }
  homeEl.datasetGovernance.innerHTML = `
    <table>
      <thead>
        <tr><th>项目</th><th>类别</th><th>阶段</th><th>状态</th><th>数量</th><th>最好 PT</th></tr>
      </thead>
      <tbody>
        ${rows.map((row) => `
          <tr>
            <td><a href="/datasets?dataset_id=${encodeURIComponent(row.dataset_id)}">${escapeHtml(row.project_name || row.dataset_id)}</a></td>
            <td>${escapeHtml(row.defect_class || row.label_name || "-")}</td>
            <td>${projectTypeText(row.project_type)}</td>
            <td><span class="badge ${escapeHtml(row.status)}">${datasetStatusText(row.status)}</span></td>
            <td>${Number(row.reviewed || 0)}/${Number(row.count || 0)}</td>
            <td title="${escapeHtml(row.best_pt || "")}">${escapeHtml((row.best_pt || "").split(/[\\/]/).pop() || "-")}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderGates({ health, catalog, models, jobs, archives, deployments, auditEvents }) {
  const activeBaseline = models.some((model) => model.id === "baseline-spall-unet-recall-v1");
  const activeDatasets = catalog.length;
  const runningJobs = jobs.filter((job) => ["queued", "running"].includes(job.status)).length;
  const baselineDeployReady = deployments.some((pkg) => pkg.checkpoint_id === "baseline-spall-unet-recall-v1");
  const gates = [
    { name: "基础服务", status: health?.ok ? "pass" : "fail", detail: health?.ok ? `设备 ${health.device}` : "服务不可用" },
    { name: "当前基线", status: activeBaseline ? "pass" : "fail", detail: activeBaseline ? "Spall recall-v1 已注册" : "缺少 Spall 基线" },
    { name: "数据清洁", status: activeDatasets === 0 ? "ready" : "pass", detail: activeDatasets === 0 ? "当前无活动数据集，适合导入测试集" : `${activeDatasets} 个活动数据集` },
    { name: "训练队列", status: runningJobs === 0 ? "ready" : "warn", detail: runningJobs === 0 ? "无运行中任务" : `${runningJobs} 个任务运行中` },
    { name: "归档区", status: archives.length === 0 ? "ready" : "pass", detail: archives.length === 0 ? "无归档项目" : `${archives.length} 个归档项目` },
    { name: "审计追踪", status: auditEvents.length ? "pass" : "ready", detail: auditEvents.length ? `最近 ${auditEvents.length} 条关键事件` : "等待关键操作写入审计" },
    { name: "部署准备", status: baselineDeployReady ? "pass" : "ready", detail: baselineDeployReady ? "当前基线已有部署准备包" : "可从模型中心生成部署包" },
  ];
  homeEl.qualityGates.innerHTML = gates.map((gate) => `
    <div class="gate-row ${gate.status}">
      <span></span>
      <strong>${escapeHtml(gate.name)}</strong>
      <small>${escapeHtml(gate.detail)}</small>
    </div>
  `).join("");
}

function renderAuditEvents(events) {
  if (!events.length) {
    homeEl.auditEvents.innerHTML = `
      <div class="empty-audit">
        <strong>暂无审计事件</strong>
        <span>生成或删除部署包后，这里会记录操作、资源、状态和时间。</span>
      </div>
    `;
    return;
  }
  homeEl.auditEvents.innerHTML = events.map((event) => {
    const checkpoint = event.payload?.checkpoint_name || event.payload?.checkpoint_id || "";
    return `
      <div class="audit-event">
        <span class="badge ${escapeHtml(event.status)}">${escapeHtml(event.status || "success")}</span>
        <div>
          <strong>${escapeHtml(auditActionText(event.action))}</strong>
          <small>${escapeHtml(event.resource_id)}${checkpoint ? ` · ${escapeHtml(checkpoint)}` : ""}</small>
        </div>
        <time>${escapeHtml(formatTime(event.created_at))}</time>
      </div>
    `;
  }).join("");
}

function renderRunbook({ health, catalog, models, jobs, archives, deployments }) {
  const hasBaseline = models.some((model) => model.id === "baseline-spall-unet-recall-v1");
  const hasFoundation = models.some((model) => model.role === "foundation");
  const hasBaselineDeployment = deployments.some((pkg) => pkg.checkpoint_id === "baseline-spall-unet-recall-v1");
  const runningJobs = jobs.filter((job) => ["queued", "running"].includes(job.status)).length;
  const rows = [
    { title: "导入测试数据集", entry: "/datasets", action: "创建项目、缺陷类别和标签颜色，再导入图像目录。", verify: "首页活动数据集 +1；数据集管理可看到图片、GT 和状态。", state: catalog.length === 0 ? "ready" : "active", stateText: catalog.length === 0 ? "可开始" : `${catalog.length} 个活动数据集` },
    { title: "使用当前基线升级", entry: "/optimize", action: "以 Spall recall-v1 进行全局预测、主动排序和 Review。", verify: "预测掩码同步到数据集；Review 后 GT 掩码同步。", state: hasBaseline ? "ready" : "blocked", stateText: hasBaseline ? "基线可用" : "缺少基线" },
    { title: "小样本生成新模型", entry: "/generate", action: "以 Scratch 权重开始，小样本 Review 后多轮主动学习。", verify: "全部 Review 后生成基线模型，并可进入升级工作台。", state: hasFoundation ? "ready" : "blocked", stateText: hasFoundation ? "初始权重可用" : "缺少初始权重" },
    { title: "训练监控与指标复核", entry: "/training", action: "检查每轮 Dice、Recall、IoU、Specificity 和最佳阈值。", verify: "任务完成后 metrics.json 与 best.pt 写入最佳阈值。", state: runningJobs === 0 ? "ready" : "active", stateText: runningJobs === 0 ? "队列空闲" : `${runningJobs} 个任务运行中` },
    { title: "生成部署准备包", entry: "/", action: "在模型中心生成 manifest、README 和模型副本。", verify: "部署门禁通过；部署包可删除并恢复清洁。", state: hasBaselineDeployment ? "active" : "ready", stateText: hasBaselineDeployment ? "已有部署包" : "可生成" },
    { title: "模型测试与清理", entry: "/testing", action: "抽样验证模型表现，测试结束删除临时数据集。", verify: "catalog、datasets、archives、events 回到 0。", state: catalog.length === 0 && archives.length === 0 && health?.ok ? "ready" : "active", stateText: catalog.length === 0 && archives.length === 0 ? "清洁" : "待清理" },
  ];
  homeEl.operationRunbook.innerHTML = rows.map((row) => `
    <a class="runbook-card ${row.state}" href="${row.entry}">
      <div><strong>${escapeHtml(row.title)}</strong><span>${escapeHtml(row.stateText)}</span></div>
      <p>${escapeHtml(row.action)}</p>
      <small>${escapeHtml(row.verify)}</small>
    </a>
  `).join("");
}

async function refreshHome() {
  try {
    const [health, models, catalog, jobs, archives, deployments, auditEvents] = await Promise.all([
      homeApi.get("/api/health"),
      homeApi.get("/api/checkpoints"),
      homeApi.get("/api/datasets/catalog"),
      homeApi.get("/api/training/jobs"),
      homeApi.get("/api/dataset-archives"),
      homeApi.get("/api/deployments"),
      homeApi.get("/api/audit/events?limit=8"),
    ]);
    homeState = { health, models, catalog, jobs, archives, deployments, auditEvents };
    homeEl.health.textContent = `就绪 · ${health.device || "cpu"}`;
    homeEl.metricDatasets.textContent = catalog.length;
    homeEl.metricDatasetsHint.textContent = catalog.length ? "活动项目" : "等待导入";
    homeEl.metricModels.textContent = models.length;
    homeEl.metricModelsHint.textContent = models.some((model) => model.id === "baseline-spall-unet-recall-v1")
      ? `含当前 Spall 基线 · 部署包 ${deployments.length}`
      : "缺少当前基线";
    homeEl.metricJobs.textContent = jobs.length;
    homeEl.metricJobsHint.textContent = jobs.some((job) => ["queued", "running"].includes(job.status)) ? "有任务运行" : "队列空闲";
    homeEl.metricArchives.textContent = archives.length;
    homeEl.metricArchivesHint.textContent = archives.length ? "可恢复项目" : "无归档";
    renderModelCenter(models);
    renderDatasets(catalog);
    renderGates(homeState);
    renderAuditEvents(auditEvents);
    renderRunbook(homeState);
  } catch (err) {
    homeEl.health.textContent = `同步失败：${err.message}`;
    homeEl.modelCenter.innerHTML = `<div class="muted">同步失败：${escapeHtml(err.message)}</div>`;
  }
}

homeEl.modelCenter.addEventListener("click", async (evt) => {
  const deployButton = evt.target.closest(".deploy-button");
  const deleteButton = evt.target.closest(".delete-deploy-button");
  if (!deployButton && !deleteButton) return;
  evt.preventDefault();
  const button = deployButton || deleteButton;
  button.disabled = true;
  button.textContent = deployButton ? "生成中" : "删除中";
  try {
    if (deployButton) {
      await homeApi.post("/api/deployments", {
        checkpoint_id: deployButton.dataset.checkpointId,
        target: "torch_package",
        note: "Created from enterprise home model center",
      });
    } else {
      await homeApi.delete(`/api/deployments/${encodeURIComponent(deleteButton.dataset.deploymentId)}`);
    }
    await refreshHome();
  } catch (err) {
    homeEl.health.textContent = `部署操作失败：${err.message}`;
    button.disabled = false;
  }
});

refreshHome();
