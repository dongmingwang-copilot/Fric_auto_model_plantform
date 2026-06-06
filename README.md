# Fric Auto Model Plantform

Fric Auto Model Plantform 是一个本地运行的金属表面缺陷主动学习平台，用于完成数据集导入、模型预测、人工 Review、训练迭代、模型优化和审计闭环。

平台当前聚焦二值分割场景：用户导入缺陷图像后，系统基于当前 checkpoint 进行预测和主动学习排序，专业人员只 Review 高价值样本；训练任务只使用已 Review 的人工 mask，训练完成后再刷新未 Review 图像并生成下一轮队列。

## 核心能力

- 数据集管理：统一维护项目名称、缺陷类别、标签颜色、数据集阶段和图像导入。
- 模型生成：从 `UNet-32 Scratch` 初始权重和小样本人工 Review 开始，生成新的缺陷类别基线模型。
- 模型优化：接入更大数据集，基于已有模型进行预测、Review 和微调升级。
- 主动学习：按不确定性、熵、预测面积、连通域和样本多样性生成 Review 队列。
- 训练监控：实时显示训练状态、batch、吞吐、GPU 显存、曲线和 checkpoint 可用状态。
- 审计追踪：记录数据集创建、导入、归档、删除、预测、Review、训练和部署等关键操作。
- 模型门禁：训练中的 `best.pt` 不会提前暴露，只有任务完成后的 checkpoint 才能被工作台调用。

## 仓库内容

```text
app/                    FastAPI 后端服务
web/                    前端页面和交互逻辑
docs/                   工作流、部署和审计说明
integrations/           环境集成说明
manual_assets/          中文操作手册截图素材
scripts/                发布打包脚本
checkpoints/baseline/   初始模型权重
checkpoints/registry.json
storage/                本地运行目录，仓库仅保留空目录占位
```

## 模型文件策略

GitHub 仓库只上传初始模型：

- 已上传：`checkpoints/baseline/unet32_scratch.pt`
- 不上传：业务 baseline、训练 run、用户微调模型、导出部署包

后续训练生成的模型会写入 `checkpoints/runs/`，该目录已被 `.gitignore` 排除。

## 快速启动

推荐在 Windows PowerShell 中运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 7860
```

浏览器打开：

```text
http://127.0.0.1:7860/
```

后续也可以直接执行：

```powershell
.\run.ps1
```

如果需要 CUDA 加速，请提前安装与本机显卡驱动匹配的 CUDA 版 PyTorch。

## 推荐流程

1. 进入首页，打开数据集管理平台。
2. 创建项目，填写缺陷类别和标签颜色。
3. 导入待 Review 图像文件夹。
4. 进入模型生成或模型优化工作台。
5. 对全数据集执行模型预测，或在模型生成第一轮生成空白主动学习队列。
6. 按主动学习队列逐张人工 Review 并保存 mask。
7. 完成本轮 Review 后创建训练任务。
8. 在实时训练监控页查看进度和指标。
9. 训练完成后使用新 checkpoint 刷新未 Review 图像，进入下一轮主动学习。

## 数据边界

- `storage/datasets/`：平台导入的数据集副本、预测结果和人工 Review mask。
- `storage/categories/`：缺陷类别目录和类别元数据。
- `storage/dataset_archives/`：归档数据集。
- `storage/audit_logs/`：关键操作审计事件。
- `storage/training_jobs/`：训练任务状态和指标。
- `checkpoints/runs/`：用户后续训练得到的新模型。

这些目录中的运行数据默认不会提交到 GitHub。删除平台数据集只删除平台内的导入副本和衍生文件，不会删除用户原始图片文件夹。

## 发布包

需要生成可发给用户的 zip 包时，在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\package_release.ps1
```

发布包会输出到 `release/`，并自动排除缓存、日志、训练 run 和本地数据。详细使用方法见 `RELEASE_CN.md`。

## 许可

本仓库沿用项目根目录中的 `LICENSE`。
