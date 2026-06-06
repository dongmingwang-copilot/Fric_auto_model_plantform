# Plantform v1 中文使用说明

发布日期：2026-06-06

## 1. 交付内容

本发布包用于本地部署金属表面缺陷主动学习平台。默认包包含：

- 后端服务：`app/`
- 前端界面：`web/`
- 初始模型权重：`checkpoints/baseline/unet32_scratch.pt`
- 模型注册表：`checkpoints/registry.json`
- 空的数据、训练、归档和审计目录：`storage/`
- 启动脚本：`run.ps1`
- Python 依赖：`requirements.txt`
- 中文说明：`README.md`、`RELEASE_CN.md`

默认发布包不包含当前开发电脑里的业务数据集、预测缓存、训练日志、导出结果和训练 run 模型，避免误交付测试数据。

## 2. 环境要求

推荐环境：

- Windows 10/11
- Python 3.10 或 3.11
- NVIDIA 显卡和 CUDA 版 PyTorch，用于训练和推理加速

如果用户只做流程体验，也可以使用 CPU 环境，但模型预测和训练会明显变慢。

## 3. 安装和启动

解压 zip 后，在解压目录打开 PowerShell，执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 7860
```

启动成功后，浏览器打开：

```text
http://127.0.0.1:7860/
```

后续再次启动时，可以在项目根目录执行：

```powershell
.\run.ps1
```

## 4. 推荐工作流

### 数据集管理

1. 进入首页。
2. 打开 `数据集管理平台`。
3. 填写项目名称、缺陷类别、标签颜色和数据集阶段。
4. 选择本地图像文件夹。
5. 点击创建并导入。

平台会在 `storage/datasets/{dataset_id}/` 下构建图像、预测、人工 Review mask 和元数据目录。没有人工 Review 前不会生成监督训练用 mask。

### 模型生成

适用于新缺陷类别的小样本冷启动。

1. 进入 `模型生成工作台`。
2. 选择或创建 generation 数据集。
3. 第一轮生成主动学习队列。
4. 人工 Review 队列中的图像并保存 mask。
5. 创建训练任务。
6. 训练完成后，平台会注册新的 checkpoint，并用新模型预测未 Review 图像。
7. 继续下一轮主动学习，直到多轮 Review 完成并得到可用基线模型。

### 模型优化

适用于已有模型接入更大数据集后的微调升级。

1. 进入 `模型优化工作台`。
2. 选择已完成的基线模型或训练模型。
3. 导入 optimization 数据集。
4. 对全数据集执行模型预测。
5. 生成主动学习队列。
6. 人工 Review 高价值样本。
7. 创建训练任务进行微调。

模型优化数据集和模型生成数据集分开维护，但同一缺陷类别会在模型生态中统一管理。

## 5. 训练监控和模型可用规则

训练页会实时显示：

- 当前状态
- 训练进度
- batch 进度
- 吞吐
- GPU 显存
- 实时曲线
- checkpoint 是否可用

训练中的 `best.pt` 不会提前出现在模型选择列表。只有训练任务状态为 `completed` 后，生成的 checkpoint 才能用于主动学习、模型优化或后续训练。

## 6. 数据目录说明

```text
storage/
  datasets/          数据集、图像副本、预测和人工 Review mask
  categories/        缺陷类别目录和类别元数据
  dataset_archives/  归档数据集
  training_jobs/     训练任务状态和指标
  model_tests/       模型测试输出
  exports/           导出结果
  audit_logs/        审计事件

checkpoints/
  baseline/          初始模型权重
  runs/              用户训练生成的新模型
```

删除平台数据集只删除平台内的导入副本和衍生文件，不会删除用户原始图片文件夹。

## 7. 常见问题

### 端口被占用

如果 7860 被占用，可以换端口启动：

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 7861
```

然后打开：

```text
http://127.0.0.1:7861/
```

### 安装 PyTorch 很慢或 CUDA 不可用

建议先在用户电脑配置好匹配显卡驱动的 PyTorch CUDA 环境，再执行：

```powershell
pip install -r requirements.txt
```

### 训练速度慢

请先确认：

- 已安装 CUDA 版 PyTorch
- 浏览器训练页显示设备为 CUDA
- 图像尺寸和 batch size 符合显存容量
- 没有其他程序占用显卡

## 8. 停止服务

在启动服务的 PowerShell 窗口按：

```text
Ctrl + C
```

如果窗口已经关闭，可以在任务管理器中结束对应的 Python 进程。

## 9. 发布版本重点

- 首页模型中心和数据集治理闭环。
- 数据集管理平台独立入口。
- 项目名称、缺陷类别、标签颜色和导入流程统一管理。
- 模型生成和模型优化数据集隔离。
- 关键操作审计记录。
- 实时训练监控。
- 训练未完成时禁止提前调用 checkpoint。
- GitHub 仓库默认清理测试数据，仅保留可启动的空目录结构和初始模型。
