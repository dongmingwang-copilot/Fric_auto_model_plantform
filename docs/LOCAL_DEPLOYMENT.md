# 本地交付和测试说明

## 交付包原则

推荐默认交付代码、初始模型和空数据目录，让专业人士在自己电脑上导入测试图像并完成一轮主动学习。

默认交付包包含：

- `app/` 后端服务
- `web/` 中文界面
- `checkpoints/baseline/unet32_scratch.pt` 初始模型权重
- `checkpoints/registry.json` 模型注册表
- `docs/` 工作流和部署说明
- `requirements.txt`
- `run.ps1`

默认不包含：

- 当前已有数据集
- 已生成预测缓存
- 训练 run 模型
- 业务 baseline 模型
- 个人测试导出文件

如果要把当前数据集也一起给对方，可以使用 `-IncludeStorage` 参数。

## 打包

在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\package_release.ps1
```

生成的 zip 位于：

```text
release\MetalWearPlatform_v1_YYYYMMDD-HHMMSS.zip
```

如果需要连当前 `storage/` 一起打包：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\package_release.ps1 -IncludeStorage
```

## 对方电脑部署

解压 zip 后，在项目根目录执行：

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

如果对方已经有 PyTorch CUDA 环境，也可以直接在该环境中安装 `requirements.txt` 并启动。

## 推荐的一轮主动学习测试

1. 新建数据集。
2. 导入待测图片文件夹。
3. 点击 `全量模型预测`，让 baseline 对所有图生成预测。
4. 设置 `本轮 Review 数`，建议第一轮从 20 到 50 张开始。
5. 点击 `生成队列`。
6. 点击 `下一张推荐图`，按队列逐张修正 mask。
7. 点击 `保存 Review`，系统自动推进到下一张推荐图。
8. 完成本轮 Review 后，点击 `创建训练任务`。
9. 点击 `实时检测训练` 查看训练过程和测试三联图。

## 推荐数和预测范围

`预测范围` 应固定为全数据集。主动学习需要先知道模型对所有图的预测状态，才能判断哪些图最值得人工 Review。

`本轮 Review 数` 不等于预测数量，只控制本轮队列长度。它代表这轮准备让专业人士优先看的样本数。

建议：

- 小规模试跑：10 到 20 张，验证流程。
- 正式第一轮：20 到 50 张，优先暴露模型短板。
- 数据差异很大时：50 到 100 张，但要避免一次 Review 太多导致反馈周期变慢。

每一轮的关键不是一次看完全部图片，而是让模型先从最不确定、最容易错、最有代表性的样本中学习。
