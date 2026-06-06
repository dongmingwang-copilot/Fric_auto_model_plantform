# 金属表面磨损主动学习工作流

## 目标

平台核心目标是围绕 `模型生成` 和 `模型优化` 建立可追溯的主动学习闭环：

- `模型生成`：独立创建新缺陷项目，从 `UNet-32 Scratch` 和小样本人工标注开始，经过多轮主动学习训练出新的类别 PT。
- `模型优化`：加载已有 PT 作为 baseline，对大数据集批量预测、人工 Review、微调精进。业务 baseline 由用户在本地维护，不随 GitHub 仓库提交。

1. 导入图像。
2. 点击左侧全量模型预测，对当前数据集全量生成 prediction。
3. 设置本轮 Review 数，自动推荐最值得 Review 的样本。
4. 人工逐张修正 mask。
5. 保存 Review 后自动进入下一张推荐图。
6. 只用已 Review 的 mask 训练。
7. 生成新模型，自动切换到最新 checkpoint。
8. 用最新模型刷新剩余未 Review 图像。
9. 自动生成下一轮主动学习队列。

## 数据对象

- Dataset：一个主动学习项目的数据容器，带有 `project_type`。
- `project_type=optimization`：模型优化数据集。旧数据集默认归入此类。
- `project_type=generation`：模型生成项目。只在模型生成工作台显示。
- Item：一张图像及其 metadata、prediction、annotation、active-learning score。
- Label：缺陷类别。一个 label 对应一个颜色和一个二值 mask 训练目标。
- Prediction：某个 checkpoint 对 item 的预测结果。
- Annotation：人工 Review 后的监督信号。
- Active Learning Batch：一次推荐队列。
- Training Job：一次基于已 Review 数据的训练。
- Checkpoint：baseline 或训练产生的新模型。

## 状态规则

- `imported`：图像已导入，但未预测、未 Review。
- `predicted`：至少有一个 checkpoint 对图像生成预测。
- `reviewed`：至少有一个 label 的人工 mask 已保存。

训练只读取 `reviewed` 样本，不会把模型 prediction 自动当作 GT。

## 主动学习队列

队列生成步骤：

1. 对全数据集使用当前 checkpoint 生成 probability map。
2. 对未 Review 图像读取当前 checkpoint 的 probability map。
3. 计算不确定性、entropy、预测面积、连通域数量、多样性。
4. 按综合 score 降序排序。
5. 写入 `active_learning_batches`。
6. 前端按 batch 顺序引导 Review。

模型生成第一轮是特殊阶段：

1. 用户可以一次导入较大图片集合，例如 100 张。
2. 点击生成 Active Set 时，不调用 scratch 模型预测。
3. 平台只生成 `initial_blank_review_v1` 空白 Review 队列，数量由 `本轮 Review 数` 控制，例如 20 张。
4. 只有队列中的图允许保存人工标注；队列外图像不能参与第一轮 Review。
5. 第一轮训练完成后，最新模型会对全数据集写入 prediction。
6. 第二轮开始再基于 prediction 进行主动学习排序。

`预测范围` 固定为全数据集；`本轮 Review 数` 只控制这轮优先进入队列的样本数量，不限制预测数量。全量预测后需要重新生成队列，因为旧队列的 score 已经不再严格对应最新 probability map。

顶部 `模型预测` 是 Review 辅助工具，支持 `单图预测` 和 `全量预测`。画错后可以清空当前掩码，再用 `单图预测` 恢复当前图像的模型 mask，不需要重新跑整批数据。

已 Review 图像受保护：界面中的单图预测不会覆盖或替换人工 mask；全量预测只刷新未 Review 图像。后端 prediction 文件与 annotation 文件分目录存放，prediction 不是 GT。

综合 score 当前使用：

- uncertainty ratio
- entropy mean
- margin uncertainty
- prediction area
- connected components
- histogram diversity against reviewed samples

## Review 规则

保存 Review 时：

1. 当前 canvas mask 保存到 `annotations/{label_id}/{item_id}_review_mask.png`。
2. item 状态更新为 `reviewed`。
3. 当前 active-learning batch 的 item 标记为 `reviewed`。
4. 前端自动跳转到下一张 pending 推荐图。

同一张图、同一个 label 再次保存 Review 时，会覆盖同一路径下的旧 mask。训练读取的是这个最新 mask，不会读取旧版本。

## 数据隔离规则

- 模型生成工作台只读取 `project_type=generation` 的数据集。
- 模型优化工作台只读取 `project_type=optimization` 且 label 与当前模型一致的数据集。
- 旧 Spall 数据集没有显式 `project_type` 时，按 `optimization` 处理。
- 模型生成项目的 dataset id 带 `gen-` 前缀，模型优化项目带 `opt-` 前缀。
- 二者仍共享统一的 checkpoint registry，因此模型生成训练出的 `run-{job_id}-best` 可以进入模型优化工作台作为 baseline 使用。

## 模型上下文规则

- Checkpoint 必须带有 `label_id` 和 `label_name`。
- 在模型优化工作台，模型按缺陷类别分组显示。
- 模型生成产物的 `model_stage=generated_baseline`，在模型优化工作台显示为 `基线模型`。
- Spall 的后续微调产物为 `model_stage=optimization_run`，在模型优化工作台显示为 `训练模型`。
- 切换到新类别基线模型时，当前类别的数据集、选中图像和主动学习队列会保存在前端 buffer 中，左侧任务栏清空并只显示该类别的数据集。
- 切回原类别模型时，自动恢复该类别的 dataset、队列和任务上下文。

## 训练规则

创建训练任务时：

1. 读取当前 dataset。
2. 读取当前 label。
3. 绑定当前 active-learning batch。
4. 自动创建 dataset snapshot。
5. 训练范围为当前数据集内全部已保存 annotation 的图像。
6. 从当前 checkpoint 继续训练。
7. 输出到 `checkpoints/runs/{job_id}/best.pt`。
8. 生成 metrics 和测试三联图。

训练任务会记录 `project_type`、`dataset_id`、`label_id` 和 label 元数据；动态 checkpoint 也会带上这些信息，用于前端筛选和后续审计。

最佳模型选择规则：

`score = 0.45 * F2 + 0.35 * Dice + 0.20 * Recall`

这样会同时奖励高召回与较高 Dice，避免只追 Recall 导致多检失控，也避免只追 Dice 导致漏检变多。

`active_batch_id` 用于追溯“这一轮训练前优先 Review 的队列”，不是训练过滤器。训练不会只吃主动学习队列里的样本，也不会使用未 Review 图像；训练范围是当前数据集中所有已 Review 的人工 mask。这样能让新 Review 样本进入训练，同时保留前几轮人工监督，避免模型遗忘。

训练完成后，平台会：

1. 注册并切换到 `checkpoints/runs/{job_id}/best.pt`。
2. 仅对未 Review 图像重新预测。
3. 清空旧队列语义，按最新模型 probability map 重新排序。
4. 生成下一轮主动学习队列。

上述动作由后端训练服务在训练完成后执行，不依赖浏览器页面是否打开。

## 模型测试

模型测试模块用于客观检查当前 checkpoint：

1. 选择数据集和 checkpoint。
2. 从已 Review 图像中随机抽样，默认 20 张。
3. GT 读取每张图最新保存的人工 Review mask。
4. 模型只做临时推理，不写入 prediction metadata，不改变主动学习队列。
5. 每张图输出横向三联图：原图、人工 GT、差异图。

差异图颜色：

- 红色：漏检 FN
- 绿色：多检 FP
- 蓝色：命中 TP

当没有未 Review 图像时，主动学习队列不会继续创建。此时应进行最终训练、导出当前数据集，或导入新一批图像后进入下一轮。

## 版本和导出

- `创建数据版本` 用于固化当前 dataset 状态。
- `导出主动学习包` 保留完整平台数据结构。
- `导出 COCO` 用于外部训练或 Datumaro 转换。
- `生成 FiftyOne` 用于独立环境下浏览、筛选和质检。

## 归档和删除

- `归档数据集` 会在 `storage/dataset_archives/{archive_id}` 下集中保存平台数据副本，包括平台内原图副本、预览图、人工 GT/mask、prediction metadata 和 dataset.json。
- 归档不会删除或改动用户导入时选择的外部原始图片路径。
- `加载归档` 会从归档副本恢复为新的平台数据集。
- `删除数据集` 是永久操作，会删除平台内该 dataset 的导入副本、prediction、mask 和 metadata；不会删除外部原始图片文件夹。

## 严格边界

- Prediction 不是 GT。
- Annotation 才是监督信号。
- Baseline checkpoint 永远只读。
- 训练 run checkpoint 与 baseline 分目录存放。
- 每次主动学习推荐都必须绑定 checkpoint、label 和 threshold。
