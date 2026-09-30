# autoresearch

这是一个让大语言模型（LLM）自主开展研究的实验。

## 设置

要启动一轮新的实验，请与用户一起完成以下步骤：

1. **确定本轮运行的标签**：根据今天的日期提出一个标签（例如 `mar5`）。分支 `autoresearch/<tag>` 必须尚不存在，因为这是一次全新的运行。
2. **创建分支**：从当前的 master 分支执行 `git checkout -b autoresearch/<tag>`。
3. **阅读本次工作涉及的文件**：先读 distill/det_vjepa/README.md、model.py、train.py、evaluation.py，以及当前最优实验的 config.json、best_prediction.metrics.json、metrics.jsonl。检查 Git 状态和正在运行的训练；不要覆盖或中断现有任务。当前 distill/det_vjepa/ 尚未被 Git 跟踪，不要直接按其他项目的做法切分支并重置整个工作树。
4. 固定数据为 /home/lx/dataset/6cls_detection/data-2cls-remap.yaml，固定 --lesion_only --black_threshold 3。沿用现有数据划分、标注、evaluation.py、metric.py、后处理和 AP 计算，固定 --conf_thres 0.25 --iou_thres 0.7。
5. 模型初始化是当前最优检测器：--init_weights distill/det_vjepa/output/lesion_resolution_640_vs_768_20260928T083805_1134/768/best.pt。后续实验出现更好权重的时候可以更换为新的best.
6. 保持现有 ViT-G backbone 的结构和权重加载方式：不换 backbone，不改 ViT block、patch/tubelet、RoPE、encoder 前向或 checkpoint 键。训练时**允许微调 backbone 参数**，也允许调其学习率等训练超参数。优先研究 neck/head、损失、训练超参数和输入分辨率。改 head 时必须保证旧权重可加载，或明确处理新参数的初始化。一次只改一个主要变量。
7. 当前成绩参考：上述 768 best.pt 的 best_prediction.metrics.json 记录完整验证集 5,000 张图，lesion_AP50_95 = 0.8104124888777733，best epoch = 2。
8. 每次使用新的实验目录和 exp_name。启动前检查四张 GPU、数据、初始化文件、磁盘。一个 run 的 best.pt 可占约 12 GiB；明确确定当前的权重优于最佳结果时才保存,否则不保存权重。
9. **初始化 results.tsv**：创建 `results.tsv`，仅写入表头行。基线结果将在首次运行后记录。
10. **确认并开始**：确认设置无误。

获得确认后，开始开展实验。

## 实验

每次实验都在4张 GPU 上运行。训练脚本的**固定时间预算为 120 分钟**（按实际经过的训练时间计算，不包括启动和编译时间）。
要在

从当前最优检测器起步的参考命令：

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
  -m distill.det_vjepa.train \
  --data /home/lx/dataset/6cls_detection/data-2cls-remap.yaml \
  --init_weights distill/det_vjepa/output/lesion_resolution_640_vs_768_20260928T083805_1134/768/best.pt \
  --output_dir distill/det_vjepa/output --exp_name autoresearch_EXPERIMENT_ID \
  --lesion_only --black_threshold 3 --selection_metric lesion_AP50_95 \
  --imgsz 768 --batch 4 --accum 2 --val_batch 4 \
  --epochs 6 --lr_backbone 2e-6 --lr_head 2e-5 \
  --drop_path 0.1 --layer_decay 0.85 --weight_decay 0.05 \
  --warmup_epochs 0.5 --min_lr_ratio 0.01 --clip_grad 1.0 --reg_weight 1.0 \
  --val_every 1 --val_max_images 0 --final_val_max_images 0 \
  --conf_thres 0.25 --iou_thres 0.7 --save_period 0 --workers 4 --seed 0 \
  > distill/det_vjepa/output/autoresearch_EXPERIMENT_ID.log 2>&1
~~~

**你可以做的事：**
- 修改 `train.py、model.py、loss.py`和相关命令 — 模型检测头、优化器、超参数、训练循环、批量大小等，都可以自由调整。

**你不可以做的事：**
- 修改任务相关评估逻辑、数据、训练常量（时间预算、序列长度等） 
- 修改评估框架。`metric.py` 是衡量结果的权威标准。
- 修改 det_vjepa/ 之外的所有文件
- 修改模型的backbone

**目标很简单：让 lesion_AP50_95 尽可能低。** 由于时间预算固定，你无需担心训练耗时 — 每次都是 120 分钟。模型检测头、优化器、超参数、批量大小等都可以调整。唯一的约束是代码运行时不能崩溃，并且必须在时间预算内完成。

**显存（VRAM）**是一个软约束。如果能显著改善 lesion_AP50_95，适当增加显存占用是可以接受的，但不应大幅失控。

**简洁性标准**：其他条件相同时，越简单越好。为了微小的提升而引入难看的复杂逻辑，并不值得。反过来，删除一些东西后得到相同或更好的结果，就是很好的成果 — 这是简化带来的收益。在判断是否保留一项改动时，要权衡复杂性成本与提升幅度。为了让 lesion_AP50_95 改善 0.001，而增加 20 行东拼西凑的代码？大概不值得。删除代码就能让 lesion_AP50_95 改善 0.001？一定要保留。性能提升约为 0，但代码简单得多？保留。

**基线结果**：已有 768 checkpoint 的 0.810412 无需重训来证明它本身的成绩。

## 输出格式

脚本完成后，会输出如下摘要：

```
---
lesion_AP50_95:   0.810412
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mAP50_95:      39.80
等等,你看哪些需要都加上去
```

注意，脚本被配置为始终在 120 分钟后停止，因此具体数值可能会因这台计算机的计算平台而有所不同。你可以从日志文件中提取关键指标：

```
grep "^lesion_AP50_95:" run.log
```

## 记录结果

每次实验完成后，都要将结果记录到 `results.tsv` 中（使用制表符分隔，不要使用逗号分隔 — 描述中出现逗号会破坏格式）。

TSV 文件包含一行表头和 5 列：

```
commit	lesion_AP50_95	memory_gb	status	description
```

1. Git 提交哈希（短格式，7 个字符）
2. 本次得到的 lesion_AP50_95（例如 0.234567）— 如果运行崩溃，填写 0.000000
3. 峰值显存占用，单位为 GB，按 `.1f` 格式保留一位小数（例如 12.3 — 将 peak_vram_mb 除以 1024）— 如果运行崩溃，填写 0.0
4. 状态：`keep`（保留）、`discard`（舍弃）或 `crash`（崩溃）
5. 简短描述本次实验尝试了什么

示例：

```
commit	lesion_AP50_95	memory_gb	status	description
a1b2c3d	0.897900	44.0	keep	baseline
b2c3d4e	0.893200	44.2	keep	increase LR to 0.04
c3d4e5f	0.905000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## 实验循环

实验在一个专用分支上进行（例如 `autoresearch/mar5` 或 `autoresearch/mar5-gpu0`）。

持续循环：

1. 查看 Git 状态：当前所在的分支和提交。
2. 根据一个实验想法，直接修改 `train.py` 中的代码进行调整。
3. 提交 Git 改动。
4. 运行实验：重定向所有输出 — 不要使用 tee，也不要让输出淹没你的上下文。
5. 读取结果：`grep "^lesion_AP50_95:\|^peak_vram_mb:" run.log`。
6. 如果 grep 没有输出，说明本次运行崩溃了。运行 `tail -n 50 run.log`，查看 Python 堆栈跟踪并尝试修复。如果尝试了好几次仍无法正常运行，就放弃这次尝试。
7. 将结果记录到 TSV 文件中。
8. 如果 lesion_AP50_95 有改善（数值更低），就保留这次 Git 提交，让分支向前推进。
9. 如果 lesion_AP50_95 持平或变差，就通过 git reset 回到本次实验开始前的位置。

这里的理念是：你是一个完全自主的研究者，不断尝试各种想法。有效就保留，无效就舍弃。通过不断推进分支，你可以持续迭代。如果你觉得自己在某个方面陷入困境，可以回退，但应当极少使用这种做法（甚至完全不用）。

**超时**：每次实验的总耗时应当约为 120 分钟（另加几秒钟的启动和评估开销）。如果一次运行超过 120 分钟，就终止它，并将其视为失败（舍弃改动并回退）。

**崩溃**：如果运行崩溃（显存不足 OOM、程序错误等），请自行判断：如果只是一个简单且容易修复的问题（例如拼写错误或缺少导入），就修复并重新运行。如果想法本身存在根本性问题，就跳过它，在 TSV 文件中将状态记录为 `crash`，然后继续。

**永不停止**：一旦实验循环开始（即完成初始设置后），不要暂停下来询问用户是否应该继续。不要问“我应该继续吗？”或“现在适合停下来吗？”。用户可能正在睡觉，或已经离开电脑，并期望你一直持续工作，直到被手动叫停。你是自主的。如果想法用尽，就更深入地思考 — 阅读代码中引用的论文，重新阅读本次工作涉及的文件以寻找新角度，尝试组合之前那些差一点就成功的方案，尝试更大胆的架构改动。循环持续进行，直到用户打断你，就这么简单。

举个使用场景：用户可能会在睡觉时让你持续运行。如果每次实验耗时约 120 分钟，你2小时就能运行约1次，在一个人通常的睡眠时间内，总共可以完成约 4 次实验。用户醒来后就能看到实验结果，而这些实验都由你在用户睡觉期间完成！
