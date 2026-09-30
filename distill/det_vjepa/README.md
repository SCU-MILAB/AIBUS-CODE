# ViT-G (V-JEPA) 全参数检测训练

把 `ckpts/vjepa_full/best_vjepa_model9639(paper).pt` 里的 ViT-G **encoder 直接当检测
backbone**，接 ViTDet 风格 simple feature pyramid 和 FCOS 检测头，**全参数端到端训练**
（不冻结任何层，不使用 teacher，不做蒸馏）。

新增目录，**没有改动 `distill/det/` 下任何现有文件**。

---

## 1. 网络结构

### Backbone：ViT-G（可直接吃 640×640）

| 项 | 值 |
|---|---|
| 结构 | `vit_giant_xformers` |
| patch / tubelet | 16 / 2 |
| embed_dim / depth / heads | 1408 / 40 / 22 |
| 位置编码 | RoPE（`use_rope=True`） |
| 参数量 | 1012.2 M |

两个关键点：

1. **tubelet_size = 2，T 必须 ≥ 2。** patch embedding 是 kernel `(2,16,16)` 的 Conv3d，
   所以每张 2D 图被复制成 **2 帧伪视频**再送进 encoder。
2. **RoPE 在 forward 时按传入的 `T/H/W` 动态生成**（`src/models/utils/modules.py`
   的 `RoPEAttention`），所以 640×640 可以直接跑，**不需要插值 pos-embed**，也不用改
   checkpoint。默认取最后一个 block 的 `[B, 1600, 1408]`，reshape 成
   `[B, 1408, 40, 40]`，即 stride 16 的单尺度特征图。

### Neck：ViTDet simple feature pyramid

ViT 只有 stride 16 一个尺度，检测要多尺度，所以从这一张图造出金字塔：

- **stride 8**：`ConvTranspose2d(stride=2)` 从 stride 16 上来
- **stride 16**：1×1 conv（ViT 特征网格与图像**不是**半格偏移，所以用 1×1 而非 3×3 ——
  这是 ViTDet 论文 Sec. 3.2 的结论）
- **stride 32**：stride-2 conv 下去
- 之后用标准 top-down FPN 融合，输出 256 通道

可选 `--out_layers 13 26 39`，依次把 ViT 第 13、26、39 号 block 的输出送入
stride 8、16、32 分支（编号从 0 开始）。这些 block 的输出**原生都是 stride 16**；
stride 8/32 仍由各分支的上/下采样生成。`--out_layers` 不传时，三个分支继续共用
最后一个 block 的输出，保持原始 SimpleFPN 行为。所选 block 数量须与
`--fpn_strides` 数量一致，严格递增，最后一个必须是 39。

### Head：FCOS（anchor-free）

- 分类：sigmoid focal loss（每个位置都算）
- 回归：预测到框四条边的 ltrb 距离，GIoU loss（只在正样本上算）
- centerness：BCE，推理时 `score = cls_sigmoid × centerness_sigmoid`
- 正样本分配：框内 + center sampling（半径 `1.5×stride`）+ 该层的回归范围
  `(8·s_{i-1}, 8·s_i]`；多个框重叠时**面积小的胜出**

**总参数量 1023.7 M**（backbone 1012.2 M + neck/head 11.5 M），全部可训练。

---

## 2. 怎么跑

环境用仓库现有的即可（torch / torchvision / ultralytics / opencv / pyyaml），无需额外安装。

### 单卡

```bash
cd /home/lx/baselines/vjepa
python -m distill.det_vjepa.train \
    --data /home/lx/dataset/6cls_detection/data.yaml \
    --imgsz 640 --batch 8 --epochs 20 \
    --output_dir distill/det_vjepa/output --exp_name vjepa_giant_fcos
```

### 4 卡 DDP

```bash
cd /home/lx/baselines/vjepa
torchrun --master_port=12457 --nproc_per_node=4 -m distill.det_vjepa.train --data /home/lx/dataset/6cls_detection/data.yaml --imgsz 640 --batch 8 --epochs 20 --output_dir distill/det_vjepa/output --exp_name vjepa_giant_mae --ckpt ckpts/vitg_mae_e1.pt
torchrun --master_port=12454 --nproc_per_node=4 -m distill.det_vjepa.train --data /home/lx/dataset/6cls_detection/data.yaml --imgsz 640 --batch 8 --epochs 20 --output_dir distill/det_vjepa/output --exp_name vjepa_giant_vjepanopretrain --ckpt /home/wcz/workspace/vjepa2-main/vitg.pt
```

多层特征版本在原命令后加 `--out_layers 13 26 39`，并使用新的 `--exp_name`。
例如：

```bash
torchrun --master_port=12458 --nproc_per_node=4 -m distill.det_vjepa.train \
    --data /home/lx/dataset/6cls_detection/data.yaml \
    --imgsz 640 --batch 8 --epochs 20 \
    --out_layers 13 26 39 --exp_name vjepa_giant_multidepth
```

> `--batch` 是**每张卡**的 batch。上面 4 卡时有效 batch = 8×4 = 32（再加 `--accum`）。
> **多卡必须用 torchrun**；`--device` 只在单进程下生效，传 `"0,1,2,3"` 会直接报错提示你
> 改用 torchrun。

使用 torchrun 时，验证自动在所有 rank 上并行推理，`--val_batch` 是每张卡的
验证 batch。验证集按连续区间分片，不补齐、不重复；`--val_max_images N`
仍表示整个验证集的前 N 张，而不是每卡 N 张。图片不足时允许部分 rank 分片为空。

预测与真实标注通过独立 CPU/Gloo 进程组汇总到 rank 0，按原数据集顺序统一
计算 mAP、precision、recall 和 F1，再广播指标。中途验证、最佳模型的完整
预测导出及最终验证均采用此流程；checkpoint 和 CSV 仍只由 rank 0 写入。
`--rank_zero_timeout` 控制验证汇总及写盘等待的秒数，默认 `86400`（24 小时），
训练通信仍使用原来的 NCCL 超时设置。单卡启动方式不变。

### 输出

```
distill/det_vjepa/output/<exp_name>/
├── config.json        本次运行的全部参数
├── metrics.jsonl      每个 epoch 的训练/验证指标（逐行 JSON）
├── best.pt            val mAP@0.5:0.95 最高的 checkpoint（含 optimizer）
├── best_prediction.csv 最佳 checkpoint 的完整验证集预测，原图像素坐标
├── best_prediction.metrics.json 对应验证指标、checkpoint epoch 与评估配置
└── last_epochN.pt     每 --save_period 个 epoch 的存档，以及最后一个 epoch
```

### 断点续训

```bash
torchrun --nproc_per_node=4 -m distill.det_vjepa.train \
    --data /home/lx/dataset/6cls_detection/data.yaml \
    ... 和上次完全相同的参数 ... \
    --resume distill/det_vjepa/output/vjepa_giant_fcos/last_epoch9.pt
```

checkpoint 里存了 model / optimizer / epoch，但**不恢复命令行参数**，续训时要重新传一遍。
多层特征实验续训时须传相同的 `--out_layers`；旧 checkpoint 不传即可。

### 从最佳权重重新微调

需要调整学习率或训练时长时，用 `--init_weights` 加载完整检测器的模型权重，
并从 epoch 0 新建优化器和 warmup/cosine 学习率计划。输出到新的 `--exp_name`，
保留原实验结果。`--resume` 则用于按原计划恢复 optimizer 和 epoch，两者不能同时使用。
两种特征模式的参数名与形状相同，可用 `--init_weights` 从原始单层检测器权重
初始化多层实验，并传入 `--out_layers`；切换模式不能用 `--resume`。

```bash
python -m distill.det_vjepa.train \
    --data /home/lx/dataset/6cls_detection/data.yaml \
    --init_weights distill/det_vjepa/output/vjepa_giant_fcos_schedfix/best.pt \
    --drop_path 0.1 --layer_decay 0.85 \
    --lr_backbone 2e-6 --lr_head 2e-5 --warmup_epochs 0.5 \
    --epochs 8 --batch 8 --exp_name vjepa_giant_fcos_ft_low_lr
```

---

## 3. 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data` | 必填 | YOLO 格式 data.yaml |
| `--ckpt` | `ckpts/vjepa_full/best_vjepa_model9639(paper).pt` | 取其中的 `encoder` |
| `--imgsz` | 640 | 必须能被 16 整除 |
| `--batch` | 8 | **每张卡**的 batch |
| `--accum` | 1 | 梯度累积步数 |
| `--init_weights` | 空 | 从检测器 checkpoint 只加载模型权重，重新开始微调 |
| `--lr_backbone` | 1e-5 | 主干最深层的学习率 |
| `--lr_head` | 1e-4 | neck + head 的学习率 |
| `--layer_decay` | 0.75 | 主干逐层衰减，第 i 层 = `lr_backbone × 0.75^(39-i)` |
| `--weight_decay` | 0.05 | 只作用于矩阵权重，norm/bias 不衰减 |
| `--warmup_epochs` | 1.0 | 线性 warmup，之后 cosine 到 `1%` |
| `--clip_grad` | 1.0 | 梯度裁剪 |
| `--reg_weight` | 1.0 | GIoU 回归损失权重；用于单独测试定位目标的权重 |
| `--grad_checkpoint` | 开 | ViT block 内部激活重算，1B 全量微调基本必须开 |
| `--amp` | 开 | bf16 autocast（损失在 fp32 算） |
| `--fpn_strides` | `8 16 32` | 也可传 `4 8 16 32` |
| `--out_layers` | 不传 | 每个 FPN 分支对应的 ViT block 编号，如 `13 26 39`；不传则共用最后一层 |
| `--val_every` | 1 | 0 表示训练中途不评测 |
| `--val_max_images` | 0 | 中途评测只跑前 N 张（0=全部），加速用 |
| `--val_batch` | 8 | 每张卡的验证 batch |
| `--rank_zero_timeout` | 86400 | 验证结果汇总、指标广播和 checkpoint 写盘等待的超时秒数 |
| `--conf_thres` / `--iou_thres` | 0.25 / 0.7 | 评测阈值，和 paper.md 一致 |
| `--save_period` | 5 | 存档间隔；每个 ckpt 约 4 GB |

---

## 4. 实测开销（640×640，RTX PRO 6000 Blackwell 96GB，`--grad_checkpoint --amp`）

以下都是**完整训练步**（forward + backward + AdamW step + 梯度裁剪）的实测值，
每项跳过第 0 个 epoch 的 warmup：

| 配置 | 每步 | 每图 | 峰值显存/卡 |
|---|---|---|---|
| 单卡 batch 8 | 0.98 s | 0.123 s | 18.5 GiB |
| 4 卡 DDP batch 8/卡（全局 32） | 0.94 s | 0.029 s | ~19 GiB |
| 4 卡 DDP batch 16/卡（全局 64） | 2.25 s\* | 0.035 s\* | **25.4 GiB** |

\* 该项只跑了 1 个 epoch，含 warmup，所以每图耗时偏高；瓶颈是显存而不是速度。
4 卡相对单卡基本是**线性加速**（每卡每图 0.117 s vs 0.123 s），1B 参数 allreduce 的开销
被计算掩盖掉了。

按 `116059` 张训练图换算：

| 配置 | 每 epoch | 20 epoch |
|---|---|---|
| 4 卡 DDP batch 8/卡 | **约 56 分钟** | **约 19 小时** |
| 单卡 batch 8 | 约 4.0 小时 | 约 80 小时 |

显存余量很大（batch 16/卡 只占 25.4 GiB），`--batch 32` 或 `--imgsz 1024` 都还有空间，
可以按需加。`--accum` 在 4 卡 batch 8 时已经等于全局 32，一般不用再加。

推理对比（说明为什么这不是可部署模型）：

推理对比（说明为什么这不是可部署模型）：

| 模型 | 参数量 | 640 推理 |
|---|---|---|
| YOLO11m | 20.1 M | **6.3 ms/图** |
| 本模型（仅 ViT-G encoder，不含 neck/head） | 1012 M | **136 ms/图** |

---

## 5. 评测协议

和 `distill/det/output/paper.md` 第 5 节完全对齐，数值可直接和表 4 / 表 5 对比：

- conf threshold **0.25**，NMS IoU **0.7**，每类分别做 NMS
- AP 用 **all-point 插值**（和 ultralytics 的统计口径一致）
- `mAP@0.5:0.95` = IoU 阈值 `0.50:0.05:0.95` 共 10 个点的平均
- 预测框和 GT 都在 **letterbox 之后的像素坐标系**里算 IoU，所以 padding 会自动抵消，
  不需要反变换回原图

---

## 6. 注意事项

1. **这不是可部署模型，建议定位为「V-JEPA 表征在检测上的上界 / 对照实验」。**
   1023.7 M 参数，640 推理约 136 ms/图，比 YOLO11m 慢 20 倍以上，和 `paper.md`
   的轻量化立论直接冲突。它的价值是回答「把 V-JEPA 特征直接拿来做检测能到什么水平」，
   而不是去和 YOLO11m 争部署指标。

2. **增强没有对齐。** 本脚本的增强只有 **letterbox + 随机水平翻转**，没有 mosaic /
   scale jitter / HSV 抖动。`paper.md` 里的 YOLO11m 基线用的是 ultralytics 默认增强。
   所以这两组数**不是增强对齐的公平对比**，写论文时要说明。

3. **学习率没有调过，而且 `--layer_decay 0.75` 对 40 层的模型可能过小。**
   `lr_backbone=1e-5 + layer_decay=0.75` 是 ViTDet（depth 24）/ BEiT 的常规起点，但这里是
   **depth 40**，第 0 个 block 的 lr = `1e-5 × 0.75³⁹ ≈ 1.5e-10`，几乎冻住了浅层。这是
   ViTDet 的原始设定（浅层本来就该少动），但如果你希望主干整体动得更多，把
   `--layer_decay` 提到 **0.85~0.9** 更合理。另外 1012 M 参数在 116k 图上全量微调
   **过拟合风险很高**，如果 val 很早就饱和，优先试：加大 `--weight_decay`、
   加 `--drop_path 0.1~0.2`、把 `--epochs` 降到 5~10。

4. **显存。** 实测 batch 8/卡 峰值 18.5 GiB、batch 16/卡 25.4 GiB（96 GB 卡上很宽松）。
   其中约 16 GB 是与 batch 无关的固定开销（1B 参数的 fp32 权重 / 梯度 / AdamW 两个动量
   各约 4 GB），其余才是激活。`--grad_checkpoint` 关掉后激活会涨很多，建议始终开着。
   另注意每个 checkpoint 约 **4 GB**，`--save_period` 太密会很快吃满磁盘。

5. **尺度对齐细节。** 回归头输出的距离以 **level stride 为单位**，`decode()` 和
   `FCOSLoss` 各乘一次 stride。这两处必须一致，否则框会整体大/小一个 stride 的倍数
   ——训练 loss 看着很正常，但推理 mAP 恒为 0。

---

## 7. 文件说明

| 文件 | 内容 |
|---|---|
| `model.py` | ViT-G backbone 加载、simple feature pyramid、FCOS head、`decode()` |
| `dataset.py` | YOLO 格式数据集、letterbox、水平翻转、ImageNet 归一化 |
| `loss.py` | FCOS 正负样本分配、focal / GIoU / centerness loss |
| `metric.py` | NMS 后处理、all-point AP、mAP@0.5、mAP@0.5:0.95、P/R/F1 |
| `train.py` | DDP 训练循环、layer-wise lr、warmup+cosine、评测、存档 |

---

## 8. 验证状态

**已经跑通的（不是只写了代码）：**

- **过拟合验证**：真实数据里抽 4 张图当训练集、4 张当验证集，60 epoch，默认超参。
  结果 **val mAP@0.5 = 1.0000、mAP@0.5:0.95 = 0.8633、P = R = F1 = 1.0**。这说明整条
  链路（数据 → backbone → neck → head → loss → decode → NMS → AP）是通的，
  特别是**回归框的尺度是对的**——如果 loss 和 decode 的 stride 不一致，这个数会恒为 0。
- **4 卡 DDP**：`torchrun --nproc_per_node=4` 正常收敛、无死锁，上面第 4 节的吞吐就是
  在真实数据子集（1024 张）上实测的。
- **全参数梯度**：1023.7 M 参数全部 `requires_grad=True`，训练后逐个检查过
  **没有参数缺梯度**（`find_unused_parameters=False` 能开起来就是证据）。

**没有验证的（用之前请自己确认）：**

- **没在 116k 全量数据上完整训过。** 上面所有数字都是子集/短跑，最终精度未知，
  超参（lr、weight_decay、epochs）都还没调。第 6 节第 3 条是调参建议的起点。
- **没有和 `paper.md` 的 YOLO11m 数字跑过同一份完整评测**，所以现在还不能下
  「V-JEPA 主干比 YOLO11m 好/差」的结论。
- 多尺度 `--fpn_strides 4 8 16 32`、`--imgsz` 非 640、`--accum > 1` 这些分支逻辑
  写好了但没单独跑过。

## 验证坐标与已完成实验重新预测

验证使用 `original_xyxy_clip_after_nms_v1`：先在输入坐标中做 NMS，随后按数据集
实际 letterbox 的 ratio/left/top 还原原图坐标，将预测框和 GT 裁剪到原图边界，
去掉裁剪后面积为零的预测框，再计算 AP。训练仅由 rank 0 评估。

每次产生新的 `best.pt`，同时保存同一权重对应的 `best_prediction.csv`。
CSV 列为 `image_path,image_width,image_height,class_id,class_name,confidence,x1,y1,x2,y2`；
每个检测框一行，无检测的图像保留一行空检测字段。坐标是原图像素，置信度为 0–1。
即使 `--val_max_images` 使用子集选 best，CSV 仍导出该权重的完整验证集。

已完成实验可独立重新预测；不依赖原始 encoder checkpoint 路径，不修改 best.pt 或旧 metrics.jsonl：

```bash
cd /home/lx/baselines/vjepa
python -m distill.det_vjepa.infer \
    --checkpoints distill/det_vjepa/output/vjepa_giant_fcos_schedfix/best.pt \
    --device 0 --batch 8 --workers 4
```

`--checkpoints` 可接多个路径，顺序加载到同一张卡。默认评估完整验证集；
`--max_images N` 仅作小批量检查，写入 `smoke_prediction.*`，不会替换完整结果。
独立预测的 `best_prediction.metrics.json` 额外包含 `legacy_unclipped`，
它与修正后的指标来自同一次 forward/NMS，用于量化边界裁剪的影响。
旧 best 是按旧验证协议选择的 checkpoint；重新预测不代表重新搜索所有历史 epoch。

两组独立的四卡短微调脚本（均从原 schedfix/best.pt 开始，完整验证、自动保存 best CSV）：

```bash
bash distill/det_vjepa/scripts/run_schedfix_768.sh
bash distill/det_vjepa/scripts/run_schedfix_reg2.sh
```

768 组为每卡 batch=4、accum=2；640/reg2 组为每卡 batch=8、accum=1。
两组有效 batch 都为 32，端口分别为 29521/29522；共享同四张卡时建议逐个运行。

### 使用不同输入尺寸重新预测

```bash
python -m distill.det_vjepa.infer \
    --checkpoints distill/det_vjepa/output/vjepa_giant_fcos_schedfix/best.pt \
    --device 0 --batch 8 --imgsz 256 --output_subdir infer_256
```

`--imgsz` 覆盖数据预处理、主干网格和 anchor 分辨率，加载同一检测器权重。
指定 `--imgsz` 时默认写入 `infer_<尺寸>/`，也可用 `--output_subdir` 自定目录。
独立子目录中的结果若已存在则拒绝覆盖。指标同时记录实际推理 `imgsz` 与训练
`checkpoint_imgsz`；原图坐标还原和边界裁剪规则保持一致。

## 可选病灶单类训练与背景补全

默认仍是常规两类训练/推理，现有命令兼容。仅显式设置 `--lesion_only`
时检测头训练一个 class 0 通道。数据 yaml 仍保留两个语义类别，不修改原标签文件：
训练过滤 class 1 正框，保留所有正常图作为空目标负样本；验证保留病灶 GT。
缺失标注文件会报错，已有空标注文件则可作为正常负样本。

从旧两类检测器 `--init_weights` 初始化时，严格加载所有权重，唯独
`head.cls_logits.weight/bias` 提取第 0 通道。优化器重新创建。
`--resume` 必须与 checkpoint 的单类/两类模式一致。

推理规则（新 checkpoint 自动识别模式，旧两类 checkpoint 也可显式 `--lesion_only`）：

- 存在经过阈值/NMS/原图裁剪的 class 0 检测：仅输出病灶框。
- 没有 class 0 检测：输出一个 class 1 框。框取原图较大非黑连通区域的联合外接矩形，
  强度阈值由 `--black_threshold` 指定（默认 3）。保留面积至少 64 原图像素且宽度
  至少为原图宽度 5%（最低 4 像素）的区域，并始终保留最大区域；过滤小噪声和
  细窄的孤立 ECG/刻度，避免只取最大区域时截掉断开的组织。该规则仍是启发式方法。
  全黑图退回整图框。它是前景区域的矩形外接框，不是逐像素分割。
- class 1 置信度 = `1 - max(raw class_0 sigmoid * centerness)`，最大值在阈值过滤前计算，
  避免将所有背景置信度设成 1。class 1 不参与病灶置信度阈值的第二次过滤。

两类评估采用显式业务协议 `lesion_only_original_gt_complement_background_perfect_match_v2`：
GT 始终使用数据集真实标注的类别和框，不删除标注、不根据模型输出或黑色阈值生成 GT。
仅当没有检出 class 0 时，使用 `black_threshold` 的前景框作为 class 1 预测。
真正背景上的 class 1 框按 IoU=1 匹配，在全部 IoU 阈值上视为定位正确。
漏检病灶仍计 class 0 FN，产生的错误背景框计 class 1 FP；正常图误检病灶则计
class 0 FP 和 class 1 FN。class 1 的总体 AP/Precision 不会在这些错误存在时强行设成 1。
报告同时保存 `lesion_AP50`、`lesion_AP50_95` 和每类 TP/FP/FN。
业务两类 mAP 与常规两类框检测 mAP 的匹配口径不同，请保留协议标识一起比较。

四卡训练脚本，两组均从原 schedfix/best.pt 初始化，不自动启动训练：

```bash
bash distill/det_vjepa/scripts/run_schedfix_lesion_only_640.sh
bash distill/det_vjepa/scripts/run_schedfix_lesion_only_256.sh
```

640 组为 8 epochs，主干/头 LR 为 2e-6/2e-5；256 组为 12 epochs，LR 为 3e-6/5e-5。
两组每卡 batch 8，有效 batch 32，端口分别为 29531/29532。
若最终固定 256 推理，256 组用于适应分辨率变化，实际提升需要训练后验证。

新单类 checkpoint 的推理会自动生成背景：

```bash
python -m distill.det_vjepa.infer \
    --checkpoints distill/det_vjepa/output/schedfix_lesion_only_640_4gpu/best.pt \
    --device 0 --output_subdir infer_lesion_only
```

旧两类 checkpoint 在显式 `--lesion_only` 下会提取病灶通道，默认保存到新的
`infer_lesion_only/`（若同时指定 `--imgsz 256` 则为 `infer_256_lesion_only/`），
保留原两类结果。没有该开关时仍为常规两类推理。

### 重新生成背景前景标注

`/home/lx/dataset/6cls_detection/data-2cls-remap.yaml` 使用独立数据入口
`data_2cls_remap/`，其中 `images` 链接到原图，`labels` 链接到
`labels_2cls_remap`。既有 `labels` 和 `data.yaml` 保留。class 0 行保持原样；
原有 class 1 行替换为阈值 3 的最新前景联合外接框，空标注保持为空。
两类训练和 `--lesion_only` 均可使用此 YAML。生成记录位于
`labels_2cls_remap/generation_summary.json`。

可复现生成脚本（目标存在时拒绝覆盖）：

```bash
python distill/det_vjepa/scripts/generate_2cls_remap.py \
  --root /home/lx/dataset/6cls_detection --black_threshold 3 --workers 8
```

`run_schedfix_lesion_only_640.sh` 和 `run_schedfix_lesion_only_256.sh` 已改用新 YAML；
实验输出名包含 `remap`，避免与此前输出混淆。

### 病灶回归权重配对实验（6 epochs）

在四张卡空闲后运行：

```bash
bash distill/det_vjepa/scripts/run_schedfix_lesion_reg_ablation.sh
# 只跑回归加权组：
bash distill/det_vjepa/scripts/run_schedfix_lesion_reg_ablation.sh reg2
```

默认按顺序跑 reg1、reg2；两组均从原 `vjepa_giant_fcos_schedfix/best.pt`
初始化，使用 remap 数据、lesion-only、640 分辨率、有效 batch 32、
主干/头 LR 2e-6/2e-5、6 epochs、seed 0。唯一实验变量为
`reg_weight=1/2`。6 轮 cosine 与旧 20 轮实验的调度不同，因此以本次
reg1 为配对对照，旧结果只作参考。单个 seed 的差异不代表统计显著性。

脚本显式使用 `--selection_metric lesion_AP50_95` 选择 best，训练终端同时
打印病灶 AP50/AP50–95；常规训练默认仍按总 mAP50–95 选择。
输出目录为 `output/schedfix_lesion_640_remap_reg{1,2}_6ep_<tag>/`，包含
`train.log`、`config.json`、`metrics.jsonl`、best checkpoint 及其预测 CSV。
默认 tag 为 UTC 时间加进程号，可通过 `RUN_TAG` 指定；已有目录会拒绝覆盖。
可用 `CUDA_VISIBLE_DEVICES` 指定四张卡，`MASTER_PORT` 指定端口（默认 29541）。
脚本不会停止正在运行的任务，也不会自动等待 GPU 空闲。

### 640 / 768 病灶分辨率对照实验

四张 GPU 可用且磁盘至少有 64 GiB 可用时运行：

```bash
bash distill/det_vjepa/scripts/run_lesion_resolution_ablation.sh
```

脚本并行运行 640 和 768，各使用两张独立 GPU、训练 6 epochs。两组从同一原始
schedfix best 初始化，使用同一 remap 数据和 lesion-only 协议；每卡 batch 4、
梯度累积 4，两卡有效 batch 32。学习率、cosine 调度、回归权重 1、随机种子和其余训练设置
相同。best checkpoint 按病灶 AP50–95 保存；与旧的 20-epoch 640 实验相比，
这次必须以脚本中新的 640 组作为配对对照。

输出在 `output/lesion_resolution_640_vs_768_<tag>/640/` 和 `768/`。
两组完成后自动生成 `comparison.json`，包含病灶 AP50/AP50–95/AP90/AP95、
每个 GT 是否存在 IoU≥0.9 预测框的覆盖率，以及按 **640 等效像素**短边
分组的覆盖率。同一套尺寸分组用于两种分辨率，便于直接比较。
`RUN_TAG` 可设置结果目录后缀，`GPUS_640`/`GPUS_768` 可指定两组显卡
（默认 `0,1`/`2,3`），`MASTER_PORT` 可指定第一组端口（默认 29551，
第二组使用端口 +1）。脚本启动前检查 CUDA 和磁盘，拒绝覆盖已有目录；
两组都完成后才生成比较报告。
