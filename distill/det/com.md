# 统一检测蒸馏对比

入口：`distill/det/com.py`。实际基线目录是 `/home/lx/baselines`。
八个图像基座模型分别作为冻结教师，蒸馏到同一个 YOLO11m 学生。
每组进行一次「有监督检测 + 特征蒸馏」联合训练，训练后直接使用学生检测器。

```bash
cd /home/lx/baselines/vjepa

# 预检查路径并打印配置，不加载模型、不训练。
python -m distill.det.com --teachers all vjepa none --dry_run

# 八个基座教师 + V-JEPA 教师 + 无蒸馏对照，按顺序训练。
# 每一组使用四卡；batch 是全局 batch，128 / 4 = 每卡 32。
python -m distill.det.com \
  --teachers all vjepa none \
  --data /home/lx/dataset/6cls_detection/data.yaml \
  --model /home/lx/baselines/vjepa/yolo11m.pt \
  --device 0,1,2,3 --batch 128 --imgsz 256 --epochs 100 \
  --run_name foundation_kd_seed0

# 只训练一组，可单卡。其余设置沿用上面的默认值。
python -m distill.det.com --teachers uscl --device 0 \
  --run_name uscl_single

# 选几组；覆盖单个教师权重时，--teachers 必须只有一个模型。
python -m distill.det.com --teachers usfm ultrasam --device 0,1,2,3 \
  --run_name selected
python -m distill.det.com --teachers uscl \
  --teacher_ckpt /home/lx/baselines/ckpts/USCL.pth \
  --device 0 --run_name custom_uscl

# CPU 冒烟检查：真实预训练教师前向、学生反向、投影层更新、EMA 复制和验证损失。
# 不读取数据集，不产生训练结果；OpenUS 使用相同扫描数学的 PyTorch 参考后端。
python -m distill.det.com --teachers all none --smoke
```

直接 `python distill/det/com.py ...` 也可以。多卡直接传 `--device`，由 Ultralytics
启动 DDP；不要再用 torchrun 包裹入口。每个教师在独立进程中运行，上一组结束后才启动下一组。
遇到失败立即停止；可以用新的 run_name 单独运行剩余教师。已有实验目录拒绝覆盖。

| 教师标识 | 默认权重（相对 /home/lx/baselines） | 教师空间特征通道 |
|---|---|---:|
| uscl | ckpts/USCL.pth | 512 |
| ultrafedfm | ckpts/UltraFedFM.pth | 768 |
| echocare | ckpts/echocare_encoder.pth | 2048 |
| usfm | ckpts/USFM_latest.pth | 768 |
| openus | ckpts/openus_cpt0150.pth | 768 |
| usf_mae | ckpts/USF-MAE_full_pretrain_43dataset_100epochs.pt | 768 |
| ultrasam | ckpts/UltraSam.pth | 256 |
| busgen | BUSGen/Checkpoints/Medical_BUSGen_6cls/ckpt_80_.pt | 256 |
| vjepa（额外选择） | 本项目 ckpts/vjepa_full/best_vjepa_model9639(paper).pt | 1408 |
| none（额外选择） | 不使用教师 | — |

`all` 只展开为上述八个基座模型。V-JEPA 和 none 需显式追加。
使用 `--baseline_root` 可替换外部仓库位置，`--vjepa_ckpt` 可替换 V-JEPA 权重。
初始化与预训练权重加载复用外部仓库 `detection/model.py` 的八种适配器，
其中 UltraSAM/BUSGen 再使用该仓库 `buildmodel.py`。运行依赖这两个文件及其模型依赖。

所有组统一：YOLO11m 初始化权重、学生输入 256、全局 batch 128、100 epochs、seed 0、
AdamW、lr 0.001、weight decay 0.0005、3 epochs warmup、cosine 到初始 LR 的 1%，
相同 YOLO 检测增强、相同训练/验证数据。关闭提前停止，AMP 默认开启。
默认全局 batch 与 nominal batch 相同，稳定阶段不额外累积梯度。
实际使用的完整 Ultralytics 配置保存在每组 args.yaml。
这些是新对比实验的显式统一配置，不声称逐项复刻旧脚本隐式使用的历史默认值。

教师默认在 224 输入上提取最后层空间特征，统一 ImageNet 归一化；
层次模型使用最后尺度，ViT 使用 patch token，UltraSAM 使用 image encoder 特征，
BUSGen 保留既有检测适配器的 128 原生编码分辨率，再插值输出特征。
V-JEPA 复制图片成两帧伪视频。`--teacher_batch 4` 控制每卡教师分块大小，
只影响教师内存占用，不改变学生 batch。OOM 时可以先改成 1。
因此比较的是各模型既有表示适配方案向同一学生转移的效果；教师的原生架构和特征尺度不同。

学生取默认第 6 层特征，经可训练 1×1 卷积匹配教师通道，教师特征插值到学生网格。
投影层注册在学生模型中，在建立优化器、DDP、EMA 之前创建；教师不进入学生 checkpoint。
只执行一次学生前向，不会因重复主干前向而额外更新 BatchNorm。

```text
每图目标 = YOLO 检测目标 + 1.0 × MSE + 0.5 × (1 − cosine)
```

实现按 Ultralytics 的检测损失口径将蒸馏项乘当前 batch 大小，
避免变更 batch 后蒸馏项相对检测损失被意外缩小。
`--alpha` / `--beta` 可以调整蒸馏权重；对比实验中应对全部教师使用相同值。
none 组不创建投影层，只有有监督检测训练。

默认输出结构：

```text
distill/det/output/<run_name>/
  uscl/
    comparison.json         教师来源、统一训练参数及版本
    args.yaml               Ultralytics 完整配置
    results.csv             每轮检测指标与损失（含 kd_mse/kd_cosine）
    weights/
      best.pt               最优 EMA checkpoint，包含投影层和蒸馏配置
      last.pt               最后 EMA checkpoint
      best_student.pt       导出的标准 YOLO 检测器，已去除投影层
  ultrafedfm/
  ...
```

best 按 Ultralytics 检测验证 fitness 选择。验证损失中蒸馏两项为 0，不调用教师。
完整训练结束时 best.pt/last.pt 的 optimizer 遵循 Ultralytics 的自动剥离行为。
部署和旧推理脚本优先使用 best_student.pt，它是标准 DetectionModel，无需导入 com.py。

```bash
python distill/det/infer.py \
  --model_path distill/det/output/foundation_kd_seed0/uscl/weights/best_student.pt \
  --output distill/det/output/foundation_kd_seed0/uscl/infer \
  --data_dir /home/lx/dataset/6cls_detection/images/val2017 \
  --label_dir /home/lx/dataset/6cls_detection/labels/val2017 \
  --conf 0.25 --imgsz 256
```

本次验证包括八种真实预训练教师的 CPU 前向及学生反向、none 对照、
USCL 四图一轮训练/验证/保存/标准权重重载，以及两进程 CPU DDP 两步后投影参数一致。
V-JEPA 分支复用原教师加载器；本次没有运行其完整加载/训练，也未运行完整数据集或 GPU DDP。

## VMamba 环境的 CUDA NMS 兼容设置（2026-10-25）

`/root/envs/vmamba` 的 torch/torchvision 为 2.8.0+cu129 / 0.23.0+cu129，
当前驱动 570.153.02 下 torchvision CUDA NMS 可复现 unsupported PTX toolchain 错误。
在该环境运行时追加 `--nms_backend torch`，使用 Ultralytics 自带的 GPU PyTorch NMS。
该参数通过配置传入所有 DDP 子进程，在 AMP 检查和检测验证之前生效；默认后端仍是 torchvision。
AMP 继续开启，教师的 selective-scan CUDA 扩展继续使用。

```bash
conda activate /root/envs/vmamba
cd /home/lx/baselines/vjepa
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m distill.det.com \
  --teachers openus --data /home/lx/dataset/6cls_detection/data.yaml \
  --model yolo11m.pt --device 0,1,2,3 --batch 128 --imgsz 256 --epochs 100 \
  --teacher_batch 4 --seed 0 --nms_backend torch \
  --output_dir distill/det/output \
  --run_name "openus_vmamba_$(date -u +%Y%m%d_%H%M%S)"
```

新增 GPU 验证：真实 OpenUS 教师 CUDA 前向输出有限；200 个随机框、三个 IoU 阈值下
GPU TorchNMS 与 torchvision CPU NMS 返回一致；VMamba 环境四卡、四张真实图片、
一轮训练通过 AMP 检查、蒸馏反向、验证、保存和最终验证。测试输出在
`output/_verification_vmamba_nms_20261025/openus/`。未启动完整 100 轮实验。
