# control 训练权重链（2026-10-10）

来源：各阶段config.json/init_weights、best_prediction.metrics.json，以及历史protocol与训练日志。按实际初始化指针追踪，不按实验编号或keep列表推测。

V-JEPA encoder → schedfix → 768 lesion-only → H16 → H29 → H32 → H49 → H72 → control。

每次 init_weights 仅继承模型权重，优化器和LR调度重新开始；选中的best之后产生的更新不进入下一阶段。

| 阶段 | 训练范围 | 预算 | 最佳checkpoint epoch（从0计） | 病灶AP50–95 (%) |
|---|---|---|---:|---:|
| vjepa_giant_fcos_schedfix | backbone+neck+完整双类head | 24 epochs | 17 | 79.586474 |
| lesion_resolution_640_vs_768_20260928T083805_1134/768 | backbone+neck+完整病灶单类head | 6 epochs | 2 | 81.041249 |
| autoresearch_oct2_h16_center_radius_1p0 | neck+完整head；backbone冻结 | 120分钟 | 1 | 81.076915 |
| autoresearch_oct3_h29_center_radius_1p25 | neck+完整head；backbone冻结 | 120分钟 | 0 | 81.106141 |
| autoresearch_oct3_h32_seed_1 | bbox output+Scale；其余冻结 | 120分钟 | 1 | 81.132893 |
| autoresearch_oct3_h49_freeze_backbone_false | backbone+bbox output+Scale | 120分钟 | 0 | 81.138954 |
| autoresearch_oct3_h72_reg_weight_0p5 | backbone+bbox output+Scale | 120分钟 | 0 | 81.189090 |
| h72_hypotheses_20261009_10ep/train/control | backbone+bbox output+Scale | 10 epochs | 2 | 81.292094 |

## 逐阶段变化

- schedfix：640常规双类；从V-JEPA encoder checkpoint初始化主干；24epochs；每卡batch8/accum1；LR1e-5/1e-4；warmup2；layer_decay.85；wd.05；drop_path.1。best_epoch17（第18轮）。原常规两类总mAP选优，与后续病灶主指标不同。
- 768：从schedfix best初始化；data-2cls-remap.yaml；lesion_only；分类投影提取病灶通道；正常图保留为空目标；背景由既有业务规则补全。两卡batch4/accum4，有效batch32；6epochs，LR2e-6/2e-5，warmup.5。best_epoch2（第3轮）。
- H16：从768 best初始化；冻结backbone，训练neck/head；center_radius1.5→1.0；四卡batch4/accum4，有效batch64；warmup.1、cosine_epochs2、120分钟。best_epoch1（完成第2轮）。
- H29：从H16 best初始化；center_radius1.0→1.25；其余主要设置同H16。best_epoch0（第1轮），后续更新不继承。
- H32：从H29 best初始化；实际采用box_projection，冻结backbone/neck/分类/reg tower/centerness，仅bbox output与Scale训练；seed1。日志确认9223个可训练参数。best_epoch1为预算截止的部分第2轮（2220microbatches）。
- H49：从H32 best初始化；解冻backbone，保持box_projection；仍reg_weight1。日志确认1012183175可训练参数。best_epoch0为预算截止的部分第1轮（2584microbatches）。
- H72：从H49 best初始化；reg_weight1→.5；其余主要设置保持。best_epoch0为部分第1轮（2984microbatches），病灶AP=.8118909046053886。
- control：从H72 best初始化；2卡batch4/accum8，有效batch64；完整10epochs、cosine_epochs10、warmup.1、train_minutes0；所有损失梯度保留；原按正位置平均回归；无新残差、无新增增强。seed1/sampler0，并在初始化后重置训练RNG。

所有autoresearch节点配置的epochs100只是上限，实际按120分钟预算结束；这些节点不能写成每次训练100epochs。

## 当前control模型行为

ViT-G主干结构不变，2D图像复制为两帧；最终stride16特征→stride8/16/32金字塔→FCOS。仅backbone、bbox projection和每层Scale可训练；冻结分支仍向backbone回传分类与centerness梯度。总损失Lcls+.5LGIoU+Lctr；不使用附加边界L1或Alpha-GIoU；固定conf.25/NMS IoU.7、完整5000张验证，以病灶AP50–95选best。

旧schedfix双类评估为original_xyxy_clip_after_nms_v1；后续单类业务评估为lesion_only_original_gt_complement_background_perfect_match_v2，两阶段总mAP不能直接作同口径增益。

手动640/768其他分支、reg2、额外768续训、增强、多层特征及其他未指向本链的autoresearch试验没有被control继承。
