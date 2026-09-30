CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 -m \
  distill.det_vjepa.train --data /home/lx/dataset/6cls_detection/data-2cls-remap.yaml \
  --init_weights distill/det_vjepa/output/lesion_resolution_640_vs_768_20260928T083805_1134/768/best.pt \
  --output_dir distill/det_vjepa/output --exp_name lesion_768_finetune_from_best --lesion_only \
  --black_threshold 3 --selection_metric lesion_AP50_95 --imgsz 768 --batch 16 --accum 1 \
  --val_batch 4 --epochs 20 --lr_backbone 1e-6 --lr_head 1e-5 --drop_path 0.1 \
  --layer_decay 0.85 --weight_decay 0.05 --warmup_epochs 0.5 --min_lr_ratio 0.01 \
  --clip_grad 1.0 --reg_weight 1.0 --val_every 1 --val_max_images 0 \
  --final_val_max_images 0 --conf_thres 0.25 --iou_thres 0.7 --save_period 0 