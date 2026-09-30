torchrun --nnodes=1 --nproc_per_node=4 -m  distill.det_vjepa.train \
        --data /home/lx/dataset/6cls_detection/data-2cls-remap.yaml \
        --init_weights distill/det_vjepa/output/lesion_resolution_640_vs_768_20260928T083805_1134/768/best.pt \
        --imgsz 768 --batch 8 --epochs 20  \
        --output_dir distill/det_vjepa/output --exp_name vjepa_giant_multidepth_768best --lesion_only \
        --out_layers 13 26 39 