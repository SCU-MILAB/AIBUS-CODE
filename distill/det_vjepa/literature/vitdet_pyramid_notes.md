# ViTDet feature-pyramid ablation and a local follow-up

Yanghao Li, Hanzi Mao, Ross Girshick, Kaiming He. **Exploring Plain Vision Transformer Backbones for Object Detection.** ECCV 2022. [arXiv record](https://arxiv.org/abs/2203.16527), [official proceedings PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690276.pdf).

## Verified observation

Section 3 and Figure 2 compare intermediate-stage FPN, last-map FPN and a parallel simple pyramid from the final stride-16 map. Table 1 reports ViT-L COCO box AP of 54.4, 54.6 and 54.6 respectively, versus 51.2 without a feature pyramid. The simple pyramid retains multiple output resolutions while removing FPN's cross-level top-down fusion. This concerns the authors' MAE-pretrained backbone and Mask R-CNN setting.

## Relevance and limits

Our ViT-G also exposes maps at a common native stride. The local neck first changes spatial resolution and then adds top-down features. Thus the existing `fpn_topdown_scale=0` offers a parameter-compatible test of independent pyramid outputs. With default final-map routing this resembles the cited neck design, but our transformations, strides, pretraining, FCOS head, ultrasound data and fixed-time continuation differ. Published COCO scores cannot predict our result.

## Candidate after the current FPN, batch and loss controls

Change only the incumbent recipe's top-down scale to zero; preserve initialization, routing, trainability, optimizer, loss, all fixed validation settings and the 120-minute training budget. Prediction: independent fine-level outputs may retain useful local detail and recover high-IoU AP, with a simpler feature combination. This is a candidate, not an already scheduled or completed run. Commit its exact protocol before execution; use complete validation and the global AP checkpoint gate. If the current incumbent has multi-depth routing, identify the test as parallel multi-depth outputs rather than a reproduction of the final-map ViTDet neck.

The earlier proposed 0.5 scale tests partial weakening. The zero-scale candidate tests the distinct simplification boundary; shrinking local numeric proposals around 1.0 would otherwise not necessarily visit it.
