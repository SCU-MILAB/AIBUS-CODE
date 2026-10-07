# Alpha-IoU candidate for a later experiment

He, Jiabo; Erfani, Sarah; Ma, Xingjun; Bailey, James; Chi, Ying; Hua, Xian-Sheng. Alpha-IoU: A Family of Power Intersection over Union Losses for Bounding Box Regression. NeurIPS 2021. Verified from the paper (arXiv v2, 2022-01-22), equations 3–4 and property 3, on 2026-10-07.

Primary source: https://arxiv.org/html/2110.13675
PDF: https://arxiv.org/pdf/2110.13675
Author code: https://github.com/jacobi93/Alpha-IoU

The paper applies power transformations separately to overlap and geometric penalty. Alpha-GIoU is 1 - IoU**alpha + ((enclosing_area - union)/enclosing_area)**alpha. Alpha=1 recovers GIoU. The IoU-term relative gradient is alpha*IoU**(alpha-1); alpha>1 emphasizes high-IoU examples. Alpha=3 is a reported robust choice, without adding model parameters.

Project hypothesis (unverified): replace only the GIoU power, first 1 to 3, retaining incumbent loss weights, training masks, initialization and evaluation. At IoU 0.9 the IoU-term gradient multiplier is 2.43; at 0.95 it is 2.7075. This may target the observed strict-IoU deficit but changes effective gradient magnitude and could hurt poorly localized examples.

Do not exponentiate signed GIoU itself. Before execution verify alpha=1 equality, zero loss for perfect boxes, finite gradients for disjoint/degenerate predictions, and unchanged empty-target DDP connectivity. Keep the current H170 source untouched; implement after reviewing the new loss/mask experiments. Other detectors' results do not establish benefits here.

Implemented on 2026-10-07 as training-only reg_iou_alpha with default 1.0 preserving original arithmetic. Candidate alpha=3 is queued after the sampler-order tests; no experiment result yet. Implementation checks are recorded in output/oct7_alpha_giou_implementation_checks.json and findings.md.
