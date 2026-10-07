# Localization-quality supervision candidates

Li, Xiang; Wang, Wenhai; Wu, Lijun; Chen, Shuo; Hu, Xiaolin; Li, Jun; Tang, Jinhui; Yang, Jian. Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection. NeurIPS 2020.

Primary source: https://proceedings.neurips.cc/paper_files/paper/2020/file/f0bda020d2470f2e74990a07a607ebd9-Paper.pdf
Verified 2026-10-07: equation 2, Figure 6 and inference description on page 6.

Quality Focal Loss uses soft targets y: zero for background and predicted-box/GT IoU for assigned positives. Its objective is BCE(probability, y) multiplied by abs(y - probability)**beta; beta=2 is studied. The full method combines classification and localization quality, uses distributional box regression, and feeds its joint classification score directly to NMS instead of multiplying by a separate centerness score.

Project candidates (untested adaptations):

- Replace geometric centerness training labels with detached predicted-box/assigned-GT IoU while retaining the existing BCE and unchanged cls*centerness decoding. This isolates one training target change and avoids altering the inference protocol. It is an IoU-supervised quality branch, not a reproduction of full GFL.
- Alternatively test a training-only QFL classification objective with fixed decoding. Label it as an adaptation because the original inference behavior differs. Current box_projection scope freezes the classification projection; encoder gradients still affect scores, but head capacity may constrain this test. Change only one principal variable per run.

Motivation from this project: original-to-incumbent AP90 increases with unchanged TP90, indicating ranking can contribute to strict-IoU AP. This is descriptive evidence, not proof quality supervision will improve AP. Test finite gradients, detached quality labels, exact empty-target behavior and checkpoint compatibility before any launch. Current training sources remain untouched; short-decay experiments are queued first.

Related primary source: Zhang, Haoyang; Wang, Ying; Dayoub, Feras; Sunderhauf, Niko. VarifocalNet: An IoU-Aware Dense Object Detector. CVPR 2021. https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.html
Its full model also introduces box-feature/refinement changes; benchmark gains do not isolate a training-only loss substitution for this project.
