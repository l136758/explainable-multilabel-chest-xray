# Grad-CAM and IoBB Localization Summary

## Overview

Grad-CAM explanations were generated for the final **ConvNeXt-Tiny 320×320 BCE** classification model and evaluated against radiologist-drawn bounding boxes from the NIH ChestX-ray14 localization subset.

Notebook: `notebooks/convnext-gradcam-localization.ipynb`  
Kaggle: `layanabdullah11/convnext-gradcam-localization`

The classification model is unchanged and is not retrained in this notebook. The notebook rebuilds the 14-output ConvNeXt-Tiny architecture, loads the frozen checkpoint `convnext_tiny_320_best.pth`, and switches the model to evaluation mode before Grad-CAM generation.

The checkpoint comes from the final BCE training experiment `D3_ConvNeXtTiny_320_BCE`. The saved best checkpoint is from **epoch 3**, with validation Macro AUROC **0.8121**.

This is an educational research prototype and is not clinically validated.

---

## Target layer

The Grad-CAM target layer is:

`model.features[-1][-1].block[0]`

This is the depthwise convolution of the last block in the last ConvNeXt stage. Its output is a 10×10 grid for a 320×320 input.

Grad-CAM is generated from the **raw logit** of the selected pathology, not the sigmoid output, because the sigmoid saturates near 0 and 1 and flattens the gradients. Forward activations and backward gradients are captured with PyTorch hooks.

For each target class:

1. Run a forward pass.
2. Select the raw class logit.
3. Backpropagate the selected score.
4. Global-average-pool the gradients.
5. Use the pooled gradients as channel weights.
6. Compute the weighted sum of the activations.
7. Apply ReLU.
8. Min-max normalize the heatmap.

The heatmap is resized to the original image size. Pixels at or above the threshold `T` are kept, and the predicted box is the rectangle around them.

---

## Metrics

| Metric | Definition | What it rewards |
|---|---|---|
| **IoBB** | intersection ÷ predicted box area | the predicted box lying inside the ground-truth box |
| **IoU** | intersection ÷ union of both boxes | a box with the right position **and** the right size |
| **Box area** | predicted box area ÷ image area | — (reported so IoBB can be read correctly) |
| **Centre control** | IoBB of a same-sized box placed at the image centre | — (a baseline that ignores the heatmap) |

IoBB follows the definition used for ChestX-ray14 (Wang et al., 2017): intersection over the detected box area.

Because IoBB divides by the predicted box only, a smaller box can never lower it. That is why IoBB is always reported next to IoU, box area, and the centre control.

---

## Choosing the heatmap threshold

The threshold is tuned on `loc_tune` only, sweeping `T` from 0.05 to 0.95. It is selected by **mean IoU**.

Selected rows of the sweep (full sweep in `results/gradcam_tune_sweep.csv`):

| T | Mean IoBB | Mean IoU | Box area |
|---:|---:|---:|---:|
| 0.30 | 0.1945 | 0.1582 | 0.525 |
| 0.45 | 0.2628 | 0.1806 | 0.339 |
| 0.50 | 0.2917 | 0.1826 | 0.281 |
| **0.55** | **0.3128** | **0.1828** | **0.238** |
| 0.60 | 0.3311 | 0.1778 | 0.197 |
| 0.75 | 0.3796 | 0.1330 | 0.097 |
| 0.90 | 0.4109 | 0.0757 | 0.037 |
| 0.95 | 0.4228 | 0.0477 | 0.020 |

Mean IoBB keeps rising all the way to `T = 0.95` while the predicted box shrinks to about 2% of the image. Mean IoU peaks in the middle of the range and drops on both sides, so it gives a real optimum.

**Selected threshold: `T = 0.55`**

It is frozen before evaluation on `loc_report`.

---

## Protocol

- Final model: ConvNeXt-Tiny, 320×320 input.
- Preprocessing: the same ImageNet-normalized preprocessing used for inference.
- Tuning split: `loc_tune` (**376** annotated image-class pairs).
- Final evaluation split: `loc_report` (**373** annotated image-class pairs).
- `loc_tune` and `loc_report` patients are never used for classification training, validation, or testing.
- All **8** annotated pathologies are evaluated. The annotation label `Infiltrate` is mapped to the model label `Infiltration`.
- If several ground-truth boxes exist for the same image and pathology, the best overlap is used.
- Grad-CAM is computed once per image-class pair and reused for every threshold.
- Classification and localization metrics are kept separate.

---

## Results on `loc_report` (T = 0.55)

### Overall

| Metric | Result |
|---|---:|
| Mean IoBB | **0.3264** |
| Mean IoU | **0.1790** |
| Mean box area | 0.236 of the image |
| IoBB ≥ 0.1 | 0.6113 |
| IoBB ≥ 0.25 | **0.4209** |
| IoBB ≥ 0.5 | 0.2761 |
| Centre control, IoBB ≥ 0.25 | 0.3217 |

### Per class

| Pathology | n | Mean IoBB | Mean IoU | Box area | IoBB ≥ 0.25 | Centre ≥ 0.25 | Gain over centre |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cardiomegaly | 81 | 0.7598 | 0.3015 | 0.139 | 0.9259 | 0.9630 | −0.037 |
| Pneumonia | 53 | 0.3141 | 0.2072 | 0.321 | 0.4151 | 0.2642 | +0.151 |
| Effusion | 44 | 0.2575 | 0.1890 | 0.289 | 0.3409 | 0.1136 | +0.227 |
| Mass | 30 | 0.2202 | 0.1534 | 0.204 | 0.3333 | 0.1667 | +0.167 |
| Atelectasis | 57 | 0.1869 | 0.1502 | 0.180 | 0.2807 | 0.0526 | +0.228 |
| Infiltration | 41 | 0.2553 | 0.1455 | 0.300 | 0.3171 | 0.2683 | +0.049 |
| Pneumothorax | 33 | 0.0749 | 0.0607 | 0.350 | 0.1212 | 0.1212 | 0.000 |
| Nodule | 34 | 0.0596 | 0.0565 | 0.199 | 0.0588 | 0.0000 | +0.059 |

### Reading the results

- **Cardiomegaly** has by far the highest IoBB, but the centre control scores slightly higher. The heart sits in the middle of almost every chest X-ray, so this number reflects anatomy more than explanation quality.
- **Effusion, Atelectasis, Mass, and Pneumonia** clearly beat the centre control (+15 to +23 points), so Grad-CAM points at the right region for these classes.
- **Infiltration** is only slightly above the control.
- **Pneumothorax** is no better than the control, and **Nodule** is almost never localized. Nodules are small and the heatmap comes from a 10×10 grid.

Pneumonia is interesting: it has the weakest classification AUROC of the model (0.696) but one of the better localization results. Part of this may come from its large boxes, so it should be read together with its box area.

---

## Representative examples

For each class shown, the example closest to that class's mean IoBB was selected, so the examples are typical rather than hand-picked.

| Pathology | Image | IoBB | IoU |
|---|---|---:|---:|
| Cardiomegaly | `00004344_022.png` | 0.759 | 0.453 |
| Effusion | `00028509_007.png` | 0.263 | 0.186 |
| Nodule | `00019013_002.png` | 0.057 | 0.056 |

Each visualization shows the chest X-ray, the class-specific Grad-CAM heatmap, the radiologist box (red), and the Grad-CAM box (blue).

---

## Known weaknesses

- Grad-CAM produces coarse heatmaps (10×10 grid), not precise lesion segmentation.
- Localization quality varies strongly across pathologies.
- Small findings such as nodules are particularly hard to localize.
- The localization subset is much smaller than the classification dataset, so per-class results (n = 30 to 81) are noisy.
- Only one target layer was evaluated.
- These results measure overlap with the available boxes. They do not establish clinical correctness.

---

## Artifacts

| File | Contents |
|---|---|
| `notebooks/convnext-gradcam-localization.ipynb` | Grad-CAM, threshold sweep, final evaluation, per-class results, examples |
| `results/gradcam_tune_sweep.csv` | Full `loc_tune` sweep: IoBB, IoU, box area, centre control for every T |
| `results/gradcam_tune_sweep.png` | Sweep plot |
| `results/gradcam_overall_results.csv` | Overall `loc_report` results at T = 0.55 |
| `results/gradcam_iobb_results.csv` | Per-class `loc_report` results |
| `results/gradcam_report_per_image.csv` | Every image-class pair: IoBB, IoU, box area, centre control |
| `results/gradcam_representative_examples.csv` | The three examples |
| `results/gradcam_iobb_config.json` | Layer, threshold, split sizes, metric definitions |
| `results/gradcam_samples/` | Example overlays |

Model checkpoints are not stored in this repository.

## Reference

Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017). *ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases.* CVPR.
