# Grad-CAM and IoBB Localization Summary

## Overview

Grad-CAM explanations were generated for the final **ConvNeXt-Tiny 320×320 BCE** classification model and evaluated against radiologist-drawn bounding boxes from the NIH ChestX-ray14 localization subset.

Notebook: `notebooks/convnext-gradcam-localization.ipynb`  
Kaggle: `layanabdullah11/convnext-gradcam-localization`

The classification model is unchanged and is not retrained in this notebook. The notebook reconstructs the 14-output ConvNeXt-Tiny architecture, loads the frozen checkpoint `convnext_tiny_320_best.pth`, and switches the model to evaluation mode before Grad-CAM generation.

The checkpoint comes from the final BCE training experiment `D3_ConvNeXtTiny_320_BCE`. The saved best checkpoint is from **epoch 3**, with validation Macro AUROC **0.8121**.

This is an educational research prototype and is not clinically validated.

---

## Target layer

The Grad-CAM target layer is:

`model.features[-1][-1].block[0]`

This is the final depthwise convolution in the last ConvNeXt block.

Grad-CAM is generated from the **raw logit** of the selected pathology. Forward activations and backward gradients are captured with PyTorch hooks.

For each target class:

1. Run a forward pass.
2. Select the raw class logit.
3. Backpropagate the selected score.
4. Global-average-pool the gradients.
5. Use the pooled gradients as channel weights.
6. Compute the weighted sum of the activations.
7. Apply ReLU.
8. Min-max normalize the heatmap.

The resulting Grad-CAM map is resized back to image space for visualization and localization.

---

## Choosing the CAM binarization threshold

The normalized Grad-CAM heatmap is converted into a predicted localization box by thresholding the heatmap and taking the rectangular extent of the activated region.

The threshold is tuned on `loc_tune` only.

| Threshold | Mean IoBB | IoBB ≥ 0.1 | IoBB ≥ 0.25 | IoBB ≥ 0.5 |
|---:|---:|---:|---:|---:|
| 0.30 | 0.1945 | 0.4574 | 0.2553 | 0.1277 |
| 0.40 | 0.2400 | 0.5186 | 0.3271 | 0.1782 |
| 0.50 | 0.2917 | 0.5532 | 0.3670 | 0.2606 |
| 0.60 | 0.3311 | 0.5824 | 0.4149 | 0.3112 |
| 0.70 | 0.3649 | 0.5851 | 0.4548 | 0.3484 |
| 0.75 | 0.3796 | 0.5718 | 0.4681 | 0.3697 |
| 0.80 | 0.3875 | 0.5691 | 0.4734 | 0.3830 |
| 0.85 | 0.4010 | 0.5665 | 0.4734 | 0.3989 |
| **0.90** | **0.4109** | 0.5532 | 0.4707 | **0.4122** |

**Selected threshold: `0.90`**

The selected threshold is frozen before final evaluation on `loc_report`.

---

## Protocol

- Final model: ConvNeXt-Tiny, 320×320 input.
- Preprocessing: the same ImageNet-normalized preprocessing used for inference.
- Tuning split: `loc_tune`.
- Final evaluation split: `loc_report`.
- Tuning set: **376 annotated image-class pairs**.
- Report set: **373 annotated image-class pairs**.
- Localization metric: IoBB.
- Evaluated pathologies: **8**.
- If several ground-truth boxes exist for the same image and pathology, the best overlap is used.
- Classification and localization metrics are kept separate.

The eight evaluated pathologies are:

`Atelectasis`, `Cardiomegaly`, `Effusion`, `Infiltration`, `Mass`, `Nodule`, `Pneumonia`, and `Pneumothorax`.

The NIH annotation label `Infiltrate` is mapped to the model label `Infiltration` before evaluation.

---

## IoBB

The notebook evaluates localization using Intersection-over-Bounding-Box (IoBB).

The implementation used here is:

**intersection area ÷ predicted bounding-box area**

For predicted box `Bp` and ground-truth box `Bg`:

`IoBB = area(Bp ∩ Bg) / area(Bp)`

The reported cutoffs are:

- IoBB ≥ 0.1
- IoBB ≥ 0.25
- IoBB ≥ 0.5

---

## Results on `loc_report`

Across **373 held-out image-class pairs**:

| Metric | Result |
|---|---:|
| Mean IoBB | **0.4167** |
| IoBB ≥ 0.1 | **0.5657** |
| IoBB ≥ 0.25 | **0.4826** |
| IoBB ≥ 0.5 | **0.3941** |

### Per-class results

| Pathology | n | Mean IoBB | IoBB ≥ 0.1 | IoBB ≥ 0.25 | IoBB ≥ 0.5 |
|---|---:|---:|---:|---:|---:|
| Cardiomegaly | 81 | **0.8864** | 0.9506 | 0.9383 | 0.8889 |
| Pneumonia | 53 | 0.4408 | 0.5849 | 0.5094 | 0.4151 |
| Effusion | 44 | 0.4238 | 0.6364 | 0.5455 | 0.4091 |
| Mass | 30 | 0.3168 | 0.5000 | 0.4333 | 0.3000 |
| Infiltration | 41 | 0.2877 | 0.4634 | 0.3415 | 0.2439 |
| Atelectasis | 57 | 0.2566 | 0.4561 | 0.3158 | 0.2281 |
| Pneumothorax | 33 | 0.1057 | 0.2121 | 0.1212 | 0.0909 |
| Nodule | 34 | 0.0651 | 0.2353 | 0.1176 | 0.0000 |

Localization performance is strongly class-dependent.

- **Cardiomegaly** shows the strongest localization result.
- **Pneumonia** and **Effusion** show moderate overlap.
- **Mass**, **Infiltration**, and **Atelectasis** are weaker.
- **Pneumothorax** and **Nodule** are the most difficult classes in this evaluation.

---

## Representative examples

Representative cases were selected from the held-out localization results.

| Pathology | Image | IoBB |
|---|---|---:|
| Cardiomegaly | `00014706_007.png` | 0.9162 |
| Effusion | `00028974_016.png` | 0.4190 |
| Nodule | `00015141_002.png` | 0.0598 |

Each visualization shows:

- the original chest X-ray,
- the class-specific Grad-CAM heatmap,
- the radiologist-provided ground-truth box,
- the Grad-CAM-derived predicted box.

---

## Known weaknesses

- Grad-CAM produces coarse explanatory heatmaps rather than precise lesion segmentation.
- Localization quality varies substantially across pathologies.
- Small findings such as nodules are particularly difficult to localize.
- The localization subset is much smaller than the classification dataset.
- IoBB depends on the geometry of the predicted and annotated boxes.
- The selected heatmap threshold is specific to the tuning protocol used here.
- These localization results evaluate overlap with available bounding-box annotations; they do not establish clinical correctness.

---

## Artifacts

| File | Contents |
|---|---|
| `notebooks/convnext-gradcam-localization.ipynb` | Grad-CAM implementation, target-layer hooks, threshold tuning, held-out IoBB evaluation, per-class results, and representative examples |
| `notebooks/convnext-tiny-320-training.ipynb` | ConvNeXt-Tiny training experiment that produced the BCE checkpoint used by the localization notebook |

Model checkpoints are not stored in this personal repository.

