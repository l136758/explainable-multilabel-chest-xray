Explainable Multi-Label Chest X-Ray Diagnosis

An explainable deep learning system for multi-label classification of 14 thoracic pathologies from chest X-rays, using ConvNeXt-Tiny for prediction and Grad-CAM for visual explanations.

Developed as a Samsung Innovation Campus AI Capstone Project by Team Core (6 members).

Research and educational prototype — not intended for clinical use.

⸻

My Contribution — Explainable AI & Localization

I worked as the XAI Specialist, focusing on the Grad-CAM explainability and localization evaluation component of the project.

My work included:

* Integrating Grad-CAM with the final ConvNeXt-Tiny model
* Selecting the Grad-CAM target layer
* Generating class-specific heatmaps
* Tuning the heatmap threshold on the dedicated loc_tune split
* Evaluating localization on the held-out loc_report split
* Comparing Grad-CAM localization with radiologist-drawn bounding boxes using Intersection-over-Bounding-Box (IoBB)
* Analyzing and reporting localization performance across the 8 pathologies with available bounding-box annotations

Localization Results

Metric	Result
Evaluated image-class pairs	373
Mean IoBB	0.417
Pairs with IoBB ≥ 0.50	39.4%
Cardiomegaly Mean IoBB	0.886
Nodule Mean IoBB	0.065

My Grad-CAM / Localization Notebook:
notebooks/convnext-gradcam-localization.ipynb

Also available on Kaggle:
https://www.kaggle.com/code/layanabdullah11/convnext-gradcam-localization

⸻

Project Overview

The system analyzes a chest X-ray and produces independent probabilities for 14 thoracic pathologies.

The final pipeline is:

Chest X-Ray
     ↓
Preprocessing
     ↓
ConvNeXt-Tiny
     ↓
14 Pathology Probabilities
     ↓
Class-Specific Decision Thresholds
     ↓
Flagged Findings
     ↓
Grad-CAM Explanation
     ↓
Web Application

Grad-CAM provides a class-specific visual explanation showing which regions of the image influenced a selected prediction.

⸻

Dataset

The project uses a working subset of the NIH ChestX-ray14 dataset:

* 31,077 chest X-ray images
* 11,907 unique patients
* 14 pathology labels
* Patient-level splitting to prevent patient leakage
* Separate loc_tune and loc_report patient groups for localization evaluation

The 14 modeled pathologies are:

Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia, Pleural Thickening, Cardiomegaly, Nodule, Mass, and Hernia.

⸻

Model

Three architectures were evaluated using the same patient-level data split:

Model	Test Macro AUROC
ResNet50	≈ 0.783
DenseNet-121	0.7959
ConvNeXt-Tiny	0.8158

ConvNeXt-Tiny was selected as the final classification model.

The final model uses:

* 320×320 input resolution
* ImageNet-pretrained weights
* BCEWithLogitsLoss
* AdamW optimizer
* Mixed-precision training
* Early stopping
* Validation-tuned per-class decision thresholds

Threshold tuning improved Test Macro F1 from 0.3034 to 0.4236.

⸻

Grad-CAM Explainability

Prediction probabilities alone do not show which image regions influenced the model.

Grad-CAM (Gradient-weighted Class Activation Mapping) was used to generate class-specific heatmaps showing the regions that influenced each selected prediction.

The same chest X-ray can therefore produce different Grad-CAM explanations for different pathologies.

⸻

Localization Evaluation

Grad-CAM was evaluated quantitatively rather than relying only on visual inspection.

Radiologist-drawn bounding-box annotations were available for 8 pathologies:

* Atelectasis
* Cardiomegaly
* Effusion
* Infiltration
* Mass
* Nodule
* Pneumonia
* Pneumothorax

The model still predicts all 14 pathologies. The 8-pathology limitation applies specifically to quantitative localization evaluation because bounding-box annotations were available only for these classes.

Localization was evaluated using Intersection-over-Bounding-Box (IoBB).

To keep localization threshold selection separate from final evaluation:

loc_tune
   ↓
Select Heatmap Threshold
   ↓
Freeze Threshold
   ↓
loc_report
   ↓
Final IoBB Evaluation

The final evaluation achieved a Mean IoBB of 0.417 across 373 image-class pairs.

Localization performance varied substantially by pathology, with Cardiomegaly showing strong localization while Nodule remained considerably more challenging.

⸻

End-to-End Application

The final system combines:

* ConvNeXt-Tiny for multi-label classification
* Grad-CAM for class-specific visual explanations
* FastAPI for model serving
* HTML / CSS / JavaScript for the web interface

A user uploads a chest X-ray, receives pathology prediction scores, and can view a Grad-CAM explanation for a selected finding.

The radiologist-drawn bounding boxes are used for localization evaluation and are not required from the application user.

⸻

Key Results

Metric	Result
Final Architecture	ConvNeXt-Tiny
Test Macro AUROC	0.8158
Test Macro F1	0.4236
Macro ECE	0.0258
Grad-CAM Mean IoBB	0.417
IoBB ≥ 0.50	39.4%

⸻

Tools & Technologies

Python · PyTorch · Torchvision · ConvNeXt-Tiny · Grad-CAM · NumPy · Pandas · Matplotlib · scikit-learn · FastAPI · HTML · CSS · JavaScript · Kaggle · GitHub

⸻

Limitations

* This project is a research and educational prototype, not a clinically validated diagnostic system.
* Performance varies across pathologies.
* Quantitative Grad-CAM localization evaluation was limited to the 8 pathologies with available bounding-box annotations.
* Grad-CAM indicates regions that influenced the model’s prediction; it does not establish the true clinical location of disease.
* External-dataset validation and expert radiologist review would be required for further clinical evaluation.

⸻

Project Resources

Original Team Project

The complete collaborative project, including the model-development notebooks, backend, frontend, and team results, is available in the original Team Core repository:

https://github.com/ABDULLHALSALHI/team-core-chest-xray

My XAI Work

Grad-CAM / Localization Notebook:
notebooks/convnext-gradcam-localization.ipynb

Grad-CAM / Localization Report:
docs/gradcam-localization-report.pdf

Kaggle Notebook:
https://www.kaggle.com/code/layanabdullah11/convnext-gradcam-localization

Dataset

NIH ChestX-ray14:
https://www.kaggle.com/datasets/nih-chest-xrays/data

⸻

Samsung Innovation Campus

Final AI Capstone Project — Team Core, Group 11

The project brought together data engineering, deep learning, explainable AI, model evaluation, backend development, and frontend development into an end-to-end chest X-ray analysis prototype.
