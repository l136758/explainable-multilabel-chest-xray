"""Grad-CAM utilities used in the chest X-ray localization notebook.

This module is extracted from the project's executed Grad-CAM notebook so the
core explainability and localization logic can be reused outside the notebook.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image


BBox = Tuple[float, float, float, float]


class GradCAM:
    """Grad-CAM helper for a trained PyTorch model and a selected target layer."""

    def __init__(self, model, target_layer, eval_transform, device):
        self.model = model
        self.target_layer = target_layer
        self.eval_transform = eval_transform
        self.device = device

        self.activations = {}
        self.gradients = {}

        self.forward_handle = target_layer.register_forward_hook(
            self._forward_hook
        )
        self.backward_handle = target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(self, module, inputs, output):
        self.activations["value"] = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients["value"] = grad_output[0]

    def generate(self, image: Image.Image, target_class: int) -> np.ndarray:
        """Generate a normalized Grad-CAM heatmap for one target class."""
        image = image.convert("RGB")
        input_tensor = self.eval_transform(image).unsqueeze(0).to(self.device)

        self.model.zero_grad()

        logits = self.model(input_tensor)
        score = logits[0, target_class]
        score.backward()

        acts = self.activations["value"]
        grads = self.gradients["value"]

        weights = grads.mean(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1)
        cam = torch.relu(cam)

        cam = cam[0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()

    def close(self):
        """Remove registered PyTorch hooks."""
        self.forward_handle.remove()
        self.backward_handle.remove()


def get_convnext_target_layer(model):
    """Return the target layer used in the project notebook.

    This is the final depthwise convolution in the last ConvNeXt block:
    model.features[-1][-1].block[0]
    """
    return model.features[-1][-1].block[0]


def heatmap_to_bbox(
    gradcam_map: np.ndarray,
    image_size: Tuple[int, int],
    threshold: float,
) -> Optional[BBox]:
    """Convert a Grad-CAM heatmap into the rectangular extent of active pixels."""
    cam_img = Image.fromarray(np.uint8(gradcam_map * 255))
    cam_img = cam_img.resize(image_size)

    cam_resized = np.array(cam_img) / 255.0
    binary_mask = cam_resized >= threshold

    ys, xs = np.where(binary_mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x = xs.min()
    y = ys.min()
    w = xs.max() - xs.min() + 1
    h = ys.max() - ys.min() + 1

    return (x, y, w, h)


def calculate_iobb(pred_bbox: Optional[BBox], gt_bbox: BBox) -> float:
    """Compute IoBB exactly as used in the project notebook.

    IoBB = intersection area / predicted bounding-box area.
    """
    if pred_bbox is None:
        return 0.0

    pred_x, pred_y, pred_w, pred_h = pred_bbox
    gt_x, gt_y, gt_w, gt_h = gt_bbox

    pred_x2 = pred_x + pred_w
    pred_y2 = pred_y + pred_h

    gt_x2 = gt_x + gt_w
    gt_y2 = gt_y + gt_h

    inter_x1 = max(pred_x, gt_x)
    inter_y1 = max(pred_y, gt_y)
    inter_x2 = min(pred_x2, gt_x2)
    inter_y2 = min(pred_y2, gt_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    intersection_area = inter_w * inter_h
    pred_area = pred_w * pred_h

    return intersection_area / pred_area if pred_area > 0 else 0.0


def calculate_iou(pred_bbox: Optional[BBox], gt_bbox: BBox) -> float:
    """Standard intersection over union of the two boxes."""
    if pred_bbox is None:
        return 0.0

    px, py, pw, ph = pred_bbox
    gx, gy, gw, gh = gt_bbox

    inter_w = max(0, min(px + pw, gx + gw) - max(px, gx))
    inter_h = max(0, min(py + ph, gy + gh) - max(py, gy))

    intersection_area = inter_w * inter_h
    union_area = pw * ph + gw * gh - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def centre_box(pred_bbox: Optional[BBox], image_size: Tuple[int, int]) -> Optional[BBox]:
    """Control box: same size as the Grad-CAM box, placed at the image centre.

    It ignores the heatmap, so Grad-CAM only shows real localization
    when it scores better than this box.
    """
    if pred_bbox is None:
        return None

    W, H = image_size
    _, _, w, h = pred_bbox

    return ((W - w) / 2, (H - h) / 2, w, h)
