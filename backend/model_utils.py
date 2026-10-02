import torch
import torch.nn as nn
import json
import base64
import numpy as np
import matplotlib.cm as cm
from io import BytesIO
from torchvision import transforms
from torchvision.models import convnext_tiny
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# إعدادات ثابتة
# ============================================================
IMAGE_SIZE = 320
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

LABELS = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"
]
NUM_CLASSES = len(LABELS)

CHECKPOINT_PATH = "model/convnext_tiny_320_best.pth"
THRESHOLDS_PATH = "model/convnext_tiny_320_bce_thresholds.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# تحميل الـ thresholds
# ============================================================
with open(THRESHOLDS_PATH, "r") as f:
    THRESHOLDS = json.load(f)

# ============================================================
# تجهيز الصورة
# ============================================================
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# ============================================================
# بناء وتحميل الموديل
# ============================================================
model = convnext_tiny(weights=None)
in_features = model.classifier[2].in_features
model.classifier[2] = nn.Linear(in_features, NUM_CLASSES)
model = model.to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("✓ الموديل جاهز على:", device)

# ============================================================
# دالة التنبؤ الرئيسية
# ============================================================
def predict(image: Image.Image):
    input_tensor = eval_transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    results = []
    for i, label in enumerate(LABELS):
        threshold = THRESHOLDS.get(label, 0.5)
        probability = float(probs[i])
        results.append({
            "name": label,
            "probability": round(probability, 4),
            "threshold": threshold,
            "positive": probability >= threshold
        })

    return results

# ============================================================
# Grad-CAM setup
# ============================================================
target_layer = model.features[-1][-1].block[0]
activations = {}
gradients = {}

def forward_hook(module, inputs, output):
    activations["value"] = output

def backward_hook(module, grad_input, grad_output):
    gradients["value"] = grad_output[0]

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

def generate_gradcam_base64(image: Image.Image, target_class: int):
    image = image.convert("RGB")
    input_tensor = eval_transform(image).unsqueeze(0).to(device)

    model.zero_grad()
    logits = model(input_tensor)
    score = logits[0, target_class]
    score.backward()

    acts = activations["value"]
    grads = gradients["value"]
    weights = grads.mean(dim=(2, 3), keepdim=True)

    cam = (weights * acts).sum(dim=1)
    cam = torch.relu(cam)[0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.detach().cpu().numpy()

    # تكبير الـ heatmap لنفس حجم الصورة الأصلية
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(image.size)
    cam_resized = np.array(cam_img) / 255.0

    # تلوين الـ heatmap بألوان حرارية (jet colormap)
    colormap = plt.get_cmap("jet")
    colored_cam = colormap(cam_resized)[:, :, :3]
    colored_cam = np.uint8(colored_cam * 255)

    # دمج الألوان فوق الصورة الأصلية
    original = np.array(image).astype(float)
    overlay = 0.6 * original + 0.4 * colored_cam.astype(float)
    overlay = np.uint8(np.clip(overlay, 0, 255))

    final_img = Image.fromarray(overlay)

    buffer = BytesIO()
    final_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")