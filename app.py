from pathlib import Path
from io import BytesIO
import json

import gdown
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm
from safetensors.torch import load_file

from flask import Flask, request, jsonify, send_from_directory

# ===== 設定 =====
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "maxvit-utkface" / "maxvit-utkface.safetensors"
META_PATH = BASE_DIR / "models" / "maxvit-utkface" / "meta.json"
STATIC_DIR = BASE_DIR / "static"

MODEL_NAME = "maxvit_tiny_tf_224"
IMG_SIZE = 224
RACE_CLASSES = ["White", "Black", "Asian", "Indian", "Others"]
GENDER_CLASSES = ["Male", "Female"]

# ===== Google Drive からモデルファイルをダウンロード =====
FILE_ID = "1T3ypLy3n46L8CYspfmYYBgOxurYOd8gy"

if not Path(MODEL_PATH).exists():
    print("\033[94mDownloading model...\033[0m")
    gdown.download(id=FILE_ID, output=str(MODEL_PATH))
    print("\033[92mModel downloaded\033[0m")


# meta.json があればそちらを優先
if META_PATH.exists():
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    MODEL_NAME = meta.get("model_name", MODEL_NAME)
    IMG_SIZE = meta.get("img_size", IMG_SIZE)
    RACE_CLASSES = meta.get("race_classes", RACE_CLASSES)
    GENDER_CLASSES = meta.get("gender_classes", GENDER_CLASSES)

NUM_RACE = len(RACE_CLASSES)
NUM_GENDER = len(GENDER_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ===== モデル定義（学習時と同一構造） =====
class MaxViTMultiTask(nn.Module):
    def __init__(self, model_name=MODEL_NAME, pretrained=False,
                 num_race=NUM_RACE, num_gender=NUM_GENDER):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.race_head = nn.Linear(feat_dim, num_race)
        self.gender_head = nn.Linear(feat_dim, num_gender)
        self.age_head = nn.Linear(feat_dim, 1)

    def forward(self, x):
        feat = self.backbone(x)
        return (
            self.race_head(feat),
            self.gender_head(feat),
            self.age_head(feat).squeeze(-1),
        )


# ===== モデル読み込み =====
print(f"Loading model from: {MODEL_PATH}")
model = MaxViTMultiTask(model_name=MODEL_NAME, pretrained=False).to(device)
state = load_file(str(MODEL_PATH))
missing, unexpected = model.load_state_dict(state, strict=False)
if missing:
    print(f"[warn] missing keys: {len(missing)}")
if unexpected:
    print(f"[warn] unexpected keys: {len(unexpected)}")
model.eval()

# ===== 前処理 =====
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def predict_image(pil_image: Image.Image):
    image = pil_image.convert("RGB")
    x = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        race_logits, gender_logits, age_out = model(x)
        race_probs = F.softmax(race_logits, dim=1).cpu().numpy()[0]
        gender_probs = F.softmax(gender_logits, dim=1).cpu().numpy()[0]
        age_pred = float(age_out.cpu().numpy().item())

    # 年齢を 0–116 にクリップ
    age_pred = max(0.0, min(116.0, age_pred))

    # 年齢は回帰なので、5歳刻みの bin にして確率風の分布も作る（表示用）
    bin_edges = list(range(0, 121, 10))  # 0-10, 10-20, ..., 110-120
    bin_labels = [f"{bin_edges[i]}-{bin_edges[i+1]-1}" for i in range(len(bin_edges) - 1)]
    # ガウシアン近似で年齢bin確率を作る
    sigma = 6.0
    centers = np.array([(bin_edges[i] + bin_edges[i+1] - 1) / 2 for i in range(len(bin_edges) - 1)])
    weights = np.exp(-((centers - age_pred) ** 2) / (2 * sigma ** 2))
    weights = weights / weights.sum()

    race_idx = int(np.argmax(race_probs))
    gender_idx = int(np.argmax(gender_probs))

    return {
        "age": {
            "value": round(age_pred, 1),
            "bins": bin_labels,
            "probs": [float(w) for w in weights],
        },
        "gender": {
            "label": GENDER_CLASSES[gender_idx],
            "confidence": float(gender_probs[gender_idx]),
            "classes": GENDER_CLASSES,
            "probs": [float(p) for p in gender_probs],
        },
        "race": {
            "label": RACE_CLASSES[race_idx],
            "confidence": float(race_probs[race_idx]),
            "classes": RACE_CLASSES,
            "probs": [float(p) for p in race_probs],
        },
    }


# ===== Flask =====
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")


@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "no image file"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    try:
        image = Image.open(BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": f"invalid image: {e}"}), 400

    try:
        result = predict_image(image)
    except Exception as e:
        return jsonify({"error": f"inference failed: {e}"}), 500

    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": str(device)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

