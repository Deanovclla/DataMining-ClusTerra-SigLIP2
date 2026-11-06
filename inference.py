import argparse
import os
import sys
import torch
import joblib
import numpy as np
from PIL import Image
from torch import nn
from transformers import AutoModel, AutoImageProcessor

DEFAULT_MODEL_ID = "google/siglip2-giant-opt-patch16-384"

def load_linear_probe(checkpoint_path: str, device: str):
    """
    Memuat linear probe head dari checkpoint PyTorch bundle.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint bundle tidak ditemukan di: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    feat_dim = ckpt["feat_dim"]
    labels_ind = ckpt["labels_ind"]

    head = nn.Sequential(
        nn.LayerNorm(feat_dim),
        nn.Linear(feat_dim, len(labels_ind))
    ).to(device)

    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    model_id = ckpt.get("model_id", DEFAULT_MODEL_ID)
    return head, labels_ind, model_id

def predict(image_path: str, model, image_processor, head, label_encoder, device: str, top_k: int = 5):
    """
    Menjalankan inferensi pada satu citra makanan.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File gambar tidak ditemukan: {image_path}")

    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        if hasattr(model, "get_image_features"):
            feats = model.get_image_features(**inputs)
        elif hasattr(model, "encode_image"):
            feats = model.encode_image(**inputs)
        else:
            out = model(**inputs)
            feats = None
            for key in ["image_embeds", "pooler_output", "last_hidden_state"]:
                if isinstance(out, dict) and key in out:
                    feats = out[key]
                    if key == "last_hidden_state":
                        feats = feats[:, 0]
                    break
            if feats is None:
                raise RuntimeError("Tidak dapat mengekstrak image feature embedding.")

        logits = head(feats)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_indices = np.argsort(probs)[::-1][:top_k]
    results = []
    for idx in top_indices:
        label = label_encoder.inverse_transform([idx])[0]
        results.append({
            "class": label,
            "confidence": float(probs[idx])
        })

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Inferensi Klasifikasi Citra Masakan Nusantara menggunakan SigLIP2 Linear Probe (Tim ClusTerra)"
    )
    parser.add_argument("--image", type=str, required=True, help="Path ke berkas gambar makanan (e.g. contoh.jpg)")
    parser.add_argument("--checkpoint", type=str, default="models/linear_probe_bundle.pth", help="Path ke linear_probe_bundle.pth")
    parser.add_argument("--encoder", type=str, default="models/label_encoder.joblib", help="Path ke label_encoder.joblib")
    parser.add_argument("--top_k", type=int, default=5, help="Jumlah prediksi teratas yang ditampilkan")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device target ('cuda' atau 'cpu')")

    args = parser.parse_args()

    # Fallback path jika dieksekusi dari root atau subfolder
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path) and os.path.exists("linear_probe_bundle.pth"):
        ckpt_path = "linear_probe_bundle.pth"

    enc_path = args.encoder
    if not os.path.exists(enc_path) and os.path.exists("label_encoder.joblib"):
        enc_path = "label_encoder.joblib"

    print(f"[*] Menggunakan perangkat: {args.device}")
    print(f"[*] Memuat linear head dari: {ckpt_path}")
    head, labels_ind, model_id = load_linear_probe(ckpt_path, args.device)

    print(f"[*] Memuat backbone SigLIP2: {model_id} ...")
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(args.device)
    model.eval()
    image_processor = AutoImageProcessor.from_pretrained(model_id)

    print(f"[*] Memuat label encoder dari: {enc_path} ...")
    le = joblib.load(enc_path)

    print(f"[*] Memproses citra: {args.image} ...")
    results = predict(args.image, model, image_processor, head, le, args.device, top_k=args.top_k)

    print("\n" + "="*50)
    print("HASIL PREDIKSI MASAKAN NUSANTARA")
    print("="*50)
    for rank, res in enumerate(results, start=1):
        bar_len = int(res["confidence"] * 30)
        bar = "█" * bar_len + "-" * (30 - bar_len)
        print(f"Top-{rank}: {res['class']:<16} | {res['confidence']*100:6.2f}% | [{bar}]")
    print("="*50)

if __name__ == "__main__":
    main()
