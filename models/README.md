# Models

Model weights are **not stored in this repository** (large files, separate licences).

## 1. Deepfake detector (manual download, one time)

| Item | Value |
|---|---|
| Model | `prithivMLmods/Deep-Fake-Detector-v2-Model` |
| Source | https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model |
| Licence | Apache-2.0 (as stated on the model page) |
| Architecture | Vision Transformer (ViT), fine-tuned from `google/vit-base-patch16-224-in21k` |
| Labels | "Realism" / "Deepfake" (read from the model's `config.json`) |
| Size | a few hundred MB for the files needed to run; the whole repository is about 1.4 GB |
| Save location | `models/deepfake_vit/` (must contain `config.json`) |

Download with the helper script from the project root:

```bash
python download_model.py
```

If the folder is missing the app still runs, shows "Deepfake detector unavailable",
and the risk result becomes **Inconclusive**. It never fabricates a prediction.

Reported accuracy (about 92%) comes from the model author's own test split and does
not guarantee performance on other manipulation methods or real-world photos.

## 2. FaceNet weights (automatic)

DeepFace downloads the FaceNet weights automatically on first use and stores them in
your user folder (`~/.deepface/weights`), outside this repository. An internet
connection is needed for that first run only.
