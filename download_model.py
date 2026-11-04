"""download_model.py - one-time download of the deepfake model into ./models.
Run once:  python download_model.py
Only the public model files are downloaded; none of your photos are involved."""
from deepfake_detector import MODEL_DIR, MODEL_ID, download_model

if __name__ == "__main__":
    print(f"Downloading {MODEL_ID} (a few hundred MB; the whole repository is about 1.4 GB) ...")
    download_model()
    print("Done. Model saved to:", MODEL_DIR)
