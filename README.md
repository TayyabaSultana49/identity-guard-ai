# Identity Guard AI

**An AI-powered deepfake identity misuse detection system**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)
![Status](https://img.shields.io/badge/status-academic%20prototype-informational)
![Privacy](https://img.shields.io/badge/privacy-local%20by%20default-success)

Identity Guard AI helps a person check whether a suspected photograph may contain a
manipulated or misused version of their face. It runs two **independent** analyses, combines
them into a transparent and configurable risk assessment, records evidence, and guides the
person on what to do next. Everything runs locally on your computer.

> **Important:** this tool provides decision-support signals only. Face similarity does not
> prove identity, and a model prediction does not prove an image is fake.

## Table of contents
1. [Problem statement](#problem-statement)
2. [Objectives](#objectives)
3. [Key features](#key-features)
4. [Technology stack](#technology-stack)
5. [System architecture](#system-architecture)
6. [Project structure](#project-structure)
7. [Installation](#installation)
8. [Running the application](#running-the-application)
9. [App pages](#app-pages)
10. [How the AI models work](#how-the-ai-models-work)
11. [Models and references](#models-and-references)
12. [Testing](#testing)
13. [Privacy and security](#privacy-and-security)
14. [Limitations](#limitations)
15. [Responsible use](#responsible-use)
16. [Future enhancements](#future-enhancements)
17. [Author](#author)

## Problem statement
Manipulated images can place a person's face in content they never consented to. Victims often
cannot tell whether a suspicious image is a manipulation of their own face or simply someone
who looks similar, and they rarely know how to preserve evidence or where to report it. This
project combines face similarity with deepfake classification to give an explainable and
cautious indication, then supports the next step.

## Objectives
- Compare a reference photo with a suspected photo using face embeddings.
- Score the suspected photo for signs of manipulation using a pretrained classifier.
- Combine both signals into **Low / Moderate / High / Inconclusive** with plain-language reasons.
- Add supporting forensic hints (ELA heatmap, metadata) without letting them override the main result.
- Produce a PDF report with an evidence record and clearly stated limitations.
- Help the person act: next-steps guidance, a batch scan, and a Protect My Photo tool.
- Keep everything local, private and honest about uncertainty.

## Key features
| Feature | Details |
|---|---|
| Upload and validation | JPG/JPEG/PNG, up to 10 MB, extension and content checks, corruption detection |
| Face detection | OpenCV Haar cascade with boxes drawn on previews and face crops shown |
| Face similarity | FaceNet embeddings via DeepFace, compared with cosine similarity |
| Deepfake detection | Pretrained Vision Transformer classifier with a manipulation score |
| Risk assessment | Configurable decision table, colour-banded signal gauges, written explanation |
| Supporting forensics | Error Level Analysis (ELA) heatmap and metadata check, shown as weak hints only |
| Evidence record | SHA-256 fingerprint of the suspected file, timestamp, model and tool versions |
| PDF report | Results, evidence, limitations and next steps; photos excluded unless you opt in |
| Batch scan | One reference against up to 5 suspected photos, ranked by risk, CSV summary |
| Protect My Photo | Removes hidden metadata and GPS data, optional visible watermark, verifies the result |
| Next steps | Guidance on preserving evidence and reporting (India; editable for other countries) |
| Honest failure handling | If the deepfake model is missing, the app says so and returns **Inconclusive** |

**Not implemented:** Grad-CAM or other model-attention heatmaps, video or audio detection, model training.

## Technology stack
Python 3.11, Streamlit, OpenCV, Pillow, NumPy, DeepFace (FaceNet), PyTorch, Hugging Face
Transformers, ReportLab, pytest.

## System architecture
```
Reference photo --+--> image_processor --> face_matcher ------> similarity ----+
                  |    (validate, faces)    (FaceNet + cosine)                 |
Suspected photo --+--> image_processor --> deepfake_detector -> label+score ---+--> risk_assessment --> Dashboard + PDF
                  |                                                            (pipeline.py runs these steps)
                  +--> forensics.py --> ELA heatmap, metadata check, SHA-256 -----> supporting hints + evidence record
Any photo ------------> protect.py --> metadata removed + optional watermark (Protect My Photo)
```
The two analyses are independent: face similarity is never used as a substitute for deepfake
detection, and vice versa. Forensic hints never change the risk category. `pipeline.py` is
shared by the single analysis and the batch scan.

## Project structure
```
identity-guard-ai/
├── app.py                 # Streamlit UI (7 pages)
├── image_processor.py     # validation, loading, face detection
├── face_matcher.py        # FaceNet embeddings + cosine similarity
├── deepfake_detector.py   # pretrained ViT deepfake classifier
├── risk_assessment.py     # configurable combination logic
├── pipeline.py            # similarity + deepfake + risk for one pair (single and batch)
├── forensics.py           # ELA heatmap, metadata check, SHA-256, tool versions
├── protect.py             # metadata removal and watermark
├── guidance.py            # "Next Steps" content (India; edit for other countries)
├── report_generator.py    # PDF report (ReportLab)
├── download_model.py      # one-time deepfake model download
├── requirements.txt
├── packages.txt           # system libraries for hosting on Streamlit Community Cloud
├── .gitignore
├── .streamlit/config.toml # dark theme
├── assets/logo.png
├── models/README.md       # where model weights come from
├── sample_images/README.md
├── reports/.gitkeep
└── tests/                 # automated tests + manual end-to-end script
```

## Installation
Requirements: **Python 3.11** and about 3 GB of free disk space. Developed on Windows with VS Code.

Download or clone this repository, open the project folder in a terminal, then run:

```powershell
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_model.py
```

On macOS or Linux, activate the environment with `source venv/bin/activate` instead.

Notes:
- Make sure `(venv)` appears in your terminal before running `pip` or `streamlit`.
- `python download_model.py` fetches the deepfake model from Hugging Face once. It first downloads only the files
  needed to run (a few hundred MB) and falls back to the full repository (about 1.4 GB) if needed. See `models/README.md`.
- The first similarity run downloads the FaceNet weights automatically (internet needed once).
- OpenCV must be version 4 (`opencv-python<5`). OpenCV 5 removed the face detector this project needs
  (symptom: `module 'cv2' has no attribute 'CascadeClassifier'`). Fix:
  ```powershell
  pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
  pip install "opencv-python<5"
  ```
- If `deepface` reports a missing module such as `werkzeug`, run `pip install werkzeug`. For a `tf-keras` or
  "Keras 3" error, run `pip install tf-keras`. For NumPy 2 conflicts, run `pip install "numpy<2"`.

## Running the application
```powershell
venv\Scripts\activate
streamlit run app.py
```
Open http://localhost:8501, go to **Identity Analysis**, upload both photos, tick the consent box and
press **Analyse Identity**. The first analysis is slower because the models load. Always start the app
with `streamlit run app.py`, not by running `app.py` directly.

## App pages
| Page | What it does |
|---|---|
| Home | Explains the project; buttons jump to the main tools |
| Identity Analysis | Upload a reference and a suspected photo, confirm consent, run the analysis |
| Batch Scan | One reference against up to 5 suspected photos; ranked table, thumbnails, CSV summary (images are numbered, file names are not recorded) |
| Results Dashboard | Risk category, signal gauges, face crops, and tabs: *Why this category*, *Forensics (supporting)*, *Evidence record*, *Report* |
| Protect My Photo | Shows hidden data (camera, software, GPS), removes it, optional watermark, verifies the output has no metadata |
| Next Steps | How to preserve evidence and where to report; buttons return to your results |
| About the Project | What is implemented, what is not, and the limitations |

Buttons inside pages and the sidebar both navigate between pages.

## How the AI models work
- **Face similarity:** DeepFace runs FaceNet, which converts each face into a 128-number embedding.
  The app computes cosine similarity between the two embeddings (1.0 = same direction, 0 = unrelated).
  If several faces are present, the largest is used and a note is added.
- **Deepfake detection:** a Vision Transformer image classifier outputs the probability that the
  suspected image is "Deepfake" versus "Realism". The image is cropped to the largest detected face
  with a 30% margin, which is a design choice of this project.
- **Supporting forensics:** Error Level Analysis re-saves the suspected image as a JPEG in memory and
  maps where the compression difference is largest. Edits can cause such differences, but so can sharp
  edges, text, resizing and re-uploads, so it is a weak hint only. The metadata check looks for EXIF camera
  and software tags; GPS coordinates are never displayed, only whether they exist.
- **Evidence record:** a SHA-256 hash identifies the exact file analysed. Any re-save, crop or screenshot
  produces a different hash.
- **Risk assessment:** a small, readable decision table in `risk_assessment.py` maps the two scores into
  categories. **The thresholds are illustrative defaults and are not scientifically validated**; adjust
  `RISK_CONFIG` for your own data.

## Models and references
- **Deepfake model:** [`prithivMLmods/Deep-Fake-Detector-v2-Model`](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-v2-Model),
  Apache-2.0 licence, fine-tuned from `google/vit-base-patch16-224-in21k`. The model card reports about 92% accuracy
  on its own test split and does not name its training dataset beyond real versus deepfake face images.
- **Face recognition:** [DeepFace](https://github.com/serengil/deepface) by Sefik Serengil, using FaceNet:
  Schroff, Kalenichenko and Philbin, "FaceNet: A Unified Embedding for Face Recognition and Clustering", CVPR 2015.
- No dataset is included in this repository and no model was trained by the author. Model weights are not
  stored in the repository (see `models/README.md`).

## Testing
Automated tests (no AI models required):
```powershell
pytest -q
```
Expected: `63 passed`. They cover:
- image validation (valid JPG/PNG, unsupported types, empty, oversized, corrupted and renamed files)
- no-face handling and the cosine similarity calculation
- the missing-model case (must return "unavailable", never a fabricated prediction)
- every risk-table outcome and the combined pipeline
- ELA, metadata reading, SHA-256 hashing and PDF generation with and without photos
- Batch Scan and Protect My Photo logic, page navigation buttons, and a render check of every page
- hosted-demo mode (notice shown, model download fallback, download failure handled)

Manual end-to-end check with real photos you have permission to use:
```powershell
python tests/manual_check.py <reference-photo> <suspected-photo>
```

| Case | Expected result |
|---|---|
| Two photos of the same person | Higher similarity than for two different people (exact values vary) |
| Image without a face | "No face detected"; the Analyse button stays disabled |
| Several faces | "N faces detected (largest face will be used)" plus a caveat |
| `.gif`, `.txt`, or text renamed to `.jpg` | Rejected with a clear message |
| Deepfake model not downloaded | "Deepfake detector unavailable"; risk is **Inconclusive** |
| Batch scan including a no-face image | That row shows "Skipped: no face detected"; the rest are ranked |
| Protect My Photo with a geotagged JPEG | Shows GPS found, then "Verified: no metadata" on the output |
| PDF report | Includes evidence record and next steps; no photographs unless you opt in |

## Privacy and security
- When you run the app on your own computer, photos are processed locally, in memory, and are never permanently stored.
- No biometric data is sent to third-party services. Model files are downloaded from public sources; your photos are not involved.
- File type, content and size are validated before processing (max 10 MB).
- Logs contain event types only, never file names or image data.
- The PDF report excludes photographs unless you tick the option; the evidence record contains only a hash.
- Use the **Clear my data** button in the sidebar to remove session data.

## Deployment (optional hosted demo)
GitHub stores the code but cannot run the app. To get a public link you need a host that runs Python, for example
[Streamlit Community Cloud](https://streamlit.io/cloud), which deploys directly from a GitHub repository:

1. Push this repository to GitHub (public).
2. On Streamlit Community Cloud, choose **Create app**, select the repository, branch `main` and the file `app.py`.
3. Under **Advanced settings**, choose **Python 3.11** and add this secret so the app switches to hosted-demo mode:
   ```toml
   IDENTITY_GUARD_HOSTED = "1"
   ```
4. Deploy. On first start the app downloads the deepfake model automatically (a few minutes).

Things to know:
- **Privacy changes when hosted.** Uploaded photos are sent to the host's server. The app shows a notice and holds photos
  in memory only, but the statement "processed locally" no longer applies. Only upload photos of people who agreed, or
  AI-generated faces.
- **Free hosting has limits.** Streamlit Community Cloud documents roughly 2.7 GB of memory and puts idle apps to sleep, so a
  large AI app like this one may run out of memory. If that happens, run it locally or use a host with more memory.
- The hosted version has not been verified by the author on every platform; check the host's build logs if deployment fails.

## Limitations
- Deepfake classifiers generalise poorly to manipulation methods that were not in their training data, and
  degrade on low-resolution or heavily compressed images. A high score on a retouched studio photograph can be a false positive.
- Face similarity changes with pose, lighting, age and image quality.
- The Haar-cascade face detector can miss side-on, small or occluded faces.
- ELA and metadata are weak hints: social media strips metadata, and recompression creates ELA patterns without any editing.
- Protect My Photo does not prevent deepfakes; watermarks can be cropped out.
- Risk thresholds are illustrative and not scientifically validated. Results must never be treated as proof.
- The Next Steps contacts are for India and were checked on 2026-11-04. Re-check them before relying on them.
  This is general information, not legal advice.
- No model-attention explainability (Grad-CAM), and no video or audio support, in this version.

## Responsible use
Use this tool only with photographs you have permission to analyse. Do not use it to accuse, harass
or identify anyone. A result from this tool is not evidence of wrongdoing on its own.

## Future enhancements
Grad-CAM or attention-based explanations, larger batch scans, threshold calibration on a labelled dataset,
reporting guidance for other countries, an EfficientNet or Xception based detector, better face detectors,
and video support.

## Author
Tayyaba Sultana, Tandle Dhanasree, Syeda Rida Faseeh


