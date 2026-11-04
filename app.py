"""app.py - Identity Guard AI Streamlit dashboard.

Run with:  streamlit run app.py
Photos live only in memory (session state) and are never written to disk."""
import csv
import io
import logging
import os
from datetime import datetime

import streamlit as st
from PIL import Image

from deepfake_detector import MODEL_ID, download_model, model_available
from forensics import library_versions, read_metadata, run_ela, sha256_hex
from guidance import GUIDANCE
from image_processor import (ImageValidationError, MAX_FILE_SIZE_MB, crop_largest_face,
                             detect_faces, draw_face_boxes, face_status, load_image)
from pipeline import analyse_pair
from protect import MAX_WATERMARK_CHARS, protect_photo
from report_generator import generate_report
from risk_assessment import RISK_CONFIG, manipulation_signal, similarity_signal

# Logs contain event types only, never filenames or image data
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("identity_guard")

st.set_page_config(page_title="Identity Guard AI", page_icon="assets/logo.png", layout="wide")

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.card {background:#111b2e; border:1px solid #1f3558; border-radius:12px; padding:1.1rem 1.3rem; margin-bottom:1rem;}
.card h4 {margin:0 0 .4rem 0; color:#22d3ee;}
.metric-label {color:#8fa3c0; font-size:.85rem;}
.metric-value {font-size:1.7rem; font-weight:700; color:#e6edf7;}
.metric-sub {color:#8fa3c0; font-size:.85rem; margin-top:.3rem;}
.badge {display:inline-block; padding:.45rem 1.1rem; border-radius:999px; font-weight:700; font-size:1.2rem;}
.b-low {background:#0f3d2e; color:#4ade80; border:1px solid #4ade80;}
.b-mod {background:#4a3a0a; color:#fbbf24; border:1px solid #fbbf24;}
.b-high {background:#4a1414; color:#f87171; border:1px solid #f87171;}
.b-inc {background:#1f2a40; color:#94a3b8; border:1px solid #94a3b8;}
.privacy {display:inline-block; padding:.2rem .7rem; border-radius:999px; background:#0c2b33;
          color:#22d3ee; border:1px solid #155e75; font-size:.8rem; margin-left:.6rem;}
.small {color:#8fa3c0; font-size:.85rem;}
.gauge {margin-bottom:1.2rem;}
.gauge-label {display:flex; justify-content:space-between; color:#c7d4e8; font-weight:600; margin-bottom:.5rem;}
.gauge-bar {position:relative; display:flex; height:14px; border-radius:7px;}
.seg {height:14px;}
.seg-low {background:#166534; border-radius:7px 0 0 7px;}
.seg-mid {background:#a16207;}
.seg-high {background:#991b1b; border-radius:0 7px 7px 0;}
.marker {position:absolute; top:-4px; width:4px; height:22px; background:#fff; border-radius:2px;
         box-shadow:0 0 6px #22d3ee; transform:translateX(-2px);}
.gauge-note {color:#8fa3c0; font-size:.85rem; margin-top:.5rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Hosted-demo mode: set the environment variable IDENTITY_GUARD_HOSTED=1 (on Streamlit Community
# Cloud, add it as a secret) when the app runs on a server instead of your own computer.
HOSTED = os.environ.get("IDENTITY_GUARD_HOSTED", "") == "1"
HOSTED_NOTICE = ("**Hosted demo:** photos you upload here are sent to the server that runs this app. "
                 "They are held in memory only and not saved by the app, but they do leave your device. "
                 "Only upload photos of people who have agreed, or AI-generated faces.")

PAGES = ["Home", "Identity Analysis", "Batch Scan", "Results Dashboard",
         "Protect My Photo", "Next Steps", "About the Project"]
MAX_BATCH = 5
RISK_RANK = {"High Risk": 0, "Moderate Risk": 1, "Inconclusive": 2, "Low Risk": 3}

# Allow code to switch page (must happen before the radio widget is created)
if "goto" in st.session_state:
    st.session_state["page"] = st.session_state.pop("goto")

with st.sidebar:
    st.image("assets/logo.png", width=90)
    st.markdown("### Identity Guard AI")
    page = st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
    st.markdown('<span class="privacy">' + ("Hosted demo. Held in memory only." if HOSTED
                else "Processed locally. Nothing stored.") + '</span>', unsafe_allow_html=True)
    st.markdown('<p class="small">Prototype for academic use. Results are decision-support '
                'signals, not proof.</p>', unsafe_allow_html=True)
    if st.button("Clear my data"):
        for key in ("results", "batch", "ref_up", "sus_up", "b_ref", "b_sus", "p_up"):
            st.session_state.pop(key, None)
        st.rerun()


def card(title, body):
    st.markdown(f'<div class="card"><h4>{title}</h4>{body}</div>', unsafe_allow_html=True)


def metric(label, value, sub=""):
    extra = f'<div class="metric-sub">{sub}</div>' if sub else ""
    st.markdown(f'<div class="card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div>{extra}</div>', unsafe_allow_html=True)


def gauge(label, value, low, high, note):
    """Colour-banded bar (green/amber/red) with a marker at `value` (0 to 1)."""
    pos = max(0.0, min(1.0, value)) * 100
    lo, hi = low * 100, high * 100
    st.markdown(
        f'<div class="gauge"><div class="gauge-label"><span>{label}</span><span>{value:.2f}</span></div>'
        f'<div class="gauge-bar"><div class="seg seg-low" style="width:{lo}%"></div>'
        f'<div class="seg seg-mid" style="width:{hi - lo}%"></div>'
        f'<div class="seg seg-high" style="width:{100 - hi}%"></div>'
        f'<div class="marker" style="left:{pos}%"></div></div>'
        f'<div class="gauge-note">{note}</div></div>', unsafe_allow_html=True)


def read_upload(upload):
    """Validate an uploaded file; return (image, error_message)."""
    if upload is None:
        return None, None
    try:
        return load_image(upload.getvalue(), upload.name), None
    except ImageValidationError as e:
        return None, str(e)


@st.cache_resource(show_spinner="Downloading the deepfake model (first start only, a few minutes)...")
def _fetch_model():
    """Hosted mode only: download the model once per server start. Failures are not cached."""
    download_model()
    return True


def prepare_hosted_model():
    """Make sure the deepfake model exists on a hosted server; never crash the app."""
    if not HOSTED or model_available() or st.session_state.get("model_dl_failed"):
        return
    try:
        _fetch_model()
    except Exception as exc:
        log.error("Hosted model download failed: %s", type(exc).__name__)
        st.session_state["model_dl_failed"] = True
        st.error("The deepfake model could not be downloaded on this server, so results will be "
                 "Inconclusive.")


def go(target):
    """Callback: ask the app to switch page on the next run."""
    st.session_state["goto"] = target


def nav_button(label, target, key, primary=False):
    """A button that takes the user to another page (same as using the sidebar)."""
    st.button(label, key=key, on_click=go, args=(target,),
              type="primary" if primary else "secondary")


def run_forensics(raw: bytes) -> dict:
    """Supporting signals; failures are reported, never fatal."""
    out = {}
    try:
        out["ela"] = run_ela(raw)
    except Exception as exc:
        log.error("ELA failed: %s", type(exc).__name__)
        out["ela_error"] = "Error Level Analysis could not be computed for this image."
    try:
        out["metadata"] = read_metadata(raw)
    except Exception as exc:
        log.error("Metadata read failed: %s", type(exc).__name__)
        out["metadata_error"] = "Metadata could not be read."
    return out


# ---------------------------------------------------------------- HOME
if page == "Home":
    st.title("Identity Guard AI")
    st.subheader("Find out if a photo may be misusing your face, and know what to do next")
    st.write("Two independent analyses are run on a suspected photo, combined into a transparent "
             "risk assessment, and packaged into an evidence report with clear next steps.")
    c1, c2, c3 = st.columns(3)
    with c1:
        card("1. Face similarity", "Compares FaceNet embeddings of the two photos with cosine "
             "similarity. Similarity is <b>not</b> proof of identity.")
    with c2:
        card("2. Deepfake detection", "A pretrained image classifier scores signs of manipulation "
             "in the suspected photo. A score is <b>not</b> proof an image is fake.")
    with c3:
        card("3. Risk assessment", "Combines both with configurable, <b>unvalidated</b> thresholds "
             "and explains the outcome in plain language.")
    c4, c5, c6 = st.columns(3)
    with c4:
        card("Supporting forensics", "Error Level Analysis heatmap and metadata checks as "
             "<b>weak, supporting hints</b>. They never change the risk category.")
    with c5:
        card("Evidence report", "PDF with a SHA-256 fingerprint of the file, tool versions and "
             "limitations. Photos are excluded unless you choose to include them.")
    with c6:
        card("Next steps", "Guidance on preserving evidence and reporting, with official "
             "contacts for India. General information, not legal advice.")
    if HOSTED:
        card("Hosted demo notice", "This copy runs on a server, so uploaded photos leave your device. "
             "They are held in memory only and not saved by the app. Use only photographs of people "
             "who have agreed, or AI-generated faces.")
    else:
        card("Privacy by design", "Photos are processed locally in memory, never permanently stored, "
             "and not sent to third-party services. Use only photographs you have permission to analyse.")
    c7, c8 = st.columns(2)
    with c7:
        card("Batch scan", "Check up to 5 suspected photos against one reference and get a "
             "ranked table, highest risk first.")
    with c8:
        card("Protect my photo", "Remove hidden location and device data from a photo before you "
             "share it, and add an optional visible watermark.")
    b1, b2, b3 = st.columns(3)
    with b1:
        nav_button("Start analysis", "Identity Analysis", "home_start", primary=True)
    with b2:
        nav_button("Batch scan", "Batch Scan", "home_batch")
    with b3:
        nav_button("Protect my photo", "Protect My Photo", "home_protect")

# ---------------------------------------------------------------- ANALYSIS
elif page == "Identity Analysis":
    st.title("Identity Analysis")
    if HOSTED:
        st.warning(HOSTED_NOTICE)
        prepare_hosted_model()
    if st.session_state.get("results"):
        nav_button("View latest results", "Results Dashboard", "view_latest")
    if not model_available():
        st.warning("The deepfake model is not downloaded yet. You can still run the analysis, "
                   "but the result will be **Inconclusive**. Run `python download_model.py` "
                   "once, then restart the app.")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Reference photograph")
        ref_up = st.file_uploader("Original photo (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"],
                                  key="ref_up", help=f"Max {MAX_FILE_SIZE_MB} MB")
    with right:
        st.markdown("#### Suspected photograph")
        sus_up = st.file_uploader("Photo that may be manipulated", type=["jpg", "jpeg", "png"],
                                  key="sus_up", help=f"Max {MAX_FILE_SIZE_MB} MB")

    ref_img, ref_err = read_upload(ref_up)
    sus_img, sus_err = read_upload(sus_up)
    ref_faces = detect_faces(ref_img) if ref_img is not None else []
    sus_faces = detect_faces(sus_img) if sus_img is not None else []

    for col, img, err, faces, label in ((left, ref_img, ref_err, ref_faces, "Reference"),
                                        (right, sus_img, sus_err, sus_faces, "Suspected")):
        with col:
            if err:
                st.error(f"{label}: {err}")
            elif img is not None:
                st.image(draw_face_boxes(img, faces), width="stretch")
                (st.success if faces else st.error)(f"{label}: {face_status(faces)}")

    consent = st.checkbox("I confirm I have permission to analyse these photographs.")
    ready = bool(ref_img is not None and sus_img is not None and ref_faces and sus_faces and consent)
    if ref_img is not None and sus_img is not None and not (ref_faces and sus_faces):
        st.warning("A face must be detected in both photos to continue.")

    if st.button("Analyse Identity", type="primary", disabled=not ready, width="stretch"):
        with st.spinner("Analysing locally... the first run can take a while (models load)."):
            core = analyse_pair(ref_img, sus_img)
            face, face_error = core["face"], core["face_error"]
            fake, risk = core["deepfake"], core["risk"]
            raw = sus_up.getvalue()          # exact bytes of the suspected file (kept in memory only)
            forensics = run_forensics(raw)
            evidence = {"sha256": sha256_hex(raw), "versions": library_versions(),
                        "models": {"face": "Facenet (DeepFace)", "deepfake": MODEL_ID}}
        st.session_state["results"] = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ref_status": face_status(ref_faces), "sus_status": face_status(sus_faces),
            "face": face, "face_error": face_error, "deepfake": fake, "risk": risk,
            "forensics": forensics, "evidence": evidence,
            "ref_img": draw_face_boxes(ref_img, ref_faces),
            "sus_img": draw_face_boxes(sus_img, sus_faces),
            "ref_crop": crop_largest_face(ref_img, ref_faces),
            "sus_crop": crop_largest_face(sus_img, sus_faces),
        }
        st.session_state["goto"] = "Results Dashboard"
        st.rerun()

# ---------------------------------------------------------------- RESULTS
elif page == "Results Dashboard":
    st.title("Results Dashboard")
    res = st.session_state.get("results")
    if not res:
        st.info("No analysis yet.")
        nav_button("Go to Identity Analysis", "Identity Analysis", "res_go_analysis", primary=True)
        st.stop()

    face, fake, risk = res["face"], res["deepfake"], res["risk"]
    badge = {"Low Risk": "b-low", "Moderate Risk": "b-mod",
             "High Risk": "b-high", "Inconclusive": "b-inc"}[risk["category"]]
    st.markdown(f'<span class="badge {badge}">{risk["category"]}</span>'
                f'<span class="privacy">Processed locally. Nothing stored.</span>', unsafe_allow_html=True)
    st.caption(f"Analysed {res['timestamp']}")

    c1, c2 = st.columns(2)
    c1.image(res["ref_img"], caption="Reference", width="stretch")
    c2.image(res["sus_img"], caption="Suspected", width="stretch")

    if res.get("ref_crop") is not None and res.get("sus_crop") is not None:
        st.markdown("##### Faces compared")
        f1, f2, _ = st.columns([1, 1, 2])
        f1.image(res["ref_crop"], caption="Reference face", width="stretch")
        f2.image(res["sus_crop"], caption="Suspected face", width="stretch")

    st.markdown("##### Signal strength")
    g1, g2 = st.columns(2)
    with g1:
        if face:
            gauge("Face similarity", face["similarity"], RISK_CONFIG["similarity_low"],
                  RISK_CONFIG["similarity_high"], similarity_signal(face["similarity"]).capitalize())
        else:
            metric("Face similarity", "Not computed", res.get("face_error") or "")
    with g2:
        if fake["status"] == "ok":
            gauge("Manipulation score", fake["fake_probability"], RISK_CONFIG["fake_low"],
                  RISK_CONFIG["fake_high"], manipulation_signal(fake["fake_probability"]).capitalize())
        else:
            metric("Manipulation score", "Unavailable", fake.get("message", ""))

    m1, m2, m3 = st.columns(3)
    with m1:
        metric("Face detection", f"{res['ref_status']}<br>{res['sus_status']}")
    with m2:
        metric("Face similarity (cosine)", f"{face['similarity']:.2f}" if face else "Not computed",
               similarity_signal(face["similarity"]) if face else "")
    with m3:
        if fake["status"] == "ok":
            metric("Deepfake detection", f"{fake['fake_probability']:.2f}",
                   f"{manipulation_signal(fake['fake_probability'])}. "
                   f"Model's own label: {fake['label']} (confidence {fake['confidence']:.2f})")
        else:
            metric("Deepfake detection", "Unavailable")

    tab_why, tab_for, tab_ev, tab_rep = st.tabs(
        ["Why this category", "Forensics (supporting)", "Evidence record", "Report"])

    with tab_why:
        for r in risk["reasons"]:
            st.write("- " + r)
        for c in risk["caveats"]:
            st.write("- Note: " + c)
        st.caption(risk["disclaimer"])
        st.caption("No model-attention heatmap (such as Grad-CAM) is produced in this version.")
        with st.expander("Thresholds used (editable in risk_assessment.py)"):
            st.json(RISK_CONFIG)

    with tab_for:
        st.info("These are **weak, supporting hints**. They do not change the risk category "
                "and never prove manipulation.")
        fx = res.get("forensics") or {}
        if "ela" in fx:
            ela = fx["ela"]
            st.markdown("##### Error Level Analysis (suspected photo)")
            st.image(ela["overlay"], caption="Red = larger recompression difference (relative to this image)",
                     width="stretch")
            st.write(f"Mean / 99th percentile / max error: **{ela['mean_error']} / "
                     f"{ela['p99_error']} / {ela['max_error']}**")
            for n in ela["notes"]:
                st.caption(n)
        elif "ela_error" in fx:
            st.warning(fx["ela_error"])
        if "metadata" in fx:
            meta = fx["metadata"]
            st.markdown("##### Metadata")
            st.write(f"Format: **{meta['format']}**  |  Camera: **{meta['camera'] or 'none recorded'}**  |  "
                     f"Software tag: **{meta['software'] or 'none'}**  |  "
                     f"GPS data: **{'yes' if meta['has_gps'] else 'no'}**")
            for n in meta["notes"]:
                st.caption(n)
        elif "metadata_error" in fx:
            st.warning(fx["metadata_error"])

    with tab_ev:
        ev = res.get("evidence") or {}
        st.write("SHA-256 fingerprint of the suspected image file (identifies this exact file):")
        st.code(ev.get("sha256", "n/a"), language=None)
        st.caption("Only the identical file produces this value. A re-save, crop or screenshot "
                   "changes it. The photo itself is not stored by this app.")
        st.write(f"Analysis time: **{res['timestamp']}**")
        st.write("Models: " + ", ".join(f"{k}: {v}" for k, v in ev.get("models", {}).items()))
        with st.expander("Tool versions"):
            st.json(ev.get("versions", {}))
        st.caption("Keep this with your screenshots and the post link if you decide to report the image.")

    with tab_rep:
        include = st.checkbox("Include the photographs and ELA overlay in the report "
                              "(off by default for privacy)")
        st.download_button("Download PDF report", data=generate_report(res, include_images=include),
                           file_name="identity_guard_report.pdf", mime="application/pdf")

    st.info("Worried about what you found? See what to do next.")
    r1, r2, r3 = st.columns(3)
    with r1:
        nav_button("Open Next Steps", "Next Steps", "res_next", primary=True)
    with r2:
        nav_button("New analysis", "Identity Analysis", "res_new")
    with r3:
        nav_button("Batch scan", "Batch Scan", "res_batch")

# ---------------------------------------------------------------- NEXT STEPS
elif page == "Next Steps":
    st.title("Next Steps")
    st.write(f"If a photo is being misused, here is what to do. Resources shown are for "
             f"**{GUIDANCE['country']}** (details checked {GUIDANCE['last_checked']}).")
    for i, (head, text) in enumerate(GUIDANCE["steps"], 1):
        card(f"{i}. {head}", text)
    st.markdown("#### Where to get help")
    for r in GUIDANCE["resources"]:
        link = f'<br><a href="{r["url"]}" target="_blank">{r["url"]}</a>' if r["url"] else ""
        card(r["name"], f'{r["detail"]}{link}')
    st.warning(GUIDANCE["minor_note"])
    st.caption(GUIDANCE["disclaimer"] + " To support another country, edit guidance.py.")
    n1, n2 = st.columns(2)
    with n1:
        nav_button("Back to results", "Results Dashboard", "ns_results")
    with n2:
        nav_button("New analysis", "Identity Analysis", "ns_new")

# ---------------------------------------------------------------- BATCH SCAN
elif page == "Batch Scan":
    st.title("Batch Scan")
    if HOSTED:
        st.warning(HOSTED_NOTICE)
        prepare_hosted_model()
    st.write(f"Compare up to {MAX_BATCH} suspected photos with one reference photo. Results are "
             "ranked with the highest risk first. Each photo runs both AI models, so allow a few "
             "seconds per image on a normal laptop.")
    bl, br = st.columns(2)
    with bl:
        st.markdown("#### Reference photograph")
        b_ref = st.file_uploader("Original photo (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], key="b_ref")
    with br:
        st.markdown(f"#### Suspected photographs (up to {MAX_BATCH})")
        b_sus = st.file_uploader("Photos that may be manipulated", type=["jpg", "jpeg", "png"],
                                 accept_multiple_files=True, key="b_sus")
    ref_img, ref_err = read_upload(b_ref)
    ref_faces = detect_faces(ref_img) if ref_img is not None else []
    if ref_err:
        st.error(f"Reference: {ref_err}")
    elif ref_img is not None:
        (st.success if ref_faces else st.error)(f"Reference: {face_status(ref_faces)}")
    if len(b_sus) > MAX_BATCH:
        st.warning(f"Only the first {MAX_BATCH} suspected photos will be scanned.")
        b_sus = b_sus[:MAX_BATCH]
    consent = st.checkbox("I confirm I have permission to analyse these photographs.", key="b_consent")
    ready = bool(ref_img is not None and ref_faces and b_sus and consent)

    if st.button("Scan all", type="primary", disabled=not ready, width="stretch"):
        rows, thumbs = [], []
        bar = st.progress(0.0, text="Scanning...")
        for i, up in enumerate(b_sus, 1):
            img, err = read_upload(up)
            row = {"Image": f"Image {i}", "Faces": "-", "Similarity": "n/a",
                   "Manipulation score": "n/a", "Risk": "", "_rank": 4}
            if err:
                row["Risk"] = f"Skipped: {err}"
            else:
                faces = detect_faces(img)
                row["Faces"] = face_status(faces)
                if not faces:
                    row["Risk"] = "Skipped: no face detected"
                else:
                    res = analyse_pair(ref_img, img)
                    if res["face"]:
                        row["Similarity"] = f"{res['face']['similarity']:.2f}"
                    if res["deepfake"]["status"] == "ok":
                        row["Manipulation score"] = f"{res['deepfake']['fake_probability']:.2f}"
                    row["Risk"] = res["risk"]["category"]
                    row["_rank"] = RISK_RANK[row["Risk"]]
                    small = Image.fromarray(img)
                    small.thumbnail((220, 220))
                    thumbs.append((row["Image"], row["Risk"], small))
            rows.append(row)
            bar.progress(i / len(b_sus), text=f"Scanned {i} of {len(b_sus)}")
        bar.empty()
        rows.sort(key=lambda r: r["_rank"])
        st.session_state["batch"] = {"rows": rows, "thumbs": thumbs,
                                     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    batch = st.session_state.get("batch")
    if batch:
        st.markdown(f"#### Ranked results ({batch['timestamp']})")
        shown = [{k: v for k, v in r.items() if k != "_rank"} for r in batch["rows"]]
        st.dataframe(shown, width="stretch", hide_index=True)
        order = {r["Image"]: n for n, r in enumerate(batch["rows"])}
        thumbs = sorted(batch["thumbs"], key=lambda t: order[t[0]])
        if thumbs:
            cols = st.columns(len(thumbs))
            for col, (name, risk, small) in zip(cols, thumbs):
                col.image(small, caption=f"{name}: {risk}", width="stretch")
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["Image", "Faces", "Similarity", "Manipulation score", "Risk"])
        for r in shown:
            writer.writerow([r["Image"], r["Faces"], r["Similarity"], r["Manipulation score"], r["Risk"]])
        st.download_button("Download summary (CSV)", out.getvalue(), file_name="batch_summary.csv",
                           mime="text/csv")
        st.caption("Images are numbered in upload order and file names are not recorded. "
                   "A category is a decision-support signal, not proof. Thresholds are "
                   "illustrative and not scientifically validated.")
        nav_button("Open Next Steps", "Next Steps", "batch_next")

# ---------------------------------------------------------------- PROTECT MY PHOTO
elif page == "Protect My Photo":
    st.title("Protect My Photo")
    if HOSTED:
        st.warning(HOSTED_NOTICE)
    st.write("Before you share a photo online, remove the hidden data inside it (such as GPS "
             "location and camera details) and optionally add a visible watermark.")
    st.info("**Honest limits:** this does not stop anyone from making a deepfake. It removes hidden "
            "location and device data, and a watermark can discourage casual reuse, but a "
            "watermark can be cropped out.")
    p_up = st.file_uploader("Your photo (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], key="p_up",
                            help=f"Max {MAX_FILE_SIZE_MB} MB")
    if p_up is not None:
        raw = p_up.getvalue()
        try:
            load_image(raw, p_up.name)                 # same validation as every other upload
            before = read_metadata(raw)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Hidden data found")
                st.write(f"Metadata present: **{'yes' if before['has_exif'] else 'no'}**")
                st.write(f"Camera: **{before['camera'] or 'none recorded'}**")
                st.write(f"Editing software tag: **{before['software'] or 'none'}**")
                st.write(f"Location (GPS) data: **{'YES' if before['has_gps'] else 'no'}**")
            with c2:
                st.markdown("##### Options")
                use_mark = st.checkbox("Add a visible watermark", value=True)
                text = st.text_input("Watermark text", value="Do not use without permission",
                                     max_chars=MAX_WATERMARK_CHARS, disabled=not use_mark)
                strength = st.slider("Watermark strength", 10, 80, 35, disabled=not use_mark)
            preview, out_bytes, fmt = protect_photo(raw, text if use_mark else "", strength / 100)
            after = read_metadata(out_bytes)
            st.image(preview, caption="Protected copy (preview)", width="stretch")
            if after["has_exif"] or after["has_gps"]:
                st.error("Metadata is still present in the output. Do not share this copy.")
            else:
                st.success("Verified: the protected copy contains no metadata or location data.")
            ext = "png" if fmt == "PNG" else "jpg"
            st.download_button("Download protected photo", out_bytes,
                               file_name=f"protected_photo.{ext}", mime=f"image/{'png' if fmt == 'PNG' else 'jpeg'}")
            st.caption("Your original file is not changed or stored. Re-encoding a JPEG can slightly "
                       "reduce quality.")
        except ImageValidationError as e:
            st.error(str(e))
        except Exception as exc:
            log.error("Protect failed: %s", type(exc).__name__)
            st.error("This photo could not be processed.")

# ---------------------------------------------------------------- ABOUT
else:
    st.title("About the Project")
    st.write("Identity Guard AI is an academic prototype that combines two independent image "
             "analyses to flag possible misuse of a person's facial identity, and helps the "
             "person act on the result.")
    card("Implemented", "Image upload and validation, face detection (OpenCV), face similarity "
         "(FaceNet via DeepFace + cosine), deepfake classification (pretrained ViT, "
         f"{MODEL_ID}), configurable risk assessment, Error Level Analysis and metadata checks, "
         "SHA-256 evidence record, PDF report, next-steps guidance, batch scan of up to 5 photos, "
         "and a Protect My Photo tool (metadata removal and watermark).")
    card("Not implemented", "Video/audio detection, model-attention heatmaps (Grad-CAM), model training.")
    card("Limitations", "Deepfake classifiers generalise poorly to unseen manipulation methods and "
         "low-quality images. Face similarity depends on pose, lighting and age. ELA and metadata "
         "are weak hints. Thresholds are illustrative. Never treat any result as proof.")
