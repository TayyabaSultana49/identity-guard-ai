"""report_generator.py - builds the PDF report in memory with ReportLab.

The report never contains filenames. Photographs (and the forensic heatmap, which is
derived from the suspected photo) are included ONLY if the user ticks the option."""
import io
from xml.sax.saxutils import escape

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from guidance import GUIDANCE

PROJECT_NAME = "Identity Guard AI"
NAVY = colors.HexColor("#0b1f3a")
CYAN = colors.HexColor("#0891b2")


def _p(text, style):
    """Paragraph with user/model text safely escaped."""
    return Paragraph(escape(str(text)), style)


def _rl_image(img_rgb, max_w_cm=7.5):
    """Convert a numpy RGB image to a ReportLab image (in memory)."""
    pil = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    w, h = pil.size
    width = max_w_cm * cm
    return RLImage(buf, width=width, height=width * h / w)


def _table(rows, body, header=("Item", "Result"), widths=(6.5 * cm, 10.5 * cm)):
    """Two-column table with a navy header row."""
    cells = [[Paragraph(f'<font color="white"><b>{escape(header[0])}</b></font>', body),
              Paragraph(f'<font color="white"><b>{escape(header[1])}</b></font>', body)]]
    cells += [[_p(a, body), _p(b, body)] for a, b in rows]
    t = Table(cells, colWidths=list(widths))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9aa7b5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def generate_report(results: dict, include_images: bool = False) -> bytes:
    """Return the PDF as bytes. `results` comes from app.py's analysis step."""
    base = getSampleStyleSheet()
    title = ParagraphStyle("T", parent=base["Title"], textColor=NAVY, fontSize=22)
    h2 = ParagraphStyle("H2", parent=base["Heading2"], textColor=CYAN, spaceBefore=12)
    body = ParagraphStyle("B", parent=base["BodyText"], fontSize=10, leading=14)
    small = ParagraphStyle("S", parent=body, fontSize=8.5, textColor=colors.HexColor("#444444"))
    mono = ParagraphStyle("M", parent=body, fontName="Courier", fontSize=9, leading=12)

    face = results.get("face")
    fake = results.get("deepfake", {})
    risk = results.get("risk", {})
    forensics = results.get("forensics") or {}
    evidence = results.get("evidence") or {}

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title=f"{PROJECT_NAME} Analysis Report")
    s = []
    s.append(_p(PROJECT_NAME, title))
    s.append(_p("AI-Powered Deepfake Identity Misuse Detection: Analysis Report", base["Heading3"]))
    s.append(_p(f"Analysis date and time: {results.get('timestamp', 'n/a')}", body))

    # Overall result
    s.append(_p("Risk assessment", h2))
    s.append(_p(f"Category: {risk.get('category', 'n/a')}", base["Heading3"]))
    for r in risk.get("reasons", []):
        s.append(_p(f"- {r}", body))
    for c in risk.get("caveats", []):
        s.append(_p(f"- Note: {c}", body))

    # Results table
    s.append(_p("Analysis results", h2))
    rows = [["Reference image face detection", results.get("ref_status", "n/a")],
            ["Suspected image face detection", results.get("sus_status", "n/a")]]
    if face:
        rows += [["Face similarity (cosine)", f"{face['similarity']:.4f}"],
                 ["Face model", face["model"]]]
    else:
        rows.append(["Face similarity", results.get("face_error") or "Not computed"])
    if fake.get("status") == "ok":
        rows += [["Deepfake prediction (model label)", fake["label"]],
                 ["Prediction confidence", f"{fake['confidence']:.4f}"],
                 ["Manipulation score (fake probability)", f"{fake['fake_probability']:.4f}"],
                 ["Deepfake model", fake["model"]],
                 ["Model input", fake.get("input_used", "n/a")]]
    else:
        rows.append(["Deepfake detection", fake.get("message", "Unavailable")])
    s.append(_table(rows, body))

    # Supporting forensics (weak signals; not part of the risk category)
    ela, meta = forensics.get("ela"), forensics.get("metadata")
    if ela or meta:
        s.append(_p("Supporting forensics (weak signals, not used for the risk category)", h2))
        frows = []
        if ela:
            frows += [["ELA mean / 99th percentile / max error",
                       f"{ela['mean_error']} / {ela['p99_error']} / {ela['max_error']}"],
                      ["ELA note", ela["notes"][0]]]
        if meta:
            frows += [["Image format", meta.get("format") or "n/a"],
                      ["Camera make/model in metadata", meta.get("camera") or "none recorded"],
                      ["Software tag", meta.get("software") or "none"],
                      ["Contains GPS data", "yes" if meta.get("has_gps") else "no"]]
            for n in meta.get("notes", []):
                frows.append(["Metadata note", n])
        s.append(_table(frows, body))
        if include_images and ela:
            s.append(Spacer(1, 6))
            s.append(_rl_image(ela["overlay"], 9))
            s.append(_p("ELA overlay (relative colours; a hint only, never proof).", small))

    # Optional images
    if include_images and results.get("ref_img") is not None and results.get("sus_img") is not None:
        s.append(_p("Photographs (included at the user's request)", h2))
        s.append(Table([[_rl_image(results["ref_img"]), _rl_image(results["sus_img"])],
                        [_p("Reference", small), _p("Suspected", small)]]))

    # Evidence record
    if evidence:
        s.append(_p("Evidence record", h2))
        sha = evidence.get("sha256", "")
        s.append(_p("SHA-256 of the suspected image file (identifies this exact file):", body))
        s.append(Paragraph(f"{escape(sha[:32])}<br/>{escape(sha[32:])}", mono))
        s.append(Spacer(1, 4))
        vers = evidence.get("versions", {})
        s.append(_p("Tools used: " + ", ".join(f"{k} {v}" for k, v in vers.items()), small))
        s.append(_p("The hash only matches the identical file. Any re-save, crop or screenshot "
                    "produces a different hash. The photo itself is not stored by this tool.", small))

    # Explanation and limitations
    s.append(_p("Explanation and limitations", h2))
    s.append(_p(risk.get("disclaimer", ""), body))
    if fake.get("limitations"):
        s.append(_p(fake["limitations"], body))
    s.append(_p("No model-attention heatmap (such as Grad-CAM) was generated in this version.", body))
    th = risk.get("thresholds", {})
    if th:
        s.append(_p("Thresholds used (illustrative, not scientifically validated): "
                    + ", ".join(f"{k}={v}" for k, v in th.items()), small))

    # What to do next
    s.append(_p(f"If you are affected: next steps ({GUIDANCE['country']})", h2))
    for i, (head, text) in enumerate(GUIDANCE["steps"], 1):
        s.append(_p(f"{i}. {head}: {text}", body))
    for r in GUIDANCE["resources"]:
        link = f" ({r['url']})" if r["url"] else ""
        s.append(_p(f"- {r['name']}{link}: {r['detail']}", small))
    s.append(_p(GUIDANCE["minor_note"], small))
    s.append(_p(f"{GUIDANCE['disclaimer']} Details checked {GUIDANCE['last_checked']}.", small))

    s.append(Spacer(1, 10))
    s.append(_p("Generated locally by Identity Guard AI. Uploaded photographs were processed in "
                "memory and are not stored.", small))
    doc.build(s)
    return buf.getvalue()
