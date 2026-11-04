"""Manual end-to-end check with real images (needs deepface installed).
Run from the project root:  python tests/manual_check.py ref.jpg suspected.jpg"""
import sys

sys.path.insert(0, ".")
from deepfake_detector import detect_deepfake
from face_matcher import FaceMatchError, compare_faces
from image_processor import ImageValidationError, detect_faces, face_status, load_image
from risk_assessment import assess_risk


def load(path):
    with open(path, "rb") as f:
        return load_image(f.read(), path)


if len(sys.argv) != 3:
    sys.exit("Usage: python tests/manual_check.py <reference> <suspected>")

try:
    ref, sus = load(sys.argv[1]), load(sys.argv[2])
    print("Reference:", face_status(detect_faces(ref)))
    print("Suspected:", face_status(detect_faces(sus)))
    try:
        face = compare_faces(ref, sus)
    except FaceMatchError as e:
        face = None
        print("Face match error:", e)
    fake = detect_deepfake(sus)
    print("FACE:", face)
    print("DEEPFAKE:", fake)
    print("RISK:", assess_risk(face, fake))
except ImageValidationError as e:
    print("Handled error:", e)
