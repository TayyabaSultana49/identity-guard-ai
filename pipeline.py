"""pipeline.py - runs the two independent analyses and the risk assessment on one pair.

Used by both the single analysis page and the batch scan, so they always agree."""
from deepfake_detector import detect_deepfake
from face_matcher import FaceMatchError, compare_faces
from risk_assessment import assess_risk


def analyse_pair(reference_rgb, suspected_rgb, config=None) -> dict:
    """Return {face, face_error, deepfake, risk} for one reference/suspected pair."""
    face, face_error = None, None
    try:
        face = compare_faces(reference_rgb, suspected_rgb)
    except FaceMatchError as e:
        face_error = str(e)
    fake = detect_deepfake(suspected_rgb)
    risk = assess_risk(face, fake, config)
    return {"face": face, "face_error": face_error, "deepfake": fake, "risk": risk}
