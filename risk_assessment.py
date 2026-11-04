"""risk_assessment.py - combines the two independent analyses.

IMPORTANT: thresholds are illustrative defaults, NOT scientifically validated.
Adjust RISK_CONFIG after testing with your own permitted images."""

RISK_CONFIG = {
    "similarity_high": 0.70,   # at/above: faces look visually similar
    "similarity_low": 0.40,    # below: faces do not look similar
    "fake_high": 0.80,         # manipulation score at/above: strong signal
    "fake_low": 0.30,          # below: weak signal
}

DISCLAIMER = (
    "This assessment is a decision-support aid, not proof. Face similarity does "
    "not establish identity, and a deepfake model's output does not prove an image "
    "is fake. The thresholds are illustrative and not scientifically validated."
)


def _level(value, low, high):
    if value >= high:
        return "high"
    if value < low:
        return "low"
    return "mid"


def assess_risk(face_result, deepfake_result, config=None) -> dict:
    """Return {category, reasons, caveats, thresholds, disclaimer}."""
    cfg = config or RISK_CONFIG
    reasons, caveats = [], []

    def done(category):
        return {"category": category, "reasons": reasons, "caveats": caveats,
                "thresholds": dict(cfg), "disclaimer": DISCLAIMER}

    # Missing inputs -> Inconclusive (never guess)
    if face_result is None:
        reasons.append("Face similarity could not be computed.")
        return done("Inconclusive")
    if deepfake_result.get("status") != "ok":
        reasons.append("Deepfake detection was unavailable, so manipulation could not be assessed. "
                       + deepfake_result.get("message", ""))
        return done("Inconclusive")

    sim = face_result["similarity"]
    fake = deepfake_result["fake_probability"]
    s_lvl = _level(sim, cfg["similarity_low"], cfg["similarity_high"])
    f_lvl = _level(fake, cfg["fake_low"], cfg["fake_high"])

    reasons.append(f"Face similarity {sim:.2f} is in the '{s_lvl}' band.")
    reasons.append(f"Manipulation score {fake:.2f} is in the '{f_lvl}' band.")

    if face_result.get("faces_suspected", 1) > 1:
        caveats.append("The suspected image has several faces; only the largest was compared.")
    if face_result.get("faces_reference", 1) > 1:
        caveats.append("The reference image has several faces; only the largest was used.")

    # Decision table
    if s_lvl == "low":
        category = "Low Risk"
        reasons.append("The suspected face does not look similar to the reference.")
        if f_lvl == "high":
            caveats.append("The image may be manipulated, but it does not resemble the reference face.")
    elif s_lvl == "high":
        category = {"high": "High Risk", "mid": "Moderate Risk", "low": "Low Risk"}[f_lvl]
        if f_lvl == "low":
            reasons.append("Similar face but no strong manipulation signal; may be a genuine photo.")
    else:  # similarity in the middle band
        category = "Moderate Risk" if f_lvl == "high" else "Inconclusive"
        if category == "Inconclusive":
            reasons.append("Similarity is ambiguous and manipulation evidence is not strong.")
    return done(category)


def manipulation_signal(score, config=None) -> str:
    """Plain-language description of a manipulation score (uses the same bands)."""
    cfg = config or RISK_CONFIG
    return {"low": "weak signal", "mid": "moderate signal, not conclusive",
            "high": "strong signal"}[_level(score, cfg["fake_low"], cfg["fake_high"])]


def similarity_signal(sim, config=None) -> str:
    """Plain-language description of a face similarity score."""
    cfg = config or RISK_CONFIG
    return {"low": "faces do not look similar", "mid": "ambiguous similarity",
            "high": "faces look very similar"}[_level(sim, cfg["similarity_low"], cfg["similarity_high"])]
