"""guidance.py - 'What to do next' content shown in the app and the PDF report.

Resources below are for INDIA. They were checked against official/public sources
on 2026-11-04, but phone numbers and websites can change: re-verify before you
present, and edit this file to support another country.
This is general information, not legal advice."""

GUIDANCE = {
    "country": "India",
    "last_checked": "2026-11-04",
    "steps": [
        ("You are not at fault",
         "Someone misusing a picture of you is not your mistake. Take a breath before acting."),
        ("Preserve evidence first",
         "Do not delete anything yet. Save the original file, and screenshot the post showing the "
         "link (URL), the username, and the date and time. Keep the SHA-256 value from this "
         "report, which identifies the exact file you analysed."),
        ("Do not forward or pay",
         "Do not share the image again, even to prove it is fake. If someone demands money or "
         "more images, do not pay and do not reply; keep the messages as evidence."),
        ("Report it on the platform",
         "Use the platform's report option for impersonation, harassment or non-consensual "
         "intimate imagery so the content can be taken down."),
        ("Report it to the authorities",
         "File a complaint on the National Cyber Crime Reporting Portal, and contact local police "
         "if you feel unsafe."),
        ("Tell someone you trust",
         "Share this with a friend, family member or counsellor, and consider legal advice from a "
         "qualified lawyer. This tool cannot give legal advice."),
    ],
    "resources": [
        {"name": "National Cyber Crime Reporting Portal",
         "url": "https://cybercrime.gov.in",
         "detail": "Government of India portal to report cybercrime online, including social media "
                   "offences. Free to use."},
        {"name": "Cyber crime helpline 1930",
         "url": "",
         "detail": "National cybercrime helpline, available 24x7. It is mainly for financial fraud; "
                   "use the portal for other offences."},
        {"name": "Emergency police 112 / Women helpline 181",
         "url": "",
         "detail": "For immediate danger, call 112. 181 is the national women helpline."},
        {"name": "StopNCII.org",
         "url": "https://stopncii.org",
         "detail": "For intimate images (adults 18+): creates a fingerprint (hash) on your own device, "
                   "so the image is not uploaded, and shares it with participating platforms to help "
                   "block matching content. It only covers public platforms of participating "
                   "companies, not encrypted services."},
    ],
    "minor_note": ("If the image involves anyone under 18, report it immediately and do not copy, "
                   "forward or save it beyond what officials require."),
    "disclaimer": "General information only, not legal advice. Re-check contact details before relying on them.",
}
