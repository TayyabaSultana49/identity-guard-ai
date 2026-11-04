"""protect.py - 'Protect My Photo': remove hidden metadata and add a visible watermark.

HONEST LIMITS: this does NOT stop anyone from making a deepfake of a photo. It removes
hidden location/device data before you share, and a visible watermark can deter casual
reuse, but a watermark can be cropped out."""
import io

from PIL import Image, ImageDraw, ImageFont, ImageOps

MAX_WATERMARK_CHARS = 60


def clean_copy(file_bytes: bytes):
    """Return (clean PIL image with NO metadata, original format 'JPEG' or 'PNG')."""
    with Image.open(io.BytesIO(file_bytes)) as img:
        fmt = img.format
        img = ImageOps.exif_transpose(img)
        mode = "RGBA" if (fmt == "PNG" and img.mode in ("RGBA", "LA")) else "RGB"
        img = img.convert(mode)
        # Rebuild from raw pixels so no EXIF, GPS or ICC data can be carried over
        return Image.frombytes(mode, img.size, img.tobytes()), fmt


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                      # very old Pillow
        return ImageFont.load_default()


def add_watermark(img: Image.Image, text: str, opacity: float = 0.35) -> Image.Image:
    """Tile diagonal text over the image. Returns the image unchanged if text is empty."""
    text = (text or "").strip()[:MAX_WATERMARK_CHARS]
    if not text:
        return img
    base = img.convert("RGBA")
    w, h = base.size
    side = int((w * w + h * h) ** 0.5) + 10
    layer = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = _font(max(14, w // 22))
    box = draw.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    alpha = int(255 * max(0.05, min(0.9, opacity)))
    step_x, step_y = int(tw * 1.5), th * 4
    row = 0
    for y in range(0, side, step_y):
        offset = (row % 2) * step_x // 2
        for x in range(-tw, side, step_x):
            draw.text((x + offset, y), text, font=font, fill=(255, 255, 255, alpha),
                      stroke_width=1, stroke_fill=(0, 0, 0, alpha))
        row += 1
    layer = layer.rotate(30, resample=Image.BICUBIC)
    left, top = (side - w) // 2, (side - h) // 2
    layer = layer.crop((left, top, left + w, top + h))
    out = Image.alpha_composite(base, layer)
    return out if img.mode == "RGBA" else out.convert("RGB")


def encode(img: Image.Image, fmt: str) -> bytes:
    """Encode to JPEG or PNG bytes (no metadata is written)."""
    buf = io.BytesIO()
    if fmt == "PNG":
        img.save(buf, format="PNG")
    else:
        img.convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def protect_photo(file_bytes: bytes, watermark_text: str = "", opacity: float = 0.35):
    """Full pipeline. Returns (preview PIL image, output bytes, format)."""
    clean, fmt = clean_copy(file_bytes)
    marked = add_watermark(clean, watermark_text, opacity)
    return marked, encode(marked, fmt), fmt
