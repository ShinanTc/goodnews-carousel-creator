"""
create_carousel.py
------------------
Generates Instagram-ready 1080x1080 PNG carousel slides.

Each slide:
  - Full-bleed background photo from Pexels (free API key required)
  - Black linear gradient: transparent at top → opaque black at bottom (25% of height)
  - Headline text in League Spartan Bold, ALL CAPS, white, bottom-left aligned
  - Soft drop shadow under text for readability

Dependencies:
    pip install Pillow requests python-dotenv

Setup:
    1. Get a free Pexels API key at https://www.pexels.com/api/
    2. Add PEXELS_API_KEY=your_key_here to your .env file
    3. League Spartan Bold is downloaded automatically on first run.
       (Source: https://fonts.google.com/specimen/League+Spartan)
"""

import os
import re
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

load_dotenv()

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")

# League Spartan Bold — downloaded automatically and cached next to this script
# Use __file__ if available, fall back to cwd (safe on all platforms/environments)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
FONT_CACHE_PATH = os.path.join(_SCRIPT_DIR, "LeagueSpartan-VariableFont_wght.ttf")
FONT_URLS = [
    "https://cdn.jsdelivr.net/gh/theleagueof/league-spartan@master/static/LeagueSpartan-Bold.ttf",
    "https://github.com/theleagueof/league-spartan/raw/master/static/LeagueSpartan-Bold.ttf",
]

SLIDE_PX = 1080          # square canvas (Instagram 1:1)
GRADIENT_FRACTION = 0.25  # bottom 25% of the slide


# ─────────────────────────────────────────────────────────────────────────────
# Font
# ─────────────────────────────────────────────────────────────────────────────

def ensure_font():
    """Download League Spartan Bold if not already cached locally."""
    if os.path.exists(FONT_CACHE_PATH):
        return FONT_CACHE_PATH

    print("  Downloading League Spartan Bold font...")
    for url in FONT_URLS:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 10_000:
                with open(FONT_CACHE_PATH, "wb") as f:
                    f.write(r.content)
                print("  ✓ Font cached to:", FONT_CACHE_PATH)
                return FONT_CACHE_PATH
        except Exception:
            continue

    raise RuntimeError(
        "Could not download League Spartan font automatically.\n"
        "Please download LeagueSpartan-Bold.ttf from "
        "https://fonts.google.com/specimen/League+Spartan "
        "and place it in the same folder as this script."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Image fetching
# ─────────────────────────────────────────────────────────────────────────────

_STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "and", "or",
    "but", "is", "are", "was", "were", "has", "have", "had", "be", "been",
    "its", "their", "this", "that", "with", "from", "by", "as", "new",
    "now", "after", "over", "up", "down", "first", "last", "more", "most",
    "than", "into", "about", "will", "record", "world", "largest", "says",
}

def extract_keywords(headline):
    """Pull 3–4 content-rich words from a headline for image search."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", headline)
    keywords = [w for w in words if w.lower() not in _STOP_WORDS]
    return " ".join(keywords[:4]) if keywords else headline[:40]


def fetch_pexels_image(keywords):
    """
    Returns raw image bytes from Pexels, or None on failure.
    Requires PEXELS_API_KEY in .env (free at https://www.pexels.com/api/).
    """
    if not PEXELS_API_KEY:
        print("  ⚠  PEXELS_API_KEY not set — using fallback background.")
        return None

    headers = {"Authorization": PEXELS_API_KEY}
    query = requests.utils.quote(keywords)
    url = (
        f"https://api.pexels.com/v1/search"
        f"?query={query}&per_page=5&orientation=square&size=large"
    )

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        photos = resp.json().get("photos", [])
        if not photos:
            print(f"  ⚠  No Pexels results for '{keywords}' — using fallback.")
            return None
        img_url = photos[0]["src"]["large2x"]
        img_resp = requests.get(img_url, timeout=30)
        img_resp.raise_for_status()
        return img_resp.content

    except Exception as e:
        print(f"  ⚠  Image fetch failed ({e}) — using fallback.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Gradient
# ─────────────────────────────────────────────────────────────────────────────

def make_gradient_mask(width, height):
    """
    Returns a grayscale (L-mode) mask image.
    Top row = 0 (fully transparent), bottom row = 255 (fully opaque).
    Used as the paste mask for the black gradient overlay.
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for y in range(height):
        value = int(255 * y / max(height - 1, 1))
        draw.line([(0, y), (width - 1, y)], fill=value)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────

def wrap_text(text, font, max_width, draw):
    """Word-wrap text to fit within max_width pixels. Returns list of lines."""
    words = text.split()
    lines, current = [], []

    for word in words:
        candidate = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]

    if current:
        lines.append(" ".join(current))

    return lines


def pick_font_size(char_count):
    """Scale font size down gracefully for longer headlines."""
    if char_count <= 40:
        return 64
    elif char_count <= 55:
        return 56
    elif char_count <= 70:
        return 48
    else:
        return 42


# ─────────────────────────────────────────────────────────────────────────────
# Slide renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_slide(img_bytes, headline, font_path):
    """
    Composites background + gradient + text into a SLIDE_PX × SLIDE_PX RGB image.
    Returns a PIL Image.
    """
    size = SLIDE_PX

    # ── Background ────────────────────────────────────────────────────────────
    if img_bytes:
        bg = Image.open(BytesIO(img_bytes)).convert("RGB")
        w, h = bg.size
        m = min(w, h)
        bg = bg.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2))
        bg = bg.resize((size, size), Image.LANCZOS)
    else:
        # Earthy dark-green fallback
        bg = Image.new("RGB", (size, size), (26, 58, 42))

    # ── Black gradient overlay (bottom 25%) ──────────────────────────────────
    grad_h = int(size * GRADIENT_FRACTION)   # 270 px
    mask = make_gradient_mask(size, grad_h)
    black_band = Image.new("RGB", (size, grad_h), (0, 0, 0))
    bg.paste(black_band, (0, size - grad_h), mask=mask)

    # ── Text ──────────────────────────────────────────────────────────────────
    draw = ImageDraw.Draw(bg)

    text = headline.upper()
    font_size = pick_font_size(len(text))
    font = ImageFont.truetype(font_path, font_size)

    side_margin = 50
    bottom_margin = 52
    max_text_w = size - side_margin * 2
    lines = wrap_text(text, font, max_text_w, draw)

    line_gap = int(font_size * 0.22)
    line_h = font_size + line_gap
    total_text_h = len(lines) * line_h - line_gap

    # Pin text to bottom, staying within gradient band
    y = max(
        size - bottom_margin - total_text_h,
        size - grad_h + 10          # never higher than gradient start
    )

    for line in lines:
        # Soft drop shadow (dark, offset 3 px down-right)
        draw.text((side_margin + 3, y + 3), line, font=font, fill=(0, 0, 0))
        # Main white text
        draw.text((side_margin, y), line, font=font, fill=(255, 255, 255))
        y += line_h

    return bg


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def create_carousel(headlines, output_dir="carousel_slides"):
    """
    Generates one PNG slide per headline and saves them to output_dir.
    Returns list of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    font_path = ensure_font()

    print(f"\nGenerating {len(headlines)} carousel slides → {os.path.abspath(output_dir)}/")

    paths = []
    for i, headline in enumerate(headlines, 1):
        print(f"  [{i:02d}/{len(headlines):02d}] {headline[:65]}{'...' if len(headline) > 65 else ''}")

        keywords = extract_keywords(headline)
        img_bytes = fetch_pexels_image(keywords)

        slide = render_slide(img_bytes, headline, font_path)

        out_path = os.path.join(output_dir, f"slide_{i:02d}.png")
        slide.save(out_path, "PNG", optimize=True)
        paths.append(out_path)

    print(f"\n✅ Done — {len(paths)} slides saved.")
    return paths