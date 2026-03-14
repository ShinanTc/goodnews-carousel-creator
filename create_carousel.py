import os
import re
import requests
import json
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
FONT_CACHE_PATH = os.path.join(SCRIPT_DIR, "LeagueSpartan-VariableFont_wght.ttf")

FONT_URLS = [
    "https://cdn.jsdelivr.net/gh/theleagueof/league-spartan@master/static/LeagueSpartan-Bold.ttf",
    "https://github.com/theleagueof/league-spartan/raw/master/static/LeagueSpartan-Bold.ttf",
]

SLIDE_PX = 1080
GRADIENT_FRACTION = 0.25


# ─────────────────────────────────────────────
# FONT
# ─────────────────────────────────────────────

def ensure_font():
    if os.path.exists(FONT_CACHE_PATH):
        return FONT_CACHE_PATH

    for url in FONT_URLS:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(FONT_CACHE_PATH, "wb") as f:
                    f.write(r.content)
                return FONT_CACHE_PATH
        except:
            pass

    raise RuntimeError("Font download failed.")


# ─────────────────────────────────────────────
# KEYWORD EXTRACTION
# ─────────────────────────────────────────────

STOP_WORDS = {
    "a","an","the","in","on","at","to","for","of","and","or",
    "but","is","are","was","were","has","have","had","be",
    "been","this","that","with","from","by","as","new",
    "after","over","first","last","more","most","than"
}

def extract_keywords(headline):
    words = re.findall(r"\b[a-zA-Z]{3,}\b", headline)
    keywords = [w for w in words if w.lower() not in STOP_WORDS]
    return " ".join(keywords[:4]) if keywords else headline


# ─────────────────────────────────────────────
# LLM IMAGE SCORING
# ─────────────────────────────────────────────

def choose_best_image_with_llm(headline, photos):

    alts = []
    for i, p in enumerate(photos):
        alt = p.get("alt", "")
        alts.append(f"{i}: {alt}")

    alt_block = "\n".join(alts)

    prompt = f"""
You are selecting the most relevant image for a news headline.

Headline:
{headline}

Images:
{alt_block}

Score each image from 0 to 10 based on relevance to the headline.

Respond ONLY with JSON in this format:

{{
"0": score,
"1": score,
"2": score,
"3": score,
"4": score
}}

No explanations.
"""

    try:

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        response = completion.choices[0].message.content.strip()

        scores = json.loads(response)

        best_index = max(scores, key=lambda k: scores[k])

        return photos[int(best_index)]

    except Exception as e:

        print("LLM ranking failed:", e)

        return photos[0]


# ─────────────────────────────────────────────
# IMAGE FETCHING
# ─────────────────────────────────────────────

def fetch_pexels_image(headline):

    if not PEXELS_API_KEY:
        return None

    keywords = extract_keywords(headline)

    headers = {"Authorization": PEXELS_API_KEY}

    query = requests.utils.quote(keywords)

    url = (
        f"https://api.pexels.com/v1/search"
        f"?query={query}&per_page=5&orientation=square&size=large"
    )

    try:

        resp = requests.get(url, headers=headers, timeout=10)

        resp.raise_for_status()

        data = resp.json()

        photos = data.get("photos", [])

        if not photos:
            return None

        best_photo = choose_best_image_with_llm(headline, photos)

        img_url = best_photo["src"]["large2x"]

        img_resp = requests.get(img_url, timeout=30)

        img_resp.raise_for_status()

        return img_resp.content

    except Exception as e:

        print("Image fetch failed:", e)

        return None


# ─────────────────────────────────────────────
# GRADIENT
# ─────────────────────────────────────────────

def make_gradient_mask(width, height):

    mask = Image.new("L", (width, height), 0)

    draw = ImageDraw.Draw(mask)

    for y in range(height):

        value = int(255 * y / max(height - 1, 1))

        draw.line([(0, y), (width, y)], fill=value)

    return mask


# ─────────────────────────────────────────────
# TEXT WRAP
# ─────────────────────────────────────────────

def wrap_text(text, font, max_width, draw):

    words = text.split()

    lines = []

    current = []

    for word in words:

        candidate = " ".join(current + [word])

        bbox = draw.textbbox((0, 0), candidate, font=font)

        if (bbox[2] - bbox[0]) <= max_width:

            current.append(word)

        else:

            lines.append(" ".join(current))

            current = [word]

    if current:

        lines.append(" ".join(current))

    return lines


def pick_font_size(chars):

    if chars <= 40:
        return 64
    elif chars <= 55:
        return 56
    elif chars <= 70:
        return 48
    else:
        return 42


# ─────────────────────────────────────────────
# RENDER SLIDE
# ─────────────────────────────────────────────

def render_slide(img_bytes, headline, font_path):

    size = SLIDE_PX

    if img_bytes:

        bg = Image.open(BytesIO(img_bytes)).convert("RGB")

        w, h = bg.size

        m = min(w, h)

        bg = bg.crop(((w-m)//2, (h-m)//2, (w+m)//2, (h+m)//2))

        bg = bg.resize((size, size), Image.LANCZOS)

    else:

        bg = Image.new("RGB", (size, size), (26,58,42))

    grad_h = int(size * GRADIENT_FRACTION)

    mask = make_gradient_mask(size, grad_h)

    black_band = Image.new("RGB", (size, grad_h), (0,0,0))

    bg.paste(black_band, (0, size-grad_h), mask=mask)

    draw = ImageDraw.Draw(bg)

    text = headline.upper()

    font_size = pick_font_size(len(text))

    font = ImageFont.truetype(font_path, font_size)

    side_margin = 50

    bottom_margin = 52

    max_w = size - side_margin*2

    lines = wrap_text(text, font, max_w, draw)

    line_gap = int(font_size*0.22)

    line_h = font_size + line_gap

    total_h = len(lines)*line_h - line_gap

    y = max(size-bottom_margin-total_h, size-grad_h+10)

    for line in lines:

        draw.text((side_margin+3,y+3), line, font=font, fill=(0,0,0))

        draw.text((side_margin,y), line, font=font, fill=(255,255,255))

        y += line_h

    return bg


# ─────────────────────────────────────────────
# CREATE CAROUSEL
# ─────────────────────────────────────────────

def create_carousel(headlines, output_dir="carousel_slides"):

    os.makedirs(output_dir, exist_ok=True)

    font_path = ensure_font()

    print(f"\nGenerating {len(headlines)} slides")

    paths = []

    for i, headline in enumerate(headlines,1):

        print(f"[{i}/{len(headlines)}] {headline}")

        img_bytes = fetch_pexels_image(headline)

        slide = render_slide(img_bytes, headline, font_path)

        out = os.path.join(output_dir, f"slide_{i:02d}.png")

        slide.save(out, "PNG", optimize=True)

        paths.append(out)

    return paths