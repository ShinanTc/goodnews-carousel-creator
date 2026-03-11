import feedparser
import datetime
import json
import re
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Dedicated environment, nature, conservation, and climate RSS feeds
# All free, no API key required
RSS_FEEDS = [
    "https://www.theguardian.com/environment/rss",
    "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "https://feeds.reuters.com/reuters/environment",
    "https://www.nasa.gov/rss/dyn/breaking_news.rss",
    "https://www.ecowatch.com/feeds/latest.rss",
    "https://news.mongabay.com/feed/",
    "https://e360.yale.edu/feed",
    "https://www.conservation.org/feed",
    "https://www.worldwildlife.org/magazine/rss",
    "https://www.nationalgeographic.com/environment/rss",
    "https://insideclimatenews.org/feed/",
    "https://www.carbonbrief.org/feed",
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Fetch
# ─────────────────────────────────────────────────────────────────────────────

def fetch_recent_headlines():
    today = datetime.date.today()
    seven_days_ago = today - datetime.timedelta(days=7)
    headlines = []

    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if not hasattr(entry, "published_parsed") or not entry.published_parsed:
                    headlines.append(entry.title)
                    continue
                published = entry.published_parsed
                published_date = datetime.date(
                    published.tm_year, published.tm_mon, published.tm_mday
                )
                if published_date >= seven_days_ago:
                    headlines.append(entry.title)
        except Exception as e:
            print(f"  Warning: Could not fetch {url}: {e}", flush=True)
            continue

    headlines = list(dict.fromkeys(headlines))
    return headlines[:120]


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Classify
# ─────────────────────────────────────────────────────────────────────────────

def extract_json(text):
    """Strip markdown fences and parse JSON array from LLM output."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def classify_headlines(headlines):
    numbered = "\n".join([f"{i+1}. {h}" for i, h in enumerate(headlines)])

    prompt = f"""
You are a strict content filter for an Instagram page focused on positive nature and conservation news.

Your job is to select ONLY headlines that confirm a CONCRETE POSITIVE OUTCOME already achieved — not debates, discoveries of neutral facts, controversies, or general interest nature stories.

━━━━━━━━━━━━━━━━━━━━━━
✅ ACCEPT — only if the headline confirms ONE of these:
━━━━━━━━━━━━━━━━━━━━━━
• A species has recovered, returned, or been saved from extinction
• A new area of land or ocean has been officially protected or designated
• A conservation effort has succeeded (animals reintroduced, habitat restored, poaching stopped)
• A country, city, or company has hit a clean energy or emissions milestone
• An ecosystem has been restored (forests replanted, rivers cleaned, coral recovered)
• A climate solution or green technology has been deployed at real-world scale
• A court or government has ruled IN FAVOUR of nature or environmental protection

━━━━━━━━━━━━━━━━━━━━━━
❌ REJECT — always reject these, no exceptions:
━━━━━━━━━━━━━━━━━━━━━━
• Animal biology facts or curiosity science ("hedgehog hearing range discovered")
• Controversies, disputes, or rows about environmental topics ("company at centre of row over...")
• Sustainability greenwashing debates or labelling disputes
• Scientific studies that describe problems, risks, or observations without a positive outcome
• Deep-sea or space discoveries that are just "interesting" but change nothing
• Beautiful places, scenic destinations, or travel recommendations
• Wildlife behaviour stories with no conservation outcome
• Any politics, crime, economics, business, sports, celebrity news
• Anything framed as a warning, threat, concern, or controversy
• Anything where the positive outcome is hypothetical, potential, or in the future

━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE DECISIONS:
━━━━━━━━━━━━━━━━━━━━━━
"Secret of hedgehog hearing discovered at far beyond human range" → ❌ REJECT (curiosity animal fact)
"World's largest krill harvester at centre of row over sustainability label" → ❌ REJECT (controversy)
"Deep ocean microbes may already be prepared to tackle climate change" → ❌ REJECT (hypothetical)
"Scotland creates largest marine protected area in its history" → ✅ ACCEPT
"Humpback whale population reaches pre-whaling levels in South Atlantic" → ✅ ACCEPT
"Costa Rica runs on 100% renewable energy for record 400 days" → ✅ ACCEPT
"European court rules against logging in ancient Polish forest" → ✅ ACCEPT
"New solar farm now powers 200,000 homes in Kenya" → ✅ ACCEPT
━━━━━━━━━━━━━━━━━━━━━━

When in doubt — REJECT. It is better to miss a story than to include a bad one.

Return ONLY a raw JSON array. No markdown, no code fences, no explanation.

[
  {{"index": 1, "relevant": true}},
  {{"index": 2, "relevant": false}}
]

HEADLINES:
{numbered}
"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = completion.choices[0].message.content
    data = extract_json(content)

    if data is None:
        print(f"  Warning: Could not parse LLM response:\n{content[:300]}", flush=True)
        return []

    selected = []
    for item in data:
        if item.get("relevant"):
            idx = item.get("index", 0) - 1
            if 0 <= idx < len(headlines):
                selected.append(headlines[idx])

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Condense
# ─────────────────────────────────────────────────────────────────────────────

def condense_headline(headline, max_chars=75):
    """
    If a headline exceeds max_chars, use Groq to shorten it while keeping
    the core message intact. Returns the original if already short enough.
    """
    if len(headline) <= max_chars:
        return headline

    prompt = f"""Condense this news headline to under {max_chars} characters.
Keep it punchy, factual, and impactful — no fluff, no ellipsis.
Return ONLY the condensed headline. No quotes, no explanation.

Original: {headline}"""

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    result = completion.choices[0].message.content.strip().strip("\"'")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Fetching headlines from nature & environment feeds...", flush=True)
    headlines = fetch_recent_headlines()
    print(f"Fetched {len(headlines)} unique headlines", flush=True)

    if not headlines:
        print("No headlines fetched. Check your internet connection or RSS feed URLs.", flush=True)
        return

    print("Filtering with LLM...", flush=True)
    filtered = classify_headlines(headlines)
    filtered = filtered[:15]

    if not filtered:
        print("\nNo qualifying headlines found this week.", flush=True)
        return

    print(f"\nFound {len(filtered)} qualifying headlines. Condensing long ones...", flush=True)
    condensed = []
    for h in filtered:
        short = condense_headline(h)
        if short != h:
            print(f"  ✂  '{h[:55]}...' → '{short}'", flush=True)
        condensed.append(short)

    print("\n🌿 Good Travel / Nature News This Week:\n", flush=True)
    for i, headline in enumerate(condensed, 1):
        print(f"{i:2d}. {headline}", flush=True)

    # ── Generate carousel slides ───────────────────────────────────────────
    from create_carousel import create_carousel
    create_carousel(condensed, output_dir="carousel_slides")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        input("Press Enter to close...")