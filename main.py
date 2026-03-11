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
    # The Guardian - Environment
    "https://www.theguardian.com/environment/rss",
    # BBC - Science & Environment
    "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    # Reuters - Environment
    "https://feeds.reuters.com/reuters/environment",
    # NASA Breaking News
    "https://www.nasa.gov/rss/dyn/breaking_news.rss",
    # EcoWatch
    "https://www.ecowatch.com/feeds/latest.rss",
    # Mongabay (conservation & wildlife)
    "https://news.mongabay.com/feed/",
    # Yale Environment 360
    "https://e360.yale.edu/feed",
    # Conservation.org
    "https://www.conservation.org/feed",
    # WWF News
    "https://www.worldwildlife.org/magazine/rss",
    # National Geographic (environment section)
    "https://www.nationalgeographic.com/environment/rss",
    # Inside Climate News
    "https://insideclimatenews.org/feed/",
    # Carbon Brief
    "https://www.carbonbrief.org/feed",
]


def fetch_recent_headlines():

    today = datetime.date.today()
    seven_days_ago = today - datetime.timedelta(days=7)

    headlines = []

    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)

            for entry in feed.entries:

                if not hasattr(entry, "published_parsed") or not entry.published_parsed:
                    # If no date, include it anyway (some feeds omit dates)
                    headlines.append(entry.title)
                    continue

                published = entry.published_parsed

                published_date = datetime.date(
                    published.tm_year,
                    published.tm_mon,
                    published.tm_mday
                )

                if published_date >= seven_days_ago:
                    headlines.append(entry.title)

        except Exception as e:
            print(f"  Warning: Could not fetch {url}: {e}")
            continue

    # Remove duplicates while preserving order
    headlines = list(dict.fromkeys(headlines))

    # Limit initial batch
    return headlines[:120]


def extract_json(text):
    """
    Robustly extract a JSON array from LLM output.
    Handles markdown code fences like ```json ... ``` and raw JSON.
    """
    # Strip markdown code fences (the main bug causing silent failures)
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first [...] block in the response
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def classify_headlines(headlines):

    numbered = "\n".join(
        [f"{i+1}. {h}" for i, h in enumerate(headlines)]
    )

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
"Secret of hedgehog hearing discovered at far beyond human range" → ❌ REJECT (curiosity animal fact, no conservation outcome)
"World's largest krill harvester at centre of row over sustainability label" → ❌ REJECT (controversy, no positive outcome)
"Deep ocean microbes may already be prepared to tackle climate change" → ❌ REJECT (hypothetical, neutral observation)
"Scotland creates largest marine protected area in its history" → ✅ ACCEPT (new protected area, concrete outcome)
"Humpback whale population reaches pre-whaling levels in South Atlantic" → ✅ ACCEPT (species recovery, concrete outcome)
"Costa Rica runs on 100% renewable energy for record 400 days" → ✅ ACCEPT (clean energy milestone achieved)
"European court rules against logging in ancient Polish forest" → ✅ ACCEPT (legal win for nature, concrete outcome)
"New solar farm now powers 200,000 homes in Kenya" → ✅ ACCEPT (real-world climate solution deployed)

━━━━━━━━━━━━━━━━━━━━━━

When in doubt — REJECT. It is better to miss a story than to include a bad one.

Return ONLY a raw JSON array. No markdown, no code fences, no explanation, nothing else.

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
        print(f"  Warning: Could not parse LLM response. Raw output:\n{content[:300]}")
        return []

    selected = []

    for item in data:
        if item.get("relevant"):
            idx = item.get("index", 0) - 1

            if 0 <= idx < len(headlines):
                selected.append(headlines[idx])

    return selected


def main():

    print("Fetching headlines from nature & environment feeds...")

    headlines = fetch_recent_headlines()

    print(f"Fetched {len(headlines)} unique headlines")

    if not headlines:
        print("No headlines fetched. Check your internet connection or RSS feed URLs.")
        return

    print("Filtering with LLM...")

    filtered = classify_headlines(headlines)

    # Limit results
    filtered = filtered[:15]

    print("\n🌿 Good Travel / Nature News This Week:\n")

    if not filtered:
        print("No qualifying headlines found this week.")
        return

    for i, headline in enumerate(filtered, 1):
        print(f"{i}. {headline}")


if __name__ == "__main__":
    main()