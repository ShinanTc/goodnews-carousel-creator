import feedparser
import datetime
import json
import re
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────────────────────────────────────────
# RSS Feeds — nature, conservation, climate, and positive/solutions news
# ─────────────────────────────────────────────────────────────────────────────

RSS_FEEDS = [
    # ── Environment & Conservation ─────────────────────────────────────────
    "https://www.theguardian.com/environment/rss",
    "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "https://feeds.reuters.com/reuters/environment",
    "https://www.nasa.gov/rss/dyn/breaking_news.rss",
    "https://www.ecowatch.com/feeds/latest.rss",
    "https://news.mongabay.com/feed/",
    "https://e360.yale.edu/feed",
    "https://www.conservation.org/feed",
    "https://insideclimatenews.org/feed/",
    "https://www.carbonbrief.org/feed",
    "https://www.rewildingbritain.org.uk/explore-rewilding/news-and-updates?format=rss",
    "https://www.fauna-flora.org/news/feed/",
    "https://www.birdlife.org/news/rss/",
    "https://www.iucn.org/news/feed",
    "https://www.clientearth.org/feed/",
    "https://www.globalforestwatch.org/blog/feed/",
    "https://www.oceanconservancy.org/feed/",
    "https://www.surfrider.org/coastal-blog/feed",
    "https://coral.org/blog/feed/",
    "https://www.awf.org/blog/feed",
    "https://www.wcs.org/our-work/species/rss",
    "https://newint.org/feed",

    # ── Good News / Solutions Journalism ──────────────────────────────────
    "https://www.goodnewsnetwork.org/feed/",
    "https://www.positive.news/feed/",
    "https://www.yesmagazine.org/feed/",
    "https://www.solutionsjournalism.org/feed",
    "https://www.upworthy.com/rss",
    "https://www.good.is/rss",
    "https://reasonstobecheerful.world/feed/",
    "https://constructivejournalism.org/feed/",
    "https://theoptimist.com/feed/",

    # ── Science & Nature ──────────────────────────────────────────────────
    "https://www.nationalgeographic.com/environment/rss",
    "https://www.worldwildlife.org/magazine/rss",
    "https://www.newscientist.com/subject/environment/feed/",
    "https://www.sciencedaily.com/rss/earth_climate.xml",
    "https://phys.org/rss-feed/earth-news/",
    "https://www.earth.com/feed/",

    # ── Clean Energy & Climate Solutions ──────────────────────────────────
    "https://cleantechnica.com/feed/",
    "https://electrek.co/feed/",
    "https://www.renewableenergyworld.com/feed/",
    "https://energymonitor.ai/feed/",
    "https://www.greenbiz.com/feeds/news",
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
    return headlines[:200]


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


HOOK_WORDS = {
    "how", "why", "what", "who", "when", "where", "which", "is", "are",
    "can", "will", "could", "should", "would", "inside", "meet", "watch",
    "see", "guess",
}

REJECT_PHRASES = [
    "researchers warn", "scientists warn", "experts warn", "study warns",
    "under threat", "at risk", "in danger", "faces threat", "faces extinction",
    "could be lost", "may disappear", "raising concerns", "sounding alarm",
]

def pre_filter(headlines):
    """Fast Python filter before the LLM — removes obvious rejects for free."""
    kept = []
    for h in headlines:
        lower = h.lower().strip()
        first_word = lower.split()[0].rstrip("'s") if lower.split() else ""

        if first_word in HOOK_WORDS:
            print(f"  ⚡ Pre-filtered (hook): {h}", flush=True)
            continue

        if any(phrase in lower for phrase in REJECT_PHRASES):
            print(f"  ⚡ Pre-filtered (warning): {h}", flush=True)
            continue

        kept.append(h)
    return kept


def classify_headlines(headlines):
    headlines = pre_filter(headlines)
    numbered = "\n".join([f"{i+1}. {h}" for i, h in enumerate(headlines)])

    prompt = f"""
You are a content curator for an Instagram page that shares uplifting, real-world wins for nature and humanity.

Your job: select headlines that report a REAL, ALREADY-HAPPENED positive outcome — something that has concretely improved the world.

━━━━━━━━━━━━━━━━━━━━━━
✅ ACCEPT — if the headline confirms any of these:
━━━━━━━━━━━━━━━━━━━━━━

NATURE & WILDLIFE:
• A species has recovered, returned, or been protected from extinction
• A new area of land or ocean has been officially protected or designated
• A conservation or rewilding effort has succeeded (animals reintroduced, habitat restored, poaching stopped)
• An ecosystem has been restored (forests, rivers cleaned, coral recovered, mangroves returning)
• A wildlife population has grown or is thriving
• An animal sanctuary, reserve, or refuge has been established

ENVIRONMENT & CLIMATE:
• A country, city, or region has hit a clean energy or emissions milestone (e.g. 100% renewables for X days)
• A climate solution or clean technology has been deployed at real scale
• A river, lake, or ocean area has been cleaned up or declared safe
• A court or government has ruled IN FAVOUR of nature or the environment

HUMAN & SOCIAL GOOD (with a clear benefit to people or nature):
• A humane or progressive criminal justice reform that has measurably worked (e.g. Netherlands reducing prisons due to falling crime)
• A public health or wellbeing initiative that has expanded or succeeded (e.g. nature prescriptions, free outdoor swimming)
• A city or country has made nature more accessible to its people
• A social or community win that directly benefits wildlife or environment

SCIENCE WITH A CLEAR UPSIDE:
• Research confirming animals or ecosystems are thriving or more resilient than expected
• A discovery that gives scientists or conservationists new tools to protect nature
• New evidence that a recovery effort is working better than expected

━━━━━━━━━━━━━━━━━━━━━━
❌ REJECT — always reject:
━━━━━━━━━━━━━━━━━━━━━━
• HOOK HEADLINES — any headline that starts with "How", "Why", "What", "The story of",
  "Inside", "Meet", "Is", "Are", "Can", "Will", "Could", "Should", or any question word.
  These are editorial hooks, not factual news. A good headline states what happened, not asks a question.
  e.g. "How reintroducing beavers is changing our landscape" → ❌ REJECT
  e.g. "Why saving seagrass could help save coastlines" → ❌ REJECT
  e.g. "How the Netherlands bent bureaucracy into something beautiful" → ❌ REJECT
• Vague or feel-good headlines with no stated concrete outcome (e.g. "Superbloom carpets Death Valley" — a natural event, not a win)
• Natural seasonal phenomena or weather events with no conservation relevance
• Curiosity animal facts with no conservation outcome ("hedgehog hearing range discovered")
• Controversies, disputes, rows, or political conflicts about environmental topics
• Greenwashing debates or labelling disputes
• Studies that only describe problems, risks, or threats — even new ones
• Anything containing the words "warn", "warning", "researchers warn", "scientists warn", "at risk", "under threat", "concern" — these are bad news dressed as news
• Travel recommendations, beautiful destinations, or scenic places
• Anything framed as a warning, concern, setback, or threat
• Outcomes that are hypothetical, planned, potential, or in the future
• Business, economics, crime, celebrity, or sports news with no nature/human-benefit angle
• Clean energy stories that are purely financial/economic with no clear human or nature benefit (e.g. "China less vulnerable to energy shocks" — this is geopolitics, not a nature/human win)

━━━━━━━━━━━━━━━━━━━━━━
EXAMPLE DECISIONS:
━━━━━━━━━━━━━━━━━━━━━━
"Giant pandas are no longer endangered" → ✅ ACCEPT
"Great Lakes otters flourish in US and Ontario" → ✅ ACCEPT (population thriving)
"Tortoiseshell butterfly recolonizes England decades after elm disease eliminated it" → ✅ ACCEPT (species recovery)
"New baby boom for cheetahs in India after world-first reintroduction" → ✅ ACCEPT (reintroduction success)
"Hawaii university hauls 84 tons of derelict gear from Pacific Ocean" → ✅ ACCEPT (cleanup completed)
"The Netherlands has closed 20+ prisons as crime rates fall due to prevention programmes" → ✅ ACCEPT (social good with real outcome)
"UK's first nature prescription pilot is expanding nationwide" → ✅ ACCEPT (health + nature win)
"Africa's mangroves are recovering faster than expected" → ✅ ACCEPT (ecosystem recovery confirmed)
"Costa Rica runs on 100% renewable energy for record 400 days" → ✅ ACCEPT
"Global Ocean Treaty is officially in force" → ✅ ACCEPT
"How reintroducing beavers is changing our landscape" → ❌ REJECT (hook headline starting with "How", states no concrete outcome)
"Why saving seagrass meadows could help save the world's coastlines" → ❌ REJECT (hook headline starting with "Why", hypothetical)
"How the Netherlands bent bureaucracy into something beautiful" → ❌ REJECT (hook headline, vague)
"Superbloom carpets Death Valley" → ❌ REJECT (natural seasonal event, no conservation outcome)
"Mangrove forests are short of breath, researchers warn" → ❌ REJECT (contains "warn" — this is bad news)
"Oregon buys Abiqua Falls property" → ❌ REJECT (too vague — no stated conservation outcome)
"China's clean energy push makes it less vulnerable to energy shocks" → ❌ REJECT (geopolitics, not a nature win)
"Secret of hedgehog hearing discovered at far beyond human range" → ❌ REJECT (curiosity fact, no outcome)
"Deep ocean microbes may already be prepared to tackle climate change" → ❌ REJECT (hypothetical)

━━━━━━━━━━━━━━━━━━━━━━
When in doubt — REJECT. Quality over quantity.

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
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Fetching headlines from nature, conservation & good news feeds...", flush=True)
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

    print(f"\nFound {len(filtered)} qualifying headlines.\n", flush=True)

    print("🌿 Good News This Week:\n", flush=True)
    for i, headline in enumerate(filtered, 1):
        print(f"{i:2d}. {headline}", flush=True)

    # ── Generate carousel slides ───────────────────────────────────────────
    from create_carousel import create_carousel
    create_carousel(filtered, output_dir="carousel_slides")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        input("Press Enter to close...")