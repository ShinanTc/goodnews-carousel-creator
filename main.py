import feedparser
import datetime
import json
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

RSS_FEEDS = [
    "https://news.google.com/rss",
    "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss?hl=en-GB&gl=GB&ceid=GB:en"
]


def fetch_recent_headlines():

    today = datetime.date.today()
    seven_days_ago = today - datetime.timedelta(days=7)

    headlines = []

    for url in RSS_FEEDS:

        feed = feedparser.parse(url)

        for entry in feed.entries:

            if not hasattr(entry, "published_parsed"):
                continue

            published = entry.published_parsed

            published_date = datetime.date(
                published.tm_year,
                published.tm_mon,
                published.tm_mday
            )

            if published_date >= seven_days_ago:
                headlines.append(entry.title)

    # remove duplicates
    headlines = list(dict.fromkeys(headlines))

    # limit initial batch
    return headlines[:120]


def classify_headlines(headlines):

    numbered = "\n".join(
        [f"{i+1}. {h}" for i, h in enumerate(headlines)]
    )

    prompt = f"""
You are evaluating news headlines.

Select headlines that represent GOOD NEWS with real positive impact for:

- nature
- wildlife
- conservation
- ecosystems
- environmental protection
- climate solutions
- travel destinations related to nature
- exploration of natural places

Accept if the headline describes:
• conservation success
• species recovery
• environmental protection
• climate breakthroughs
• ecosystem restoration
• new protected natural areas
• discoveries that help the planet

Reject if it is:
• random animal fact
• curiosity science
• space imagery
• beautiful scenery
• neutral discoveries
• politics, crime, sports, business

For each headline return JSON in this format:

[
  {{
    "index": 1,
    "relevant": true
  }},
  {{
    "index": 2,
    "relevant": false
  }}
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

    try:
        data = json.loads(content)
    except:
        return []

    selected = []

    for item in data:
        if item.get("relevant"):
            idx = item.get("index", 0) - 1

            if 0 <= idx < len(headlines):
                selected.append(headlines[idx])

    return selected


def main():

    print("Fetching headlines...")

    headlines = fetch_recent_headlines()

    print(f"Fetched {len(headlines)} headlines")

    print("Filtering with LLM...")

    filtered = classify_headlines(headlines)

    # limit results
    filtered = filtered[:15]

    print("\nGood Travel / Nature News:\n")

    if not filtered:
        print("No qualifying headlines found this week.")
        return

    for i, headline in enumerate(filtered, 1):
        print(f"{i}. {headline}")


if __name__ == "__main__":
    main()