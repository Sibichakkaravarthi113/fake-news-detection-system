# ============================================================
# Google-based Online Fact Check using SerpAPI (Multilingual)
# ============================================================

import requests
import difflib
from urllib.parse import urlparse
from langdetect import detect, DetectorFactory

# Make langdetect deterministic
DetectorFactory.seed = 0

# 1) Put your SerpAPI key here
SERPAPI_KEY = "e803e2e8ec7e819522306a41e9c3fd5dd739e53a0227a6bc42df76c61275890d"  # <-- REPLACE THIS
SERPAPI_URL = "https://serpapi.com/search"

# 2) Trusted domains (Set C + expanded)
TRUSTED_DOMAINS = [
    # International
    "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com", "cnn.com",
    "nytimes.com", "theguardian.com", "aljazeera.com", "washingtonpost.com",
    "usatoday.com", "npr.org",

    # Indian
    "thehindu.com", "hindustantimes.com", "indiatoday.in", "ndtv.com",
    "timesofindia.com", "livemint.com", "economictimes.com", "business-standard.com",
    "news18.com", "deccanherald.com",

    # Tech
    "techcrunch.com", "theverge.com", "wired.com", "engadget.com",
    "arstechnica.com", "gsmarena.com", "tomshardware.com",

    # Business
    "bloomberg.com", "cnbc.com", "forbes.com", "ft.com", "moneycontrol.com"

     # --- Tamil news sources ---
    "dinamalar.com",        # தினமலர் 
    "dinakaran.com",        # தினகரன்
    "dailythanthi.com",     # தினத்தந்தி
    "vikatan.com",          # விகடன்
    "polimernews.com",      # பொலிமர் நியூஸ்
    "news7tamil.live",      # நியூஸ் 7 தமிழ்
    "puthiyathalaimurai.com",  # புதுயதலைமுறை
    "oneindia.com",         # OneIndia தமிழ்
]


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def clean_text(text: str) -> str:
    """Collapse extra whitespace."""
    return " ".join(text.strip().split())


def similarity(a: str, b: str) -> float:
    """Simple string similarity between 0 and 1."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def is_trusted(url: str) -> bool:
    """Check if the URL belongs to a trusted domain."""
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return any(dom in host for dom in TRUSTED_DOMAINS)


def detect_language_code(text: str) -> str:
    """
    Detect language code using langdetect.
    Fallback to 'en' if detection fails.
    """
    try:
        code = detect(text)
        return code
    except Exception:
        return "en"


def _search_google_news(query: str, lang_code: str):
    """
    Call SerpAPI (Google News) and return news_results list + optional error.

    lang_code: e.g., 'en', 'hi', 'ta', 'te', ...
    Mapped directly to Google's 'hl' param (UI language).
    """
    params = {
        "engine": "google",
        "q": query,
        "tbm": "nws",      # news vertical
        "num": 10,
        "hl": lang_code,
        "api_key": SERPAPI_KEY,
    }

    resp = requests.get(SERPAPI_URL, params=params, timeout=15)
    if resp.status_code != 200:
        return [], f"HTTP {resp.status_code}: {resp.text}"

    data = resp.json()
    news_results = data.get("news_results", [])
    return news_results, None


# ------------------------------------------------------------
# Main fact-check function
# ------------------------------------------------------------
def fact_check(text: str) -> dict:
    """
    Fact-check the given news text using Google News via SerpAPI.

    Multilingual logic:
      - Detect input language (ta, hi, en, etc.)
      - First search Google News in that language (hl=<code>)
      - If no strong match found → fallback to English (hl='en')

    Returns:
      {
        "query": "...",
        "verdict": "REAL" | "UNVERIFIED",
        "confidence": "High" | "Medium" | "Low",
        "matches": [
           {"source": "...", "title": "...", "url": "...", "similarity": 0.73},
           ...
        ]
      }
    """
    text_clean = clean_text(text)
    # Use the first ~12 words as concise search query
    query = " ".join(text_clean.split()[:12])

    # 1) Detect language of input
    detected_lang = detect_language_code(text_clean)

    # 2) Try searching in detected language
    results, error = _search_google_news(query, detected_lang)

    # Prepare output structure
    out = {
        "query": query,
        "verdict": "UNVERIFIED",
        "confidence": "Low",
        "matches": [],
        "search_language": detected_lang,   # for debugging / display
    }

    if error:
        out["matches"] = [{
            "source": "ERROR",
            "title": error,
            "url": "",
            "similarity": 0.0
        }]
        return out

    # If no results in that language, fallback to English
    if not results and detected_lang != "en":
        results, error_en = _search_google_news(query, "en")
        out["search_language"] = f"{detected_lang} -> en"
        if error_en:
            out["matches"] = [{
                "source": "ERROR",
                "title": error_en,
                "url": "",
                "similarity": 0.0
            }]
            return out

    if not results:
        return out

    # 3) Filter and keep only trusted & similar articles
    for item in results:
        title = item.get("title") or ""
        snippet = item.get("snippet") or ""
        link = item.get("link") or ""
        source = item.get("source") or "Unknown"

        if not link or not is_trusted(link):
            continue

        combined = f"{title}. {snippet}"
        sim = similarity(text_clean, combined)

        # Relaxed threshold for demo: 0.30
        if sim >= 0.25:
            out["matches"].append({
                "source": source,
                "title": title,
                "url": link,
                "similarity": round(sim, 3),
            })

    if out["matches"]:
        out["verdict"] = "REAL"
        if len(out["matches"]) >= 3:
            out["confidence"] = "High"
        else:
            out["confidence"] = "Medium"

    return out


# ------------------------------------------------------------
# Local test (run: python src/fact_check_google.py)
# ------------------------------------------------------------
if __name__ == "__main__":
    from pprint import pprint

    # Try English
    print("=== English Example ===")
    news_en = "OpenAI releases a new version of ChatGPT with improved reasoning."
    pprint(fact_check(news_en))

    # Try a non-English example (Tamil / Hindi etc.) if you like:
    # news_ta = "இந்தியா 2023ல் சூரியனை ஆய்வு செய்ய Aditya-L1 விண்கலத்தை அனுப்பியது."
    # pprint(fact_check(news_ta))
