import boto3
from googlesearch import search
import requests
from bs4 import BeautifulSoup

def extract_entities(text):
    """Extract entities (names, places, etc.) from the input text using AWS Comprehend."""
    comprehend = boto3.client("comprehend", region_name="us-east-1")
    response = comprehend.detect_entities(Text=text, LanguageCode="en")
    entities = [ent["Text"] for ent in response.get("Entities", []) if ent["Score"] > 0.90]
    return list(set(entities))  # unique entities


def verify_entity_online(entity):
    """Search Google for the entity and check if credible news sites confirm it."""
    credible_sources = ["bbc.com", "reuters.com", "ndtv.com", "cnn.com", "nytimes.com", "thehindu.com", "hindustantimes.com"]
    try:
        results = list(search(entity, num_results=5))
        credible_hits = [url for url in results if any(src in url for src in credible_sources)]
        return len(credible_hits) > 0, credible_hits
    except Exception as e:
        return False, [f"Error during search: {e}"]


def fact_check(text):
    """
    Perform entity extraction and credibility verification.
    Returns a dictionary with detected entities and reliability scores.
    """
    result = {"entities": [], "verified": [], "unverified": []}
    entities = extract_entities(text)

    for entity in entities:
        is_verified, links = verify_entity_online(entity)
        result["entities"].append(entity)
        if is_verified:
            result["verified"].append({"entity": entity, "sources": links})
        else:
            result["unverified"].append({"entity": entity, "note": "No trusted confirmation found."})

    return result


if __name__ == "__main__":
    test_news = "MrBeast has 44 million subscribers on YouTube."
    print(fact_check(test_news))
