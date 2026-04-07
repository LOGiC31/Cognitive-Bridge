"""
Build the MedlinePlus glossary JSON for the Cognitive Bridge extension.

This script fetches medical terms and definitions from the MedlinePlus
Health Topics API and compiles them into a JSON file for offline use.

Usage:
    python scripts/build_glossary.py

Output:
    extension/data/medlineplus_glossary.json
"""

import json
import os
import re
import xml.etree.ElementTree as ET
from urllib.request import urlopen, Request
from urllib.error import URLError

MEDLINEPLUS_API = "https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term=health"

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "extension", "data", "medlineplus_glossary.json"
)

CURATED_TERMS = {
    "hypertension": {
        "definition": "High blood pressure — a condition where the force of blood against your artery walls is consistently too high.",
        "url": "https://medlineplus.gov/highbloodpressure.html"
    },
    "diabetes": {
        "definition": "A condition where your body has trouble controlling blood sugar levels.",
        "url": "https://medlineplus.gov/diabetes.html"
    },
    "myocardial infarction": {
        "definition": "A heart attack — when blood flow to part of the heart is blocked, causing damage.",
        "url": "https://medlineplus.gov/heartattack.html"
    },
    "pneumonia": {
        "definition": "An infection that inflames the air sacs in one or both lungs.",
        "url": "https://medlineplus.gov/pneumonia.html"
    },
    "anemia": {
        "definition": "A condition where you don't have enough healthy red blood cells to carry adequate oxygen.",
        "url": "https://medlineplus.gov/anemia.html"
    },
}


def fetch_medlineplus_topics():
    """Fetch health topics from MedlinePlus web services API."""
    terms = {}

    try:
        req = Request(MEDLINEPLUS_API, headers={"User-Agent": "CognitiveBridge/1.0"})
        with urlopen(req, timeout=30) as response:
            data = response.read()

        root = ET.fromstring(data)

        for doc in root.iter("document"):
            title_el = doc.find(".//content[@name='title']")
            snippet_el = doc.find(".//content[@name='FullSummary']")
            url_attr = doc.get("url", "")

            if title_el is not None and title_el.text:
                title = title_el.text.strip().lower()
                snippet = ""
                if snippet_el is not None and snippet_el.text:
                    snippet = clean_html(snippet_el.text.strip())
                    snippet = truncate(snippet, 200)

                if snippet:
                    terms[title] = {
                        "definition": snippet,
                        "url": url_attr,
                    }
    except (URLError, ET.ParseError) as e:
        print(f"Warning: Could not fetch from MedlinePlus API: {e}")
        print("Using curated terms only.")

    return terms


def clean_html(text):
    """Remove HTML tags from text."""
    return re.sub(r"<[^>]+>", "", text).strip()


def truncate(text, max_words=200):
    """Truncate text to max_words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def build_glossary():
    """Build the complete glossary by combining API results with curated terms."""
    print("Fetching MedlinePlus health topics...")
    api_terms = fetch_medlineplus_topics()
    print(f"  Fetched {len(api_terms)} terms from API.")

    glossary = {**api_terms, **CURATED_TERMS}

    existing_path = OUTPUT_PATH
    if os.path.exists(existing_path):
        with open(existing_path, "r") as f:
            existing = json.load(f)
        for key, val in existing.items():
            if key not in glossary:
                glossary[key] = val

    print(f"Total glossary entries: {len(glossary)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(glossary, f, indent=2, sort_keys=True)

    print(f"Glossary written to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_glossary()
