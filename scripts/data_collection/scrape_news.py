"""
Scrape Telugu news articles from major Telugu news websites.

Sources:
- Sakshi (sakshi.com) - HIGH priority, clean HTML
- Andhra Jyothi (andhrajyothy.com) - HIGH priority
- Namaste Telangana (ntnews.com) - HIGH priority
- Eenadu (eenadu.net) - MEDIUM priority, JS-heavy
- Prajasakti (prajasakti.com) - HIGH priority

Target: 5,000 news articles across categories:
- Politics, Sports, Business, Entertainment, Technology, Culture

Usage:
    python scripts/data_collection/scrape_news.py --source sakshi --limit 1000
    python scripts/data_collection/scrape_news.py --all --limit 5000
    python scripts/data_collection/scrape_news.py --verify
"""

import os
import sys
import json
import time
import random
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Installing required packages...")
    os.system(f"{sys.executable} -m pip install requests beautifulsoup4 lxml -q")
    import requests
    from bs4 import BeautifulSoup


@dataclass
class NewsArticle:
    """Represents a scraped news article."""
    id: str
    title: str
    content: str
    url: str
    source: str
    category: str
    date_scraped: str
    date_published: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


# News source configurations
NEWS_SOURCES = {
    "sakshi": {
        "name": "Sakshi",
        "base_url": "https://www.sakshi.com",
        "categories": {
            "politics": "/news/politics",
            "sports": "/sports",
            "business": "/business",
            "entertainment": "/entertainment",
            "technology": "/technology",
            "national": "/news/national"
        },
        "article_selector": "article.story-card a, div.news-card a, a.story-link",
        "title_selector": "h1.story-title, h1.article-title, h1",
        "content_selector": "div.story-content, div.article-body, div.content-area",
        "date_selector": "time, span.date, div.date",
        "priority": "HIGH"
    },
    "andhrajyothy": {
        "name": "Andhra Jyothi",
        "base_url": "https://www.andhrajyothy.com",
        "categories": {
            "politics": "/politics",
            "sports": "/sports",
            "business": "/business",
            "entertainment": "/entertainment",
            "national": "/national"
        },
        "article_selector": "div.news-item a, article a, a.news-link",
        "title_selector": "h1.title, h1.news-title, h1",
        "content_selector": "div.news-content, div.article-content, div.story-body",
        "date_selector": "span.date, time, div.published-date",
        "priority": "HIGH"
    },
    "ntnews": {
        "name": "Namaste Telangana",
        "base_url": "https://www.ntnews.com",
        "categories": {
            "telangana": "/telangana",
            "politics": "/politics",
            "sports": "/sports",
            "national": "/national"
        },
        "article_selector": "div.news-item a, a.article-link",
        "title_selector": "h1.entry-title, h1",
        "content_selector": "div.entry-content, div.article-content",
        "date_selector": "time, span.date",
        "priority": "HIGH"
    },
    "prajasakti": {
        "name": "Prajasakti",
        "base_url": "https://www.prajasakti.com",
        "categories": {
            "politics": "/category/politics",
            "national": "/category/national",
            "telangana": "/category/telangana",
            "andhra": "/category/andhra-pradesh"
        },
        "article_selector": "article a, div.post a",
        "title_selector": "h1.entry-title, h1",
        "content_selector": "div.entry-content, article.content",
        "date_selector": "time, span.posted-on",
        "priority": "HIGH"
    }
}

# HTTP headers to mimic browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "te,en-US;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# Output directory
OUTPUT_DIR = Path("data/domain/news/raw")


def generate_article_id(url: str) -> str:
    """Generate unique ID for article based on URL."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


def is_telugu_text(text: str) -> bool:
    """Check if text contains significant Telugu content."""
    if not text:
        return False
    # Telugu Unicode range: 0C00-0C7F
    telugu_chars = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')
    return telugu_chars > len(text) * 0.3  # At least 30% Telugu


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove common garbage
    garbage = ["Advertisement", "Share", "Tweet", "WhatsApp", "Facebook"]
    for g in garbage:
        text = text.replace(g, "")
    return text.strip()


def fetch_page(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    """Fetch and parse a web page."""
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            response.encoding = 'utf-8'
            return BeautifulSoup(response.text, 'lxml')
        except requests.RequestException as e:
            print(f"  Attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
    return None


def extract_article_links(soup: BeautifulSoup, config: Dict, base_url: str) -> List[str]:
    """Extract article links from a category page."""
    links = []
    selector = config["article_selector"]
    
    for a_tag in soup.select(selector):
        href = a_tag.get("href", "")
        if href:
            # Make absolute URL
            full_url = urljoin(base_url, href)
            # Filter out non-article links
            if any(x in full_url for x in ["/photo", "/video", "/gallery", "#", "javascript"]):
                continue
            links.append(full_url)
    
    return list(set(links))  # Remove duplicates


def scrape_article(url: str, config: Dict) -> Optional[NewsArticle]:
    """Scrape a single news article."""
    soup = fetch_page(url)
    if not soup:
        return None
    
    # Extract title
    title_elem = soup.select_one(config["title_selector"])
    title = title_elem.get_text(strip=True) if title_elem else ""
    
    # Extract content
    content_elem = soup.select_one(config["content_selector"])
    if content_elem:
        # Get all paragraph text
        paragraphs = content_elem.find_all(['p', 'div'], recursive=True)
        content = " ".join(p.get_text(strip=True) for p in paragraphs)
    else:
        content = ""
    
    content = clean_text(content)
    
    # Skip if not enough Telugu content
    if not is_telugu_text(content) or len(content) < 200:
        return None
    
    # Extract date
    date_elem = soup.select_one(config["date_selector"])
    date_published = date_elem.get_text(strip=True) if date_elem else None
    
    return NewsArticle(
        id=generate_article_id(url),
        title=clean_text(title),
        content=content,
        url=url,
        source=config["name"],
        category="",  # Will be set by caller
        date_scraped=datetime.now().isoformat(),
        date_published=date_published
    )


def scrape_source(source_key: str, limit: int = 1000, delay: float = 1.5) -> List[NewsArticle]:
    """Scrape articles from a single news source."""
    if source_key not in NEWS_SOURCES:
        print(f"Unknown source: {source_key}")
        return []
    
    config = NEWS_SOURCES[source_key]
    articles = []
    seen_urls = set()
    
    print(f"\n📰 Scraping {config['name']} ({config['priority']} priority)")
    print(f"   Base URL: {config['base_url']}")
    print(f"   Target: {limit} articles")
    print("-" * 50)
    
    for category, path in config["categories"].items():
        if len(articles) >= limit:
            break
        
        category_url = config["base_url"] + path
        print(f"\n  Category: {category}")
        print(f"  URL: {category_url}")
        
        # Fetch category page
        soup = fetch_page(category_url)
        if not soup:
            print(f"  ✗ Failed to fetch category page")
            continue
        
        # Get article links
        links = extract_article_links(soup, config, config["base_url"])
        print(f"  Found {len(links)} article links")
        
        # Scrape each article
        category_articles = 0
        for url in links:
            if len(articles) >= limit:
                break
            if url in seen_urls:
                continue
            seen_urls.add(url)
            
            article = scrape_article(url, config)
            if article:
                article.category = category
                articles.append(article)
                category_articles += 1
                print(f"    ✓ [{len(articles)}] {article.title[:50]}...")
            
            # Polite delay
            time.sleep(delay + random.uniform(0, 0.5))
        
        print(f"  Scraped {category_articles} articles from {category}")
    
    print(f"\n✓ Total: {len(articles)} articles from {config['name']}")
    return articles


def scrape_all_sources(limit: int = 5000, delay: float = 1.5) -> List[NewsArticle]:
    """Scrape from all configured news sources."""
    all_articles = []
    per_source_limit = limit // len(NEWS_SOURCES)
    
    for source_key in NEWS_SOURCES:
        articles = scrape_source(source_key, per_source_limit, delay)
        all_articles.extend(articles)
    
    return all_articles


def save_articles(articles: List[NewsArticle], filename: str = None):
    """Save scraped articles to JSON file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"news_articles_{timestamp}.json"
    
    output_path = OUTPUT_DIR / filename
    
    data = {
        "metadata": {
            "total_articles": len(articles),
            "sources": list(set(a.source for a in articles)),
            "categories": list(set(a.category for a in articles)),
            "scraped_at": datetime.now().isoformat()
        },
        "articles": [a.to_dict() for a in articles]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Saved {len(articles)} articles to {output_path}")
    return output_path


def verify_scraped_data():
    """Verify and summarize scraped data."""
    print("=" * 60)
    print("Verifying Scraped News Data")
    print("=" * 60)
    
    if not OUTPUT_DIR.exists():
        print(f"✗ Output directory not found: {OUTPUT_DIR}")
        return
    
    json_files = list(OUTPUT_DIR.glob("*.json"))
    
    if not json_files:
        print("✗ No JSON files found")
        return
    
    total_articles = 0
    sources = set()
    categories = set()
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        articles = data.get("articles", [])
        total_articles += len(articles)
        
        for article in articles:
            sources.add(article.get("source", "unknown"))
            categories.add(article.get("category", "unknown"))
        
        print(f"\n📄 {json_file.name}")
        print(f"   Articles: {len(articles)}")
    
    print(f"\n📊 Summary")
    print(f"   Total articles: {total_articles}")
    print(f"   Sources: {', '.join(sources)}")
    print(f"   Categories: {', '.join(categories)}")
    
    # Check content quality
    if json_files:
        with open(json_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if data.get("articles"):
            sample = data["articles"][0]
            print(f"\n📝 Sample article:")
            print(f"   Title: {sample.get('title', '')[:60]}...")
            print(f"   Content length: {len(sample.get('content', ''))} chars")
            print(f"   Telugu check: {is_telugu_text(sample.get('content', ''))}")


def main():
    parser = argparse.ArgumentParser(description="Scrape Telugu news articles")
    parser.add_argument("--source", type=str, help="Scrape specific source (sakshi, andhrajyothy, ntnews, prajasakti)")
    parser.add_argument("--all", action="store_true", help="Scrape from all sources")
    parser.add_argument("--limit", type=int, default=1000, help="Max articles to scrape (default: 1000)")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between requests in seconds")
    parser.add_argument("--verify", action="store_true", help="Verify scraped data")
    parser.add_argument("--output", type=str, help="Output filename")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_scraped_data()
    elif args.all:
        articles = scrape_all_sources(args.limit, args.delay)
        if articles:
            save_articles(articles, args.output)
    elif args.source:
        articles = scrape_source(args.source, args.limit, args.delay)
        if articles:
            save_articles(articles, args.output)
    else:
        parser.print_help()
        print("\n📋 Available sources:")
        for key, config in NEWS_SOURCES.items():
            print(f"  - {key}: {config['name']} ({config['priority']} priority)")


if __name__ == "__main__":
    main()
