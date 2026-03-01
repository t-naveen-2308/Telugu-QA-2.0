"""
Practical scraper to collect real Telugu content from working sources.
Uses simpler extraction methods that work with current site structures.
"""
import os
import sys
import json
import time
import random
import hashlib
import re
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse, quote

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "te,en-US;q=0.9,en;q=0.8",
}

def is_telugu(text, threshold=0.3):
    if not text: return False
    tc = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')
    return tc > len(text) * threshold

def clean(text):
    if not text: return ""
    text = re.sub(r'\s+', ' ', text)
    for g in ["Advertisement", "Share", "Tweet", "WhatsApp", "Facebook", "Download PDF", "Print"]:
        text = text.replace(g, "")
    return text.strip()

def fetch(url, timeout=15):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, verify=False)
        r.encoding = 'utf-8'
        return BeautifulSoup(r.text, 'lxml')
    except Exception as e:
        print(f"  Failed: {url} - {e}")
        return None

def get_all_links(soup, base_url):
    """Get all internal links from a page."""
    links = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        base_parsed = urlparse(base_url)
        if parsed.netloc == base_parsed.netloc and '#' not in full:
            links.add(full)
    return list(links)

def extract_telugu_paragraphs(soup, min_len=100):
    """Extract all Telugu paragraphs from a page."""
    texts = []
    for p in soup.find_all(['p', 'div', 'article', 'section']):
        text = p.get_text(strip=True)
        if len(text) > min_len and is_telugu(text):
            texts.append(clean(text))
    # Deduplicate (shorter texts that are substrings of longer ones)
    texts.sort(key=len, reverse=True)
    unique = []
    for t in texts:
        if not any(t in u for u in unique):
            unique.append(t)
    return unique

def get_title(soup):
    for sel in ['h1', 'title']:
        el = soup.find(sel)
        if el:
            t = el.get_text(strip=True)
            if t: return t
    return ""

# ============ NEWS SCRAPING ============
def scrape_news():
    """Scrape real news from working Telugu news sites."""
    print("=" * 60)
    print("📰 SCRAPING REAL NEWS")
    print("=" * 60)
    
    sources = {
        "sakshi": {
            "name": "Sakshi",
            "start_urls": [
                "https://www.sakshi.com/telugu-news/national",
                "https://www.sakshi.com/telugu-news/andhra-pradesh",
                "https://www.sakshi.com/telugu-news/telangana",
                "https://www.sakshi.com/sports",
                "https://www.sakshi.com/telugu-news/business",
            ]
        },
        "ntnews": {
            "name": "Namaste Telangana",
            "start_urls": [
                "https://www.ntnews.com/telangana",
                "https://www.ntnews.com/andhra-pradesh",
                "https://www.ntnews.com/national",
                "https://www.ntnews.com/sports",
            ]
        },
        "eenadu": {
            "name": "Eenadu",
            "start_urls": [
                "https://www.eenadu.net/telugu-news/state/1",
                "https://www.eenadu.net/telugu-news/national/2",
                "https://www.eenadu.net/telugu-news/sports/8",
                "https://www.eenadu.net/telugu-news/business/3",
            ]
        }
    }
    
    all_articles = []
    
    for source_key, config in sources.items():
        print(f"\n--- {config['name']} ---")
        articles_from_source = []
        visited = set()
        article_links = set()
        
        # Phase 1: Collect links from category pages
        for start_url in config["start_urls"]:
            print(f"  Fetching: {start_url}")
            soup = fetch(start_url)
            if not soup:
                continue
            links = get_all_links(soup, start_url)
            # Filter: article links tend to be longer paths
            for link in links:
                path = urlparse(link).path
                # Skip homepage, category-level pages
                if path.count('/') >= 2 and len(path) > 20:
                    article_links.add(link)
            time.sleep(1)
        
        print(f"  Found {len(article_links)} candidate article links")
        
        # Phase 2: Visit article pages and extract Telugu content
        for url in list(article_links)[:30]:  # limit
            if url in visited:
                continue
            visited.add(url)
            
            soup = fetch(url)
            if not soup:
                continue
            
            title = get_title(soup)
            paragraphs = extract_telugu_paragraphs(soup, min_len=80)
            
            if paragraphs:
                content = " ".join(paragraphs)
                if len(content) > 200 and is_telugu(content):
                    article = {
                        "id": hashlib.md5(url.encode()).hexdigest()[:12],
                        "title": title,
                        "content": content[:2000],  # Cap at 2000 chars
                        "url": url,
                        "source": config["name"],
                        "category": "general",
                        "date_scraped": datetime.now().isoformat(),
                        "date_published": None
                    }
                    articles_from_source.append(article)
                    print(f"    ✓ [{len(articles_from_source)}] {title[:50]}...")
            
            time.sleep(1.0 + random.uniform(0, 0.5))
            
            if len(articles_from_source) >= 10:
                break
        
        print(f"  Got {len(articles_from_source)} articles from {config['name']}")
        all_articles.extend(articles_from_source)
    
    # Save
    out_dir = Path("data/domain/news/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "news_scraped_real.json"
    
    data = {
        "metadata": {
            "total_articles": len(all_articles),
            "sources": list(set(a["source"] for a in all_articles)),
            "scraped_at": datetime.now().isoformat(),
            "method": "Web Scraping (requests + BeautifulSoup)"
        },
        "articles": all_articles
    }
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Saved {len(all_articles)} real news articles to {out_path}")
    return all_articles


# ============ GOVERNMENT SCRAPING ============
def scrape_government():
    """Scrape real government content from Telangana/AP portals."""
    print("\n" + "=" * 60)
    print("🏛️ SCRAPING REAL GOVERNMENT DATA")
    print("=" * 60)
    
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    gov_urls = [
        ("https://www.telangana.gov.in", "Telangana Government Portal"),
        ("https://www.telangana.gov.in/about/telangana", "Telangana About"),
        ("https://www.telangana.gov.in/government/departments", "Telangana Departments"),
        ("https://www.ap.gov.in", "AP Government Portal"),
        ("https://www.ap.gov.in/know-andhra-pradesh/", "AP About"),
        ("https://irrigation.telangana.gov.in", "TS Irrigation"),
        ("https://finance.telangana.gov.in", "TS Finance"),
        ("https://www.aponline.gov.in", "AP Online"),
        ("https://cdse.telangana.gov.in", "TS Education"),
        ("https://hmda.telangana.gov.in", "HMDA"),
        ("https://www.ghmc.gov.in", "GHMC"),
        ("https://tsrtc.telangana.gov.in", "TSRTC"),
    ]
    
    documents = []
    
    for url, name in gov_urls:
        print(f"\n  Trying: {name} ({url})")
        soup = fetch(url, timeout=20)
        if not soup:
            continue
        
        title = get_title(soup)
        paragraphs = extract_telugu_paragraphs(soup, min_len=50)
        
        # Also get English content from gov sites (they're bilingual)
        all_text = soup.get_text(separator=" ", strip=True)
        all_text = clean(all_text)
        
        if paragraphs:
            content = " ".join(paragraphs)
        elif len(all_text) > 200:
            content = all_text[:2000]
        else:
            print(f"    ✗ No usable content")
            continue
        
        # Also try to find sub-pages
        links = get_all_links(soup, url)
        sub_docs = []
        
        for link in links[:10]:
            sub_soup = fetch(link, timeout=10)
            if not sub_soup:
                continue
            sub_title = get_title(sub_soup)
            sub_paras = extract_telugu_paragraphs(sub_soup, min_len=50)
            if sub_paras:
                sub_content = " ".join(sub_paras)
                if len(sub_content) > 100:
                    sub_docs.append({
                        "id": hashlib.md5(link.encode()).hexdigest()[:12],
                        "title": sub_title or name,
                        "content": sub_content[:2000],
                        "url": link,
                        "source": name,
                        "doc_type": "government_portal",
                        "department": "General",
                        "date_scraped": datetime.now().isoformat(),
                        "date_published": None
                    })
                    print(f"    ✓ Sub-page: {sub_title[:40] if sub_title else link[:40]}...")
            time.sleep(1)
            if len(sub_docs) >= 3:
                break
        
        doc = {
            "id": hashlib.md5(url.encode()).hexdigest()[:12],
            "title": title or name,
            "content": content[:2000],
            "url": url,
            "source": name,
            "doc_type": "government_portal",
            "department": "General",
            "date_scraped": datetime.now().isoformat(),
            "date_published": None
        }
        documents.append(doc)
        documents.extend(sub_docs)
        print(f"    ✓ Main + {len(sub_docs)} sub-pages from {name}")
        
        time.sleep(1.5)
    
    # Save
    out_dir = Path("data/domain/government/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gov_scraped_real.json"
    
    data = {
        "metadata": {
            "total_documents": len(documents),
            "sources": list(set(d["source"] for d in documents)),
            "scraped_at": datetime.now().isoformat(),
            "method": "Web Scraping (requests + BeautifulSoup)"
        },
        "documents": documents
    }
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Saved {len(documents)} real gov documents to {out_path}")
    return documents


# ============ LITERATURE SCRAPING ============
def scrape_literature():
    """Scrape real Telugu literature from Wikisource via API."""
    print("\n" + "=" * 60)
    print("📚 SCRAPING REAL LITERATURE (Telugu Wikisource)")
    print("=" * 60)
    
    api_url = "https://te.wikisource.org/w/api.php"
    
    categories = [
        "శతకములు",
        "తెలుగు_సాహిత్యం",
        "ప్రాచీన_తెలుగు_సాహిత్యం",
        "తెలుగు_రచయితలు",
        "ఆధునిక_తెలుగు_సాహిత్యం",
        "తెలుగు_కవులు",
        "తెలుగు_కథలు",
        "తెలుగు_కవిత్వం",
        "తెలుగు_పద్యములు",
        "పోతన_భాగవతము",
        "వేమన_పద్యములు",
    ]
    
    passages = []
    seen = set()
    
    for category in categories:
        print(f"\n  Category: {category}")
        
        # Get pages in category
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": 20,
            "format": "json"
        }
        
        try:
            r = requests.get(api_url, params=params, headers=HEADERS, timeout=10)
            data = r.json()
            pages = [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
            print(f"    Found {len(pages)} pages")
        except Exception as e:
            print(f"    Failed to get category: {type(e).__name__}")
            continue
        
        for page_title in pages[:8]:
            if page_title in seen:
                continue
            seen.add(page_title)
            
            # Fetch page content
            params = {
                "action": "query",
                "titles": page_title,
                "prop": "extracts",
                "explaintext": True,
                "format": "json"
            }
            
            try:
                r = requests.get(api_url, params=params, headers=HEADERS, timeout=10)
                data = r.json()
                
                page_data = list(data.get("query", {}).get("pages", {}).values())
                if not page_data or page_data[0].get("pageid") is None:
                    continue
                
                content = page_data[0].get("extract", "")
                content = clean(content)
                
                if len(content) > 100:
                    # Determine genre from category
                    genre = "prose"
                    if any(k in category for k in ["కవిత్వం", "పద్యములు", "శతకములు", "కవులు"]):
                        genre = "poetry"
                    elif "భాగవతము" in category:
                        genre = "epic"
                    elif "కథలు" in category:
                        genre = "prose"
                    
                    # Split long content into passages
                    if len(content) > 800:
                        chunks = [content[i:i+700] for i in range(0, min(len(content), 3500), 600)]
                    else:
                        chunks = [content]
                    
                    for idx, chunk in enumerate(chunks):
                        if len(chunk) > 100:
                            passage = {
                                "id": hashlib.md5(chunk.encode()).hexdigest()[:12],
                                "title": f"{page_title}" + (f" - భాగం {idx+1}" if len(chunks) > 1 else ""),
                                "content": chunk,
                                "source": "Telugu Wikisource",
                                "url": f"https://te.wikisource.org/wiki/{quote(page_title)}",
                                "author": None,
                                "genre": genre,
                                "work_title": page_title,
                                "date_scraped": datetime.now().isoformat()
                            }
                            passages.append(passage)
                            print(f"    ✓ [{len(passages)}] {page_title[:40]}...")
                
            except Exception as e:
                print(f"    Skipped: {page_title} - {type(e).__name__}")
                continue
            
            time.sleep(0.5)
        
        if len(passages) >= 30:
            break
    
    # Save
    out_dir = Path("data/domain/literature/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "literature_scraped_real.json"
    
    genre_dist = {}
    for p in passages:
        genre_dist[p["genre"]] = genre_dist.get(p["genre"], 0) + 1
    
    data = {
        "metadata": {
            "total_passages": len(passages),
            "genres": genre_dist,
            "sources": list(set(p["source"] for p in passages)),
            "collected_at": datetime.now().isoformat(),
            "method": "Wikisource API (MediaWiki)"
        },
        "passages": passages
    }
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Saved {len(passages)} real literature passages to {out_path}")
    return passages


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--news", action="store_true")
    parser.add_argument("--gov", action="store_true")
    parser.add_argument("--lit", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    
    if args.all or args.news:
        scrape_news()
    if args.all or args.gov:
        scrape_government()
    if args.all or args.lit:
        scrape_literature()
    
    if not any([args.all, args.news, args.gov, args.lit]):
        print("Usage: python scrape_real_data.py --all | --news | --gov | --lit")
