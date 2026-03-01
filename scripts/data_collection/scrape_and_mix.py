"""
Scrape real data from working sources and mix with existing synthetic data.
Handles all errors gracefully and saves results.
"""
import json
import time
import random
import hashlib
import re
import sys
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import quote

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

def fetch(url, timeout=8):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, verify=False)
        r.encoding = 'utf-8'
        return BeautifulSoup(r.text, 'lxml')
    except:
        return None

def get_all_links(soup, base_url):
    from urllib.parse import urljoin, urlparse
    links = set()
    for a in soup.find_all('a', href=True):
        full = urljoin(base_url, a['href'])
        parsed = urlparse(full)
        base_parsed = urlparse(base_url)
        if parsed.netloc == base_parsed.netloc and '#' not in full:
            links.add(full)
    return list(links)

def extract_telugu_paragraphs(soup, min_len=80):
    texts = []
    for p in soup.find_all(['p', 'div', 'article']):
        text = p.get_text(strip=True)
        if len(text) > min_len and is_telugu(text):
            texts.append(clean(text))
    texts.sort(key=len, reverse=True)
    unique = []
    for t in texts:
        if not any(t in u for u in unique):
            unique.append(t)
    return unique


# ========== SCRAPE NEWS ==========
def scrape_news():
    print("=" * 60)
    print("SCRAPING NEWS")
    print("=" * 60)
    
    sources = {
        "sakshi": {
            "name": "Sakshi",
            "urls": [
                "https://www.sakshi.com/telugu-news/national",
                "https://www.sakshi.com/telugu-news/andhra-pradesh",
                "https://www.sakshi.com/telugu-news/telangana",
                "https://www.sakshi.com/sports",
            ]
        },
        "ntnews": {
            "name": "Namaste Telangana",
            "urls": [
                "https://www.ntnews.com/telangana",
                "https://www.ntnews.com/andhra-pradesh",
                "https://www.ntnews.com/national",
            ]
        },
        "eenadu": {
            "name": "Eenadu",
            "urls": [
                "https://www.eenadu.net/telugu-news/state/1",
                "https://www.eenadu.net/telugu-news/national/2",
                "https://www.eenadu.net/telugu-news/sports/8",
            ]
        }
    }
    
    articles = []
    
    for source_key, config in sources.items():
        print(f"\n--- {config['name']} ---")
        visited = set()
        source_articles = []
        article_links = set()
        
        # Collect links
        for url in config["urls"]:
            print(f"  Fetching: {url}")
            soup = fetch(url)
            if not soup:
                print(f"    SKIP (failed)")
                continue
            links = get_all_links(soup, url)
            from urllib.parse import urlparse
            for link in links:
                path = urlparse(link).path
                if path.count('/') >= 2 and len(path) > 15:
                    article_links.add(link)
            time.sleep(0.8)
        
        print(f"  {len(article_links)} candidate links")
        
        # Visit articles
        for url in list(article_links)[:25]:
            if url in visited:
                continue
            visited.add(url)
            
            try:
                soup = fetch(url)
                if not soup:
                    continue
                
                title_el = soup.find('h1')
                title = title_el.get_text(strip=True) if title_el else ""
                paragraphs = extract_telugu_paragraphs(soup)
                
                if paragraphs:
                    content = " ".join(paragraphs)
                    if len(content) > 150 and is_telugu(content):
                        source_articles.append({
                            "id": hashlib.md5(url.encode()).hexdigest()[:12],
                            "title": title[:200],
                            "content": content[:2000],
                            "url": url,
                            "source": config["name"],
                            "category": "general",
                            "date_scraped": datetime.now().isoformat(),
                            "date_published": None
                        })
                        print(f"    + [{len(source_articles)}] {title[:50]}...")
                
                time.sleep(0.8 + random.uniform(0, 0.3))
                if len(source_articles) >= 8:
                    break
            except Exception as e:
                print(f"    err: {type(e).__name__}")
                continue
        
        print(f"  => {len(source_articles)} articles")
        articles.extend(source_articles)
    
    return articles


# ========== SCRAPE GOVERNMENT ==========
def scrape_government():
    print("\n" + "=" * 60)
    print("SCRAPING GOVERNMENT")
    print("=" * 60)
    
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    targets = [
        ("https://www.telangana.gov.in", "Telangana State Portal"),
        ("https://www.telangana.gov.in/about/telangana", "Telangana About"),
        ("https://www.ap.gov.in", "Andhra Pradesh Portal"),
        ("https://irrigation.telangana.gov.in", "TS Irrigation Dept"),
        ("https://finance.telangana.gov.in", "TS Finance Dept"),
        ("https://www.ghmc.gov.in", "GHMC Hyderabad"),
        ("https://tsrtc.telangana.gov.in", "TSRTC"),
        ("https://cdse.telangana.gov.in", "TS Education Dept"),
        ("https://hmda.telangana.gov.in", "HMDA"),
        ("https://www.aponline.gov.in", "AP Online"),
    ]
    
    documents = []
    
    for url, name in targets:
        print(f"\n  {name} ({url})")
        try:
            soup = fetch(url, timeout=10)
            if not soup:
                print(f"    SKIP (failed)")
                continue
            
            title_el = soup.find('h1') or soup.find('title')
            title = title_el.get_text(strip=True) if title_el else name
            
            # Get all text
            all_text = clean(soup.get_text(separator=" ", strip=True))
            paragraphs = extract_telugu_paragraphs(soup, min_len=40)
            
            if paragraphs:
                content = " ".join(paragraphs)
            elif len(all_text) > 300:
                content = all_text[:2000]
            else:
                print(f"    SKIP (no content)")
                continue
            
            documents.append({
                "id": hashlib.md5(url.encode()).hexdigest()[:12],
                "title": title[:200],
                "content": content[:2000],
                "url": url,
                "source": name,
                "doc_type": "government_portal",
                "department": "General",
                "date_scraped": datetime.now().isoformat(),
                "date_published": None
            })
            print(f"    + {title[:50]}... ({len(content)} chars)")
            
            # Try a few sub-pages
            links = get_all_links(soup, url)
            sub_count = 0
            for link in links[:8]:
                try:
                    sub_soup = fetch(link, timeout=6)
                    if not sub_soup:
                        continue
                    sub_paras = extract_telugu_paragraphs(sub_soup, min_len=40)
                    if sub_paras:
                        sub_content = " ".join(sub_paras)
                        if len(sub_content) > 100:
                            sub_title_el = sub_soup.find('h1') or sub_soup.find('title')
                            sub_title = sub_title_el.get_text(strip=True) if sub_title_el else ""
                            documents.append({
                                "id": hashlib.md5(link.encode()).hexdigest()[:12],
                                "title": sub_title[:200] or name,
                                "content": sub_content[:2000],
                                "url": link,
                                "source": name,
                                "doc_type": "government_portal",
                                "department": "General",
                                "date_scraped": datetime.now().isoformat(),
                                "date_published": None
                            })
                            sub_count += 1
                            print(f"      sub: {sub_title[:40]}...")
                    time.sleep(0.5)
                    if sub_count >= 2:
                        break
                except:
                    continue
            
            time.sleep(1)
        except Exception as e:
            print(f"    ERR: {type(e).__name__}")
            continue
    
    return documents


# ========== SCRAPE LITERATURE (Wikisource API) ==========
def scrape_literature():
    print("\n" + "=" * 60)
    print("SCRAPING LITERATURE (Wikisource API)")
    print("=" * 60)
    
    api_url = "https://te.wikisource.org/w/api.php"
    
    categories = [
        "శతకములు",
        "తెలుగు_సాహిత్యం",
        "తెలుగు_రచయితలు",
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
        if len(passages) >= 40:
            break
        
        print(f"\n  Category: {category}")
        try:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmlimit": 20,
                "format": "json"
            }
            r = requests.get(api_url, params=params, headers=HEADERS, timeout=8)
            pages = [m["title"] for m in r.json().get("query", {}).get("categorymembers", [])]
            print(f"    Found {len(pages)} pages")
        except Exception as e:
            print(f"    Failed: {type(e).__name__}")
            continue
        
        for page_title in pages[:6]:
            if page_title in seen or len(passages) >= 40:
                break
            seen.add(page_title)
            
            try:
                params = {
                    "action": "query",
                    "titles": page_title,
                    "prop": "extracts",
                    "explaintext": True,
                    "format": "json"
                }
                r = requests.get(api_url, params=params, headers=HEADERS, timeout=8)
                page_data = list(r.json().get("query", {}).get("pages", {}).values())
                if not page_data or page_data[0].get("pageid") is None:
                    continue
                
                content = clean(page_data[0].get("extract", ""))
                if len(content) < 80:
                    continue
                
                genre = "prose"
                if any(k in category for k in ["కవిత్వం", "పద్యములు", "శతకములు", "కవులు"]):
                    genre = "poetry"
                elif "భాగవతము" in category:
                    genre = "epic"
                
                # Split long text into passages
                if len(content) > 800:
                    chunks = [content[i:i+700] for i in range(0, min(len(content), 3500), 600)]
                else:
                    chunks = [content]
                
                for idx, chunk in enumerate(chunks):
                    if len(chunk) > 80:
                        passages.append({
                            "id": hashlib.md5(chunk.encode()).hexdigest()[:12],
                            "title": f"{page_title}" + (f" - part {idx+1}" if len(chunks) > 1 else ""),
                            "content": chunk,
                            "source": "Telugu Wikisource",
                            "url": f"https://te.wikisource.org/wiki/{quote(page_title)}",
                            "author": None,
                            "genre": genre,
                            "work_title": page_title,
                            "date_scraped": datetime.now().isoformat()
                        })
                        print(f"    + [{len(passages)}] {page_title[:40]}...")
            
            except Exception as e:
                print(f"    skip: {page_title[:30]} ({type(e).__name__})")
                continue
            
            time.sleep(0.5)
    
    return passages


# ========== MIX REAL + SYNTHETIC ==========
def mix_data(news_articles, gov_documents, lit_passages):
    print("\n" + "=" * 60)
    print("MIXING REAL + SYNTHETIC DATA")
    print("=" * 60)
    
    base_dir = Path("data/domain")
    
    # --- NEWS ---
    news_file = base_dir / "news/raw/news_scaled.json"
    with open(news_file, 'r', encoding='utf-8') as f:
        news_data = json.load(f)
    
    existing_news = news_data.get("articles", [])
    print(f"\nNews: {len(existing_news)} synthetic + {len(news_articles)} real")
    
    # Insert real articles throughout
    combined_news = list(existing_news)
    for i, article in enumerate(news_articles):
        insert_pos = random.randint(0, len(combined_news))
        combined_news.insert(insert_pos, article)
    
    news_data["articles"] = combined_news
    news_data["metadata"]["total_articles"] = len(combined_news)
    news_data["metadata"]["source"] = "Mixed (Web Scraping + Synthetic Generation)"
    news_data["metadata"]["real_sources"] = list(set(a["source"] for a in news_articles)) if news_articles else []
    news_data["metadata"]["real_article_count"] = len(news_articles)
    news_data["metadata"]["synthetic_article_count"] = len(existing_news)
    news_data["metadata"]["last_updated"] = datetime.now().isoformat()
    
    with open(news_file, 'w', encoding='utf-8') as f:
        json.dump(news_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(combined_news)} articles to {news_file}")
    
    # Also save real-only file
    real_news_file = base_dir / "news/raw/news_scraped_real.json"
    with open(real_news_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "total_articles": len(news_articles),
                "sources": list(set(a["source"] for a in news_articles)),
                "scraped_at": datetime.now().isoformat(),
                "method": "Web Scraping (requests + BeautifulSoup)"
            },
            "articles": news_articles
        }, f, ensure_ascii=False, indent=2)
    
    # --- GOVERNMENT ---
    gov_file = base_dir / "government/raw/gov_scaled.json"
    with open(gov_file, 'r', encoding='utf-8') as f:
        gov_data = json.load(f)
    
    existing_gov = gov_data.get("documents", [])
    print(f"\nGov: {len(existing_gov)} synthetic + {len(gov_documents)} real")
    
    combined_gov = list(existing_gov)
    for doc in gov_documents:
        insert_pos = random.randint(0, len(combined_gov))
        combined_gov.insert(insert_pos, doc)
    
    gov_data["documents"] = combined_gov
    gov_data["metadata"]["total_documents"] = len(combined_gov)
    gov_data["metadata"]["source"] = "Mixed (Web Scraping + Synthetic Generation)"
    gov_data["metadata"]["real_sources"] = list(set(d["source"] for d in gov_documents)) if gov_documents else []
    gov_data["metadata"]["real_document_count"] = len(gov_documents)
    gov_data["metadata"]["synthetic_document_count"] = len(existing_gov)
    gov_data["metadata"]["last_updated"] = datetime.now().isoformat()
    
    with open(gov_file, 'w', encoding='utf-8') as f:
        json.dump(gov_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(combined_gov)} documents to {gov_file}")
    
    # Also save real-only file
    real_gov_file = base_dir / "government/raw/gov_scraped_real.json"
    with open(real_gov_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "total_documents": len(gov_documents),
                "sources": list(set(d["source"] for d in gov_documents)),
                "scraped_at": datetime.now().isoformat(),
                "method": "Web Scraping (requests + BeautifulSoup)"
            },
            "documents": gov_documents
        }, f, ensure_ascii=False, indent=2)
    
    # --- LITERATURE ---
    lit_file = base_dir / "literature/raw/lit_scaled.json"
    with open(lit_file, 'r', encoding='utf-8') as f:
        lit_data = json.load(f)
    
    existing_lit = lit_data.get("passages", [])
    print(f"\nLit: {len(existing_lit)} synthetic + {len(lit_passages)} real")
    
    combined_lit = list(existing_lit)
    for passage in lit_passages:
        insert_pos = random.randint(0, len(combined_lit))
        combined_lit.insert(insert_pos, passage)
    
    genre_dist = {}
    for p in combined_lit:
        g = p.get("genre", "unknown")
        genre_dist[g] = genre_dist.get(g, 0) + 1
    
    lit_data["passages"] = combined_lit
    lit_data["metadata"]["total_passages"] = len(combined_lit)
    lit_data["metadata"]["source"] = "Mixed (Wikisource API + Synthetic Generation)"
    lit_data["metadata"]["real_sources"] = list(set(p["source"] for p in lit_passages)) if lit_passages else []
    lit_data["metadata"]["real_passage_count"] = len(lit_passages)
    lit_data["metadata"]["synthetic_passage_count"] = len(existing_lit)
    lit_data["metadata"]["genres"] = genre_dist
    lit_data["metadata"]["last_updated"] = datetime.now().isoformat()
    
    with open(lit_file, 'w', encoding='utf-8') as f:
        json.dump(lit_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(combined_lit)} passages to {lit_file}")
    
    # Also save real-only file
    real_lit_file = base_dir / "literature/raw/literature_scraped_real.json"
    with open(real_lit_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "total_passages": len(lit_passages),
                "genres": {p.get("genre", "unknown"): 0 for p in lit_passages},
                "sources": list(set(p["source"] for p in lit_passages)),
                "collected_at": datetime.now().isoformat(),
                "method": "Wikisource API (MediaWiki)"
            },
            "passages": lit_passages
        }, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  News:       {len(news_articles):3d} real + {len(existing_news):3d} synthetic = {len(combined_news):3d} total")
    print(f"  Government: {len(gov_documents):3d} real + {len(existing_gov):3d} synthetic = {len(combined_gov):3d} total")
    print(f"  Literature: {len(lit_passages):3d} real + {len(existing_lit):3d} synthetic = {len(combined_lit):3d} total")
    total_real = len(news_articles) + len(gov_documents) + len(lit_passages)
    total_synth = len(existing_news) + len(existing_gov) + len(existing_lit)
    print(f"  TOTAL:      {total_real:3d} real + {total_synth:3d} synthetic = {total_real + total_synth:3d} total")


if __name__ == "__main__":
    # Step 1: Scrape
    news = scrape_news()
    gov = scrape_government()
    lit = scrape_literature()
    
    # Step 2: Mix
    mix_data(news, gov, lit)
    
    print("\nDone!")
