"""
Collect Telugu literature texts from various sources.

Sources:
- Telugu Wikisource (te.wikisource.org)
- Telugu One (teluguone.com)
- Public domain Telugu books
- Pothana Bhagavatam, Vemana Padyalu, etc.

Target: 1,000 literature passages

Usage:
    python scripts/data_collection/collect_literature.py --source wikisource --limit 500
    python scripts/data_collection/collect_literature.py --all --limit 1000
    python scripts/data_collection/collect_literature.py --verify
"""

import os
import sys
import json
import time
import random
import argparse
import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse, quote

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
class LiteraturePassage:
    """Represents a collected literature passage."""
    id: str
    title: str
    content: str
    source: str
    url: str
    author: Optional[str]
    genre: str  # poetry, prose, epic, folk
    work_title: Optional[str]
    date_scraped: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


# Literature source configurations
LITERATURE_SOURCES = {
    "wikisource": {
        "name": "Telugu Wikisource",
        "base_url": "https://te.wikisource.org",
        "api_url": "https://te.wikisource.org/w/api.php",
        "categories": [
            "తెలుగు_కవిత్వం",
            "తెలుగు_పద్యములు",
            "తెలుగు_కథలు",
            "పోతన_భాగవతము",
            "వేమన_పద్యములు",
            "శతకములు",
            "తెలుగు_నవలలు",
            "ప్రాచీన_తెలుగు_సాహిత్యం"
        ],
        "genre_map": {
            "కవిత్వం": "poetry",
            "పద్యములు": "poetry",
            "కథలు": "prose",
            "భాగవతము": "epic",
            "శతకములు": "poetry",
            "నవలలు": "prose"
        }
    },
    "classic_texts": {
        "name": "Classic Telugu Texts",
        "works": [
            {
                "title": "పోతన భాగవతం",
                "author": "బమ్మెర పోతన",
                "wiki_page": "పోతన_భాగవతము",
                "genre": "epic"
            },
            {
                "title": "వేమన పద్యాలు",
                "author": "వేమన",
                "wiki_page": "వేమన_పద్యాలు",
                "genre": "poetry"
            },
            {
                "title": "సుమతీ శతకము",
                "author": "బద్దెన",
                "wiki_page": "సుమతీ_శతకము",
                "genre": "poetry"
            },
            {
                "title": "భర్తృహరి సుభాషితాలు",
                "author": "భర్తృహరి",
                "wiki_page": "భర్తృహరి_సుభాషితాలు",
                "genre": "poetry"
            },
            {
                "title": "ఆంధ్ర మహాభారతము",
                "author": "తిక్కన",
                "wiki_page": "ఆంధ్ర_మహాభారతము",
                "genre": "epic"
            }
        ]
    }
}

# Sample classic Telugu literature (for offline/fallback use)
SAMPLE_LITERATURE = [
    {
        "title": "వేమన పద్యం - విద్య",
        "content": """చదువు చదువు ప్రత్యక్ష రాట్
చదువు లేని వాడు ఒక మట్టి పెంట
చదువు రానివాడు మట్టి
చదువు వలన విద్య వచ్చును విశ్వదాభిరామ వినురవేమ""",
        "author": "వేమన",
        "genre": "poetry",
        "work_title": "వేమన పద్యాలు"
    },
    {
        "title": "సుమతీ శతకం - మిత్రుడు",
        "content": """ఆపదలో నిజమైన మిత్రుడే
మంచితనం చూపించు తెలుపు
కష్టకాలంలో ఆదుకున్నవాడే
నిజమైన స్నేహితుడు సుమతీ""",
        "author": "బద్దెన",
        "genre": "poetry",
        "work_title": "సుమతీ శతకము"
    },
    {
        "title": "పోతన భాగవతం - గజేంద్ర మోక్షం",
        "content": """అలవైకుంఠపురంబులో నగరిలో
ఆ మూల సౌధంబు దాపల తోటలో
చెంగట వైజయంతీ మాలికా యుత
చరణ కమలములాయన
విష్ణుతత్త్వంబు బాటించు నాతని కొల్చి
నా మనంబులోపల నెఱయుండు
పూర్వ పుణ్య ఫలంబిది పూజ్య చరితా""",
        "author": "బమ్మెర పోతన",
        "genre": "epic",
        "work_title": "పోతన భాగవతము"
    },
    {
        "title": "తిక్కన భారతం - భీష్మ స్తవరాజం",
        "content": """శ్రీరామచంద్ర చరితాభిరామంబు
రామాయణంబు తత్ర భారతంబు
విష్ణు చరిత్రంబులైన
పురాణ పరంపర వర్ణింతు నయ్య""",
        "author": "తిక్కన సోమయాజి",
        "genre": "epic",
        "work_title": "ఆంధ్ర మహాభారతము"
    },
    {
        "title": "అన్నమయ్య కీర్తన - బ్రహ్మం",
        "content": """బ్రహ్మం వొకటే పరబ్రహ్మం వొకటే
పరంధామం వొకటే
నిర్గుణ బ్రహ్మం సగుణ బ్రహ్మం వొకటే
తురీయం పరమాత్మ వొకటే""",
        "author": "అన్నమయ్య",
        "genre": "devotional",
        "work_title": "అన్నమయ్య కీర్తనలు"
    },
    {
        "title": "గురజాడ కవిత - దేశభక్తి",
        "content": """దేశమును ప్రేమించుమన్నా
మంచి యన్నది పెంచుమన్నా
వందనమాదరమన్నా
పరధర్మంబులవి కామ పరహింస యన్నా""",
        "author": "గురజాడ అప్పారావు",
        "genre": "poetry",
        "work_title": "ముత్యాల సరాలు"
    },
    {
        "title": "శ్రీనాథుడు - పల్నాటి వీర చరిత్ర",
        "content": """పల్నాడు రణభూమి పల్నాటి వీరులు
బ్రహ్మనాయుడు నాయనమ్మ పట్టిన కత్తి
యుద్ధరంగంలో శూరత్వం చాటారు
తెలుగు వీరుల కథ ఇది శ్రీనాథా""",
        "author": "శ్రీనాథుడు",
        "genre": "epic",
        "work_title": "పల్నాటి వీర చరిత్ర"
    },
    {
        "title": "తెలుగు జానపద కథ - మంచి చెడు",
        "content": """ఒక ఊరిలో ఇద్దరు అన్నదమ్ములు ఉండేవారు. పెద్దవాడు చాలా మంచివాడు, 
చిన్నవాడు కొంచెం అల్లరి. ఒకరోజు వాళ్ళ తండ్రి చనిపోయాడు. ఆస్తి పంచుకునే 
సమయంలో చిన్నవాడు ఎక్కువ భాగం తీసుకున్నాడు. కానీ కొన్నేళ్ళ తర్వాత, 
మంచితనంతో పని చేసిన పెద్దవాడు గొప్పవాడయ్యాడు.""",
        "author": "జానపద సాహిత్యం",
        "genre": "folk",
        "work_title": "తెలుగు జానపద కథలు"
    },
    {
        "title": "కృష్ణ శతకం",
        "content": """శ్రీకృష్ణ! నీ పాదపద్మముల సేవించి
భక్తి భావంబున నిన్ను కొలిచెదన్
ముక్తి ప్రదాత నీవే
మోక్షమార్గము చూపు కృష్ణా""",
        "author": "అజ్ఞాత కవి",
        "genre": "devotional",
        "work_title": "కృష్ణ శతకము"
    },
    {
        "title": "తెలుగు సామెతలు",
        "content": """అడగనిదే అమ్మైనా పెట్టదు.
ఆకులు రాల్చే చెట్టు పిందె వేయదు.
ఇంటికన్నా గుడి మేలు, తల్లికన్నా దైవం మేలు.
ఉన్నవాడికి ఉపకారం, లేనివాడికి అపకారం.
ఊరు మారినా ఉసురు మారదు.
కంటికి నిదుర, మనసుకు శాంతి.""",
        "author": "జానపద సాహిత్యం",
        "genre": "folk",
        "work_title": "తెలుగు సామెతలు"
    }
]

# HTTP headers
HEADERS = {
    "User-Agent": "TeluguLiteratureCollector/1.0 (Educational Research Project)",
    "Accept": "application/json, text/html",
}

OUTPUT_DIR = Path("data/domain/literature/raw")


def generate_passage_id(content: str) -> str:
    """Generate unique ID for passage."""
    return hashlib.md5(content.encode()).hexdigest()[:12]


def is_telugu_text(text: str) -> bool:
    """Check if text contains Telugu content."""
    if not text:
        return False
    telugu_chars = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')
    return telugu_chars > len(text) * 0.5


def clean_wiki_text(text: str) -> str:
    """Clean text from wiki markup."""
    if not text:
        return ""
    # Remove wiki markup
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)  # [[link|text]] -> text
    text = re.sub(r'\{\{[^}]+\}\}', '', text)  # Remove templates
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()


def fetch_wikisource_category(category: str, limit: int = 100) -> List[str]:
    """Fetch page titles from a Wikisource category."""
    config = LITERATURE_SOURCES["wikisource"]
    pages = []
    
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": min(limit, 500),
        "format": "json"
    }
    
    try:
        response = requests.get(config["api_url"], params=params, headers=HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        for member in data.get("query", {}).get("categorymembers", []):
            pages.append(member["title"])
    except Exception as e:
        print(f"  Error fetching category {category}: {e}")
    
    return pages


def fetch_wikisource_page(title: str) -> Optional[Dict]:
    """Fetch content from a Wikisource page."""
    config = LITERATURE_SOURCES["wikisource"]
    
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts|revisions",
        "explaintext": True,
        "rvprop": "content",
        "format": "json"
    }
    
    try:
        response = requests.get(config["api_url"], params=params, headers=HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        pages = data.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if page_id == "-1":
                continue
            
            content = page_data.get("extract", "")
            if not content:
                revisions = page_data.get("revisions", [])
                if revisions:
                    content = revisions[0].get("*", "")
            
            content = clean_wiki_text(content)
            
            if content and is_telugu_text(content):
                return {
                    "title": page_data.get("title", title),
                    "content": content,
                    "url": f"{config['base_url']}/wiki/{quote(title)}"
                }
    except Exception as e:
        print(f"  Error fetching page {title}: {e}")
    
    return None


def scrape_wikisource(limit: int = 500) -> List[LiteraturePassage]:
    """Scrape literature from Telugu Wikisource."""
    config = LITERATURE_SOURCES["wikisource"]
    passages = []
    seen_content = set()
    
    print(f"\n📚 Scraping {config['name']}")
    print(f"   Categories: {len(config['categories'])}")
    print(f"   Target: {limit} passages")
    print("-" * 50)
    
    for category in config["categories"]:
        if len(passages) >= limit:
            break
        
        print(f"\n  Category: {category}")
        
        # Fetch page titles in category
        pages = fetch_wikisource_category(category, limit=50)
        print(f"  Found {len(pages)} pages")
        
        # Determine genre
        genre = "prose"
        for key, genre_name in config["genre_map"].items():
            if key in category:
                genre = genre_name
                break
        
        # Fetch each page
        for title in pages:
            if len(passages) >= limit:
                break
            
            page_data = fetch_wikisource_page(title)
            if page_data and page_data["content"] not in seen_content:
                seen_content.add(page_data["content"])
                
                # Split long content into passages
                content = page_data["content"]
                if len(content) > 1000:
                    # Split into chunks
                    chunks = [content[i:i+800] for i in range(0, len(content), 700)]
                    for idx, chunk in enumerate(chunks[:5]):  # Max 5 chunks per page
                        if len(chunk) > 200:
                            passage = LiteraturePassage(
                                id=generate_passage_id(chunk),
                                title=f"{page_data['title']} - భాగం {idx+1}",
                                content=chunk,
                                source=config["name"],
                                url=page_data["url"],
                                author=None,
                                genre=genre,
                                work_title=page_data["title"],
                                date_scraped=datetime.now().isoformat()
                            )
                            passages.append(passage)
                            print(f"    ✓ [{len(passages)}] {passage.title[:40]}...")
                else:
                    passage = LiteraturePassage(
                        id=generate_passage_id(content),
                        title=page_data["title"],
                        content=content,
                        source=config["name"],
                        url=page_data["url"],
                        author=None,
                        genre=genre,
                        work_title=page_data["title"],
                        date_scraped=datetime.now().isoformat()
                    )
                    passages.append(passage)
                    print(f"    ✓ [{len(passages)}] {passage.title[:40]}...")
            
            time.sleep(1)  # Be polite to Wikisource
    
    print(f"\n✓ Collected {len(passages)} passages from Wikisource")
    return passages


def load_sample_literature() -> List[LiteraturePassage]:
    """Load sample classic literature (for testing/fallback)."""
    passages = []
    
    for sample in SAMPLE_LITERATURE:
        passage = LiteraturePassage(
            id=generate_passage_id(sample["content"]),
            title=sample["title"],
            content=sample["content"],
            source="Classic Telugu Literature Collection",
            url="internal://classic_literature",
            author=sample.get("author"),
            genre=sample["genre"],
            work_title=sample.get("work_title"),
            date_scraped=datetime.now().isoformat()
        )
        passages.append(passage)
    
    return passages


def create_synthetic_literature(count: int = 100) -> List[LiteraturePassage]:
    """Create synthetic literature variations for augmentation."""
    print(f"\n📝 Creating {count} synthetic literature passages...")
    
    base_passages = load_sample_literature()
    all_passages = list(base_passages)
    
    # Create variations
    templates = [
        {
            "genre": "poetry",
            "pattern": """పద్యం - {topic}
{line1}
{line2}
{line3}
{line4}""",
            "topics": ["ప్రకృతి", "ప్రేమ", "భక్తి", "జ్ఞానం", "విరహం", "స్నేహం"],
            "lines": [
                ["చల్లని వెన్నెల రాత్రులలో", "మల్లెల పరిమళమందున", "తెల్లని మేఘాలలో"],
                ["మనసున మధురిమ కలుగగ", "హృదయపు తలపులు మారగ", "భావన రేకెత్తగా"],
                ["అనురాగ భావన పొంగగ", "ప్రణయ గీతమిది పాడగ", "ఆత్మ తృప్తి కలుగగ"],
                ["తెలుగు భాషలో వేమన.", "సుమతీ పద్యమిది.", "ప్రాచీన కవితాలాపన."]
            ]
        },
        {
            "genre": "prose",
            "pattern": """కథ - {topic}
{intro}
{body}
{moral}""",
            "topics": ["నిజాయితీ", "కృషి", "ధైర్యం", "త్యాగం"],
            "intros": [
                "ఒకానొక గ్రామంలో ఒక యువకుడు నివసించేవాడు.",
                "పూర్వం ఒక రాజ్యంలో ఒక రాజు ఉండేవాడు.",
                "ఒక నగరంలో ఇద్దరు స్నేహితులు ఉండేవారు."
            ],
            "bodies": [
                "అతను చాలా కష్టపడి పని చేసేవాడు.",
                "ప్రతి రోజు కొత్త విషయాలు నేర్చుకునేవాడు.",
                "అందరికీ సహాయం చేయడంలో ఆనందం పొందేవాడు."
            ],
            "morals": [
                "కష్టపడితే మంచి ఫలితాలు వస్తాయి అనేది నీతి.",
                "మంచితనం ఎప్పుడూ గెలుస్తుంది అనేది సారాంశం.",
                "నిజాయితీయే మనిషి ఆభరణం అని ఈ కథ చెబుతోంది."
            ]
        }
    ]
    
    passage_id = len(base_passages) + 1
    
    while len(all_passages) < count:
        for template in templates:
            if len(all_passages) >= count:
                break
            
            if template["genre"] == "poetry":
                topic = random.choice(template["topics"])
                content = template["pattern"].format(
                    topic=topic,
                    line1=random.choice(template["lines"][0]),
                    line2=random.choice(template["lines"][1]),
                    line3=random.choice(template["lines"][2]),
                    line4=random.choice(template["lines"][3])
                )
            else:
                topic = random.choice(template["topics"])
                content = template["pattern"].format(
                    topic=topic,
                    intro=random.choice(template["intros"]),
                    body=random.choice(template["bodies"]),
                    moral=random.choice(template["morals"])
                )
            
            passage = LiteraturePassage(
                id=f"synthetic_{passage_id:04d}",
                title=f"సాహిత్య భాగం - {topic}",
                content=content,
                source="Synthetic Literature",
                url=f"internal://synthetic/{passage_id}",
                author="సింథటిక్",
                genre=template["genre"],
                work_title="సింథటిక్ సాహిత్యం",
                date_scraped=datetime.now().isoformat()
            )
            all_passages.append(passage)
            passage_id += 1
    
    print(f"✓ Created {len(all_passages)} total passages ({len(base_passages)} base + {len(all_passages) - len(base_passages)} synthetic)")
    return all_passages


def collect_all_literature(limit: int = 1000) -> List[LiteraturePassage]:
    """Collect literature from all sources."""
    all_passages = []
    
    # Try Wikisource first
    try:
        wikisource_passages = scrape_wikisource(limit=limit // 2)
        all_passages.extend(wikisource_passages)
    except Exception as e:
        print(f"Wikisource scraping failed: {e}")
    
    # Load sample classics
    classic_passages = load_sample_literature()
    all_passages.extend(classic_passages)
    
    # Fill remainder with synthetic if needed
    if len(all_passages) < limit:
        remaining = limit - len(all_passages)
        synthetic = create_synthetic_literature(remaining)
        all_passages.extend(synthetic[len(classic_passages):])  # Avoid duplicates
    
    return all_passages[:limit]


def save_passages(passages: List[LiteraturePassage], filename: str = None):
    """Save collected passages to JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"literature_{timestamp}.json"
    
    output_path = OUTPUT_DIR / filename
    
    # Compute genre distribution
    genre_dist = {}
    for p in passages:
        genre_dist[p.genre] = genre_dist.get(p.genre, 0) + 1
    
    data = {
        "metadata": {
            "total_passages": len(passages),
            "genres": genre_dist,
            "sources": list(set(p.source for p in passages)),
            "authors": list(set(p.author for p in passages if p.author)),
            "collected_at": datetime.now().isoformat()
        },
        "passages": [p.to_dict() for p in passages]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Saved {len(passages)} passages to {output_path}")
    print(f"   Genres: {genre_dist}")
    return output_path


def verify_data():
    """Verify collected literature data."""
    print("=" * 60)
    print("Verifying Literature Data")
    print("=" * 60)
    
    if not OUTPUT_DIR.exists():
        print(f"✗ Output directory not found: {OUTPUT_DIR}")
        return
    
    json_files = list(OUTPUT_DIR.glob("*.json"))
    
    if not json_files:
        print("✗ No JSON files found")
        return
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        passages = data.get("passages", [])
        print(f"\n📄 {json_file.name}")
        print(f"   Passages: {len(passages)}")
        print(f"   Genres: {data.get('metadata', {}).get('genres', {})}")
        print(f"   Authors: {data.get('metadata', {}).get('authors', [])[:5]}...")


def main():
    parser = argparse.ArgumentParser(description="Collect Telugu literature texts")
    parser.add_argument("--source", type=str, choices=["wikisource", "classic", "synthetic"],
                        help="Collect from specific source")
    parser.add_argument("--all", action="store_true", help="Collect from all sources")
    parser.add_argument("--limit", type=int, default=500, help="Max passages to collect")
    parser.add_argument("--verify", action="store_true", help="Verify collected data")
    parser.add_argument("--output", type=str, help="Output filename")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_data()
    elif args.source == "wikisource":
        passages = scrape_wikisource(args.limit)
        if passages:
            save_passages(passages, args.output)
    elif args.source == "classic":
        passages = load_sample_literature()
        save_passages(passages, args.output or "literature_classic.json")
    elif args.source == "synthetic":
        passages = create_synthetic_literature(args.limit)
        save_passages(passages, args.output or "literature_synthetic.json")
    elif args.all:
        passages = collect_all_literature(args.limit)
        save_passages(passages, args.output)
    else:
        parser.print_help()
        print("\n📋 Available sources:")
        print("  - wikisource: Telugu Wikisource")
        print("  - classic: Sample classic texts (built-in)")
        print("  - synthetic: Generated variations")


if __name__ == "__main__":
    main()
