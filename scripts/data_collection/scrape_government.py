"""
Scrape Telugu content from Telangana and Andhra Pradesh government portals.

Sources:
- Telangana State Portal (telangana.gov.in)
- Andhra Pradesh Portal (ap.gov.in)
- GOIR (Government Orders) 
- Press Releases
- Scheme/Policy documents

Target: 2,000 government documents

Usage:
    python scripts/data_collection/scrape_government.py --source telangana --limit 500
    python scripts/data_collection/scrape_government.py --all --limit 2000
    python scripts/data_collection/scrape_government.py --verify
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
class GovDocument:
    """Represents a scraped government document."""
    id: str
    title: str
    content: str
    url: str
    source: str
    doc_type: str  # press_release, go, scheme, policy
    department: str
    date_scraped: str
    date_published: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


# Government portal configurations
GOV_SOURCES = {
    "telangana": {
        "name": "Telangana Government",
        "base_url": "https://www.telangana.gov.in",
        "sections": {
            "press_releases": {
                "url": "/news",
                "selector": "div.news-item a, article a",
                "type": "press_release"
            },
            "departments": {
                "url": "/government/departments",
                "selector": "div.dept-item a, a.department-link",
                "type": "department_info"
            },
            "schemes": {
                "url": "/government/schemes",
                "selector": "div.scheme-item a, a.scheme-link",
                "type": "scheme"
            }
        },
        "title_selector": "h1, h2.title, .page-title",
        "content_selector": "div.content, article, div.main-content, div.body-content",
        "date_selector": "time, .date, span.published"
    },
    "ap": {
        "name": "Andhra Pradesh Government",
        "base_url": "https://www.ap.gov.in",
        "sections": {
            "press_releases": {
                "url": "/Home/LatestNews",
                "selector": "div.news-list a, table a",
                "type": "press_release"
            },
            "go_orders": {
                "url": "/Home/GoList", 
                "selector": "table a, div.go-item a",
                "type": "government_order"
            },
            "schemes": {
                "url": "/Schemes",
                "selector": "div.scheme a, a.scheme-link",
                "type": "scheme"
            }
        },
        "title_selector": "h1, h2, .page-header h1",
        "content_selector": "div.content-area, div.main-content, article",
        "date_selector": "span.date, time, td.date"
    },
    "ts_goir": {
        "name": "Telangana GOIR",
        "base_url": "https://goir.telangana.gov.in",
        "sections": {
            "orders": {
                "url": "/pdfshow.aspx",
                "selector": "table a, div.go-list a",
                "type": "government_order"
            }
        },
        "title_selector": "h1, h2, .title",
        "content_selector": "div.content, div.order-text",
        "date_selector": ".date, span.order-date"
    },
    "meeseva": {
        "name": "Telangana Meeseva",
        "base_url": "https://ts.meeseva.telangana.gov.in",
        "sections": {
            "services": {
                "url": "/meeseva/home.htm",
                "selector": "div.service a, a.service-link",
                "type": "citizen_service"
            }
        },
        "title_selector": "h1, h2, .service-title",
        "content_selector": "div.service-content, div.description",
        "date_selector": ".date"
    }
}

# HTTP headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "te,en-US;q=0.9,en;q=0.8",
}

OUTPUT_DIR = Path("data/domain/government/raw")


def generate_doc_id(url: str) -> str:
    """Generate unique ID for document."""
    return hashlib.md5(url.encode()).hexdigest()[:12]


def is_telugu_text(text: str) -> bool:
    """Check if text contains Telugu content."""
    if not text:
        return False
    telugu_chars = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')
    # Government docs may be bilingual, so lower threshold
    return telugu_chars > len(text) * 0.15 or len(text) < 100


def extract_telugu_content(text: str) -> str:
    """Extract Telugu portions from bilingual text."""
    if not text:
        return ""
    
    # Split into sentences/segments
    segments = re.split(r'[.।\n]+', text)
    
    telugu_segments = []
    for seg in segments:
        seg = seg.strip()
        if seg:
            # Check if segment has Telugu
            telugu_ratio = sum(1 for c in seg if '\u0C00' <= c <= '\u0C7F') / max(len(seg), 1)
            if telugu_ratio > 0.3:
                telugu_segments.append(seg)
    
    return " ".join(telugu_segments)


def clean_text(text: str) -> str:
    """Clean document text."""
    if not text:
        return ""
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove common web garbage
    garbage = ["Download PDF", "Print", "Share", "Home >", "Skip to content"]
    for g in garbage:
        text = text.replace(g, "")
    return text.strip()


def fetch_page(url: str, retries: int = 3) -> Optional[BeautifulSoup]:
    """Fetch and parse page with retries."""
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=20, verify=False)
            response.raise_for_status()
            response.encoding = 'utf-8'
            return BeautifulSoup(response.text, 'lxml')
        except requests.RequestException as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)
    return None


def scrape_document(url: str, config: Dict, doc_type: str) -> Optional[GovDocument]:
    """Scrape a single government document."""
    soup = fetch_page(url)
    if not soup:
        return None
    
    # Extract title
    title_elem = soup.select_one(config["title_selector"])
    title = title_elem.get_text(strip=True) if title_elem else ""
    
    # Extract content
    content_elem = soup.select_one(config["content_selector"])
    if content_elem:
        # Remove scripts, styles, nav elements
        for tag in content_elem.find_all(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        content = content_elem.get_text(separator=" ", strip=True)
    else:
        # Fallback: get main body text
        body = soup.find('body')
        content = body.get_text(separator=" ", strip=True) if body else ""
    
    content = clean_text(content)
    
    # Extract Telugu content if bilingual
    if not is_telugu_text(content):
        telugu_content = extract_telugu_content(content)
        if telugu_content:
            content = telugu_content
    
    # Skip if too short
    if len(content) < 100:
        return None
    
    # Extract date
    date_elem = soup.select_one(config["date_selector"])
    date_published = date_elem.get_text(strip=True) if date_elem else None
    
    # Try to identify department from URL or content
    department = "General"
    dept_keywords = {
        "health": "వైద్య ఆరోగ్య శాఖ",
        "education": "విద్యా శాఖ",
        "finance": "ఆర్థిక శాఖ",
        "agriculture": "వ్యవసాయ శాఖ",
        "revenue": "రెవెన్యూ శాఖ",
        "it": "IT శాఖ",
        "police": "పోలీసు శాఖ"
    }
    for key, telugu_name in dept_keywords.items():
        if key in url.lower() or key in content.lower():
            department = telugu_name
            break
    
    return GovDocument(
        id=generate_doc_id(url),
        title=clean_text(title),
        content=content,
        url=url,
        source=config["name"],
        doc_type=doc_type,
        department=department,
        date_scraped=datetime.now().isoformat(),
        date_published=date_published
    )


def scrape_source(source_key: str, limit: int = 500, delay: float = 2.0) -> List[GovDocument]:
    """Scrape documents from a government source."""
    if source_key not in GOV_SOURCES:
        print(f"Unknown source: {source_key}")
        return []
    
    config = GOV_SOURCES[source_key]
    documents = []
    seen_urls = set()
    
    print(f"\n🏛️ Scraping {config['name']}")
    print(f"   Base URL: {config['base_url']}")
    print(f"   Target: {limit} documents")
    print("-" * 50)
    
    # Disable SSL warnings for gov sites with cert issues
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    for section_name, section_config in config["sections"].items():
        if len(documents) >= limit:
            break
        
        section_url = config["base_url"] + section_config["url"]
        print(f"\n  Section: {section_name}")
        print(f"  URL: {section_url}")
        
        soup = fetch_page(section_url)
        if not soup:
            print(f"  ✗ Failed to fetch section page")
            continue
        
        # Find document links
        links = []
        for a_tag in soup.select(section_config["selector"]):
            href = a_tag.get("href", "")
            if href:
                full_url = urljoin(config["base_url"], href)
                if full_url not in seen_urls:
                    links.append(full_url)
                    seen_urls.add(full_url)
        
        print(f"  Found {len(links)} document links")
        
        # Scrape each document
        section_docs = 0
        for url in links[:limit - len(documents)]:
            doc = scrape_document(url, config, section_config["type"])
            if doc:
                documents.append(doc)
                section_docs += 1
                print(f"    ✓ [{len(documents)}] {doc.title[:40]}...")
            
            time.sleep(delay + random.uniform(0, 1))
        
        print(f"  Scraped {section_docs} documents from {section_name}")
    
    print(f"\n✓ Total: {len(documents)} documents from {config['name']}")
    return documents


def scrape_all_sources(limit: int = 2000, delay: float = 2.0) -> List[GovDocument]:
    """Scrape from all government sources."""
    all_docs = []
    per_source_limit = limit // len(GOV_SOURCES)
    
    for source_key in GOV_SOURCES:
        docs = scrape_source(source_key, per_source_limit, delay)
        all_docs.extend(docs)
    
    return all_docs


def save_documents(documents: List[GovDocument], filename: str = None):
    """Save scraped documents to JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gov_documents_{timestamp}.json"
    
    output_path = OUTPUT_DIR / filename
    
    data = {
        "metadata": {
            "total_documents": len(documents),
            "sources": list(set(d.source for d in documents)),
            "doc_types": list(set(d.doc_type for d in documents)),
            "departments": list(set(d.department for d in documents)),
            "scraped_at": datetime.now().isoformat()
        },
        "documents": [d.to_dict() for d in documents]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Saved {len(documents)} documents to {output_path}")
    return output_path


def create_synthetic_gov_data():
    """Create synthetic government data for testing when scraping fails."""
    print("\n📝 Creating synthetic government data for testing...")
    
    # Sample Telugu government content templates with diverse content
    templates = [
        {
            "title": "తెలంగాణ ప్రభుత్వ ఉత్తర్వులు - {dept_name}",
            "content": """తెలంగాణ ప్రభుత్వం {dept_name} కొత్త మార్గదర్శకాలను విడుదల చేసింది. 
ఈ ఉత్తర్వుల ప్రకారం, రాష్ట్రంలోని అన్ని జిల్లాల్లో కొత్త పథకాలు అమలు చేయబడతాయి.
ముఖ్యమంత్రి నేతృత్వంలో జరిగిన సమావేశంలో ఈ నిర్ణయం తీసుకోబడింది.
ప్రజలకు మెరుగైన సేవలు అందించడం ప్రభుత్వ లక్ష్యం.
{dept_specific}""",
            "doc_type": "government_order",
            "items": [
                {"name": "వైద్య ఆరోగ్య శాఖ", "specific": "ప్రాథమిక ఆరోగ్య కేంద్రాలలో వైద్య సిబ్బంది నియామకం చేపట్టబడింది."},
                {"name": "పాఠశాల విద్యా శాఖ", "specific": "విద్యార్థులకు ఉచిత పుస్తకాలు, యూనిఫారాలు అందజేయబడతాయి."},
                {"name": "వ్యవసాయ శాఖ", "specific": "రైతులకు సబ్సిడీపై విత్తనాలు, ఎరువులు పంపిణీ చేయబడతాయి."},
                {"name": "ఆర్థిక శాఖ", "specific": "రాష్ట్ర బడ్జెట్​లో సంక్షేమ పథకాలకు నిధులు కేటాయించబడ్డాయి."},
                {"name": "IT మరియు కమ్యూనికేషన్ల శాఖ", "specific": "డిజిటల్ సేవలు విస్తరించడానికి కొత్త డేటా సెంటర్ ఏర్పాటు."},
                {"name": "రవాణా శాఖ", "specific": "కొత్త బస్సు రూట్లు ప్రారంభం, టికెట్ ధరలలో తగ్గింపు."}
            ]
        },
        {
            "title": "ఆంధ్రప్రదేశ్ ప్రెస్ రిలీజ్ - {event}",
            "content": """ఆంధ్రప్రదేశ్ ప్రభుత్వం {event} గురించి ప్రకటన విడుదల చేసింది.
రాష్ట్ర అభివృద్ధి కోసం కొత్త కార్యక్రమాలు ప్రారంభమవుతున్నాయి.
ముఖ్యమంత్రి ఈ కార్యక్రమాలను ప్రారంభించారు.
రాష్ట్ర ప్రజలకు మేలు కలిగించే విధంగా ఈ పథకాలు రూపొందించబడ్డాయి.
{event_details}""",
            "doc_type": "press_release",
            "items": [
                {"name": "కొత్త పథకం ప్రారంభం", "details": "ఈ పథకం ద్వారా లక్షలాది కుటుంబాలకు ప్రయోజనం చేకూరుతుంది."},
                {"name": "బడ్జెట్ ప్రకటన", "details": "మొత్తం రూ. 2.5 లక్షల కోట్లతో బడ్జెట్ ప్రతిపాదించబడింది."},
                {"name": "అభివృద్ధి కార్యక్రమం", "details": "మౌలిక సదుపాయాల అభివృద్ధికి ప్రత్యేక ప్యాకేజీ ప్రకటించబడింది."},
                {"name": "సంక్షేమ పథకం", "details": "మహిళలు, వృద్ధులు, వికలాంగులకు ప్రత్యేక సహాయం అందజేయబడుతుంది."},
                {"name": "పారిశ్రామిక విధానం", "details": "పరిశ్రమలకు భూమి కేటాయింపు, పన్ను రాయితీలు ప్రకటించబడ్డాయి."},
                {"name": "విద్యా సంస్కరణలు", "details": "కొత్త పాఠ్యప్రణాళిక ప్రకారం బోధన ప్రారంభమవుతుంది."}
            ]
        },
        {
            "title": "పౌర సేవలు - {service}",
            "content": """మీసేవ కేంద్రాల ద్వారా {service} సేవలు అందుబాటులో ఉన్నాయి.
పౌరులు సమీపంలోని మీసేవ కేంద్రానికి వెళ్లి సేవలు పొందవచ్చు.
అవసరమైన పత్రాలు సమర్పించి, అప్లికేషన్ దాఖలు చేయవచ్చు.
ఆన్​లైన్ ద్వారా కూడా దరఖాస్తు చేసుకోవచ్చు.
{service_process}""",
            "doc_type": "citizen_service",
            "items": [
                {"name": "జనన ధృవీకరణపత్రం", "process": "ఆసుపత్రి రికార్డులు, తల్లిదండ్రుల ఆధార్ కార్డులు అవసరం. 7 రోజుల్లో జారీ."},
                {"name": "ఆదాయ ధృవీకరణపత్రం", "process": "రేషన్ కార్డు, ఆధార్, బ్యాంక్ స్టేట్​మెంట్ అవసరం. రూ.35 ఫీజు."},
                {"name": "కులధృవీకరణపత్రం", "process": "కుటుంబ కార్డు, పాత కుల సర్టిఫికేట్ అవసరం. తనిఖీ తర్వాత జారీ."},
                {"name": "నివాస ధృవీకరణపత్రం", "process": "ఆధార్, విద్యుత్ బిల్లు, అద్దె ఒప్పందం అవసరం. 3 రోజుల్లో జారీ."},
                {"name": "భూమి రికార్డులు", "process": "సర్వే నంబర్, పట్టాదారు పాస్ బుక్ అవసరం. ఆన్​లైన్ లో చూడవచ్చు."},
                {"name": "డ్రైవింగ్ లైసెన్స్", "process": "లెర్నర్ లైసెన్స్ తర్వాత డ్రైవింగ్ టెస్ట్ పాస్ అవ్వాలి. RTO కార్యాలయంలో."}
            ]
        },
        {
            "title": "సంక్షేమ పథకం - {scheme}",
            "content": """తెలంగాణ/ఆంధ్రప్రదేశ్ ప్రభుత్వం {scheme} పథకాన్ని ప్రారంభించింది.
ఈ పథకం ద్వారా అర్హులైన లబ్ధిదారులకు ప్రయోజనాలు అందజేయబడతాయి.
దరఖాస్తు ప్రక్రియ మరియు అర్హత ప్రమాణాలు క్రింద వివరించబడ్డాయి.
మరిన్ని వివరాలకు సమీపంలోని ప్రభుత్వ కార్యాలయాన్ని సంప్రదించండి.
{scheme_details}""",
            "doc_type": "scheme",
            "items": [
                {"name": "రైతు బంధు", "details": "ప్రతి సీజన్​కు ఎకరాకు రూ.5000 పెట్టుబడి సహాయం. భూమి ఉన్న రైతులకు మాత్రమే."},
                {"name": "ఆసరా పెన్షన్", "details": "వృద్ధులు, వితంతువులు, వికలాంగులకు నెలకు రూ.2016 పెన్షన్."},
                {"name": "కల్యాణ లక్ష్మి", "details": "పేద కుటుంబాల ఆడపిల్లల వివాహానికి రూ.1,00,116 ఆర్థిక సహాయం."},
                {"name": "షాది ముబారక్", "details": "మైనారిటీ కుటుంబాల పెళ్ళికి రూ.1,00,116 సహాయం. ఆన్​లైన్ దరఖాస్తు."},
                {"name": "అమ్మ ఒడి", "details": "తల్లులకు పిల్లల చదువు కోసం ఏడాదికి రూ.15000 నగదు బదిలీ."},
                {"name": "జగనన్న విద్యా దీవెన", "details": "విద్యార్థుల ఫీజు రీయింబర్స్​మెంట్, స్కాలర్​షిప్. ప్రతి సెమిస్టర్​కు."}
            ]
        }
    ]
    
    documents = []
    doc_id = 1
    
    for template in templates:
        items = template.get("items", [])
        doc_type = template["doc_type"]
        
        for item in items:
            item_name = item["name"]
            
            # Build title
            if "dept_name" in template["title"]:
                title = template["title"].format(dept_name=item_name)
            elif "event" in template["title"]:
                title = template["title"].format(event=item_name)
            elif "service" in template["title"]:
                title = template["title"].format(service=item_name)
            elif "scheme" in template["title"]:
                title = template["title"].format(scheme=item_name)
            else:
                title = template["title"]
            
            # Build content with specific details
            extra_detail = item.get("specific") or item.get("details") or item.get("process") or ""
            
            if "dept_name" in template["content"]:
                content = template["content"].format(dept_name=item_name, dept_specific=extra_detail)
            elif "event" in template["content"]:
                content = template["content"].format(event=item_name, event_details=extra_detail)
            elif "service" in template["content"]:
                content = template["content"].format(service=item_name, service_process=extra_detail)
            elif "scheme" in template["content"]:
                content = template["content"].format(scheme=item_name, scheme_details=extra_detail)
            else:
                content = template["content"]
            
            # Determine department
            if doc_type == "government_order":
                department = item_name
            else:
                department = "General"
            
            doc = GovDocument(
                id=f"synthetic_{doc_id:04d}",
                title=title.strip(),
                content=content.strip(),
                url=f"https://synthetic.gov.in/doc/{doc_id}",
                source="Synthetic Government Data",
                doc_type=doc_type,
                department=department,
                date_scraped=datetime.now().isoformat(),
                date_published=datetime.now().strftime("%Y-%m-%d")
            )
            documents.append(doc)
            doc_id += 1
    
    print(f"✓ Created {len(documents)} synthetic government documents")
    return documents


def verify_data():
    """Verify scraped government data."""
    print("=" * 60)
    print("Verifying Government Data")
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
        
        docs = data.get("documents", [])
        print(f"\n📄 {json_file.name}")
        print(f"   Documents: {len(docs)}")
        print(f"   Sources: {data.get('metadata', {}).get('sources', [])}")
        print(f"   Types: {data.get('metadata', {}).get('doc_types', [])}")


def main():
    parser = argparse.ArgumentParser(description="Scrape Telugu government documents")
    parser.add_argument("--source", type=str, help="Scrape specific source (telangana, ap, ts_goir, meeseva)")
    parser.add_argument("--all", action="store_true", help="Scrape from all sources")
    parser.add_argument("--synthetic", action="store_true", help="Create synthetic data for testing")
    parser.add_argument("--limit", type=int, default=500, help="Max documents to scrape")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between requests")
    parser.add_argument("--verify", action="store_true", help="Verify scraped data")
    parser.add_argument("--output", type=str, help="Output filename")
    
    args = parser.parse_args()
    
    if args.verify:
        verify_data()
    elif args.synthetic:
        docs = create_synthetic_gov_data()
        save_documents(docs, args.output or "gov_synthetic.json")
    elif args.all:
        docs = scrape_all_sources(args.limit, args.delay)
        if docs:
            save_documents(docs, args.output)
        else:
            print("\n⚠️ No documents scraped. Creating synthetic data...")
            docs = create_synthetic_gov_data()
            save_documents(docs, "gov_synthetic.json")
    elif args.source:
        docs = scrape_source(args.source, args.limit, args.delay)
        if docs:
            save_documents(docs, args.output)
    else:
        parser.print_help()
        print("\n📋 Available sources:")
        for key, config in GOV_SOURCES.items():
            print(f"  - {key}: {config['name']}")


if __name__ == "__main__":
    main()
