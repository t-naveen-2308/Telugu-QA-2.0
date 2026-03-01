"""Quick verification of mixed data files."""
import json

files = [
    ('data/domain/news/raw/news_scaled.json', 'articles'),
    ('data/domain/government/raw/gov_scaled.json', 'documents'),
    ('data/domain/literature/raw/lit_scaled.json', 'passages'),
]

for f, key in files:
    d = json.load(open(f, 'r', encoding='utf-8'))
    meta = d.get('metadata', {})
    items = d.get(key, [])
    
    real_count = meta.get('real_article_count', 
                 meta.get('real_document_count',
                 meta.get('real_passage_count', '?')))
    synth_count = meta.get('synthetic_article_count',
                  meta.get('synthetic_document_count', 
                  meta.get('synthetic_passage_count', '?')))
    src = meta.get('source', '?')
    real_srcs = meta.get('real_sources', [])
    
    print(f"{f}:")
    print(f"  Total:     {len(items)}")
    print(f"  Real:      {real_count}")
    print(f"  Synthetic: {synth_count}")
    print(f"  Source:    {src}")
    print(f"  Real from: {real_srcs}")
    
    # Show a real item
    for item in items:
        item_src = item.get('source', '')
        if 'Synthetic' not in item_src and 'synthetic' not in item_src.lower():
            title = item.get('title', '')[:80]
            content_len = len(item.get('content', ''))
            url = item.get('url', '')[:60]
            print(f"  Sample real: {title}")
            print(f"    content: {content_len} chars, url: {url}")
            break
    print()
