"""
Generate QA pairs from domain-specific Telugu text data.

Methods:
1. Template-based generation (for structured content)
2. Entity-based extraction (NER + rule-based questions)
3. LLM-based generation (OpenAI/Claude API)

Output: SQuAD-format JSON for training

Usage:
    python scripts/qa_generation/generate_qa_pairs.py --domain news --method template
    python scripts/qa_generation/generate_qa_pairs.py --domain government --method llm --api-key XXX
    python scripts/qa_generation/generate_qa_pairs.py --all --verify
"""

import os
import sys
import json
import argparse
import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class QAPair:
    """Represents a question-answer pair."""
    id: str
    context: str
    question: str
    answers: List[Dict[str, any]]  # [{"text": "...", "answer_start": N}]
    domain: str
    source: str
    difficulty: str  # easy, medium, hard
    question_type: str  # factoid, descriptive, reasoning
    
    def to_squad_format(self) -> Dict:
        """Convert to SQuAD training format."""
        return {
            "id": self.id,
            "context": self.context,
            "question": self.question,
            "answers": self.answers,
            "domain": self.domain,
            "source": self.source
        }


# Telugu question templates by domain
NEWS_TEMPLATES = [
    # Factoid questions
    {"pattern": r"(.+?)\s+(?:అని|గా)\s+ప్రకటించారు", "q": "{0} ఎప్పుడు/ఎవరు ప్రకటించారు?", "type": "factoid"},
    {"pattern": r"(.+?)\s+జరిగింది", "q": "{0} ఎక్కడ జరిగింది?", "type": "factoid"},
    {"pattern": r"ముఖ్యమంత్రి\s+(.+?)\s+", "q": "ముఖ్యమంత్రి ఏమి చేసారు?", "type": "factoid"},
    {"pattern": r"రూ\.\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(లక్షలు|కోట్లు)?", "q": "మొత్తం ఎంత?", "type": "factoid"},
    
    # Generic news templates
    {"prefix": "ఎవరు", "suffix": "ప్రకటించారు?", "type": "factoid"},
    {"prefix": "ఏమి", "suffix": "జరిగింది?", "type": "descriptive"},
    {"prefix": "ఎక్కడ", "suffix": "జరిగింది?", "type": "factoid"},
    {"prefix": "ఎప్పుడు", "suffix": "జరుగుతుంది?", "type": "factoid"}
]

GOVERNMENT_TEMPLATES = [
    # Scheme-related
    {"keywords": ["పథకం", "రైతు బంధు", "ఆసరా", "కల్యాణ లక్ష్మి"], 
     "questions": [
         "ఈ పథకం ద్వారా ఎంత సహాయం అందుతుంది?",
         "ఈ పథకం ఎవరికి వర్తిస్తుంది?",
         "పథకం కోసం ఎలా దరఖాస్తు చేసుకోవాలి?",
         "ఈ పథకం ఎప్పుడు ప్రారంభమైంది?"
     ], "type": "factoid"},
    
    # Certificate-related
    {"keywords": ["ధృవీకరణపత్రం", "సర్టిఫికేట్", "పత్రాలు"],
     "questions": [
         "ఈ సర్టిఫికేట్ పొందడానికి ఏ పత్రాలు అవసరం?",
         "ఎన్ని రోజుల్లో సర్టిఫికేట్ జారీ అవుతుంది?",
         "ఫీజు ఎంత చెల్లించాలి?",
         "ఎక్కడ దరఖాస్తు చేసుకోవాలి?"
     ], "type": "factoid"},
    
    # Government order related
    {"keywords": ["ఉత్తర్వులు", "మార్గదర్శకాలు", "నిబంధనలు"],
     "questions": [
         "ఈ ఉత్తర్వులు ఏ శాఖకు సంబంధించినవి?",
         "కొత్త మార్గదర్శకాలు ఏమిటి?",
         "ఈ నిర్ణయం ఎవరు తీసుకున్నారు?"
     ], "type": "factoid"}
]

LITERATURE_TEMPLATES = [
    # Poetry analysis - more inclusive keywords
    {"keywords": ["పద్యం", "కవిత", "శతకం", "వేమన", "సుమతీ", "కీర్తన", "భాగవత", "భారత"],
     "questions": [
         "ఈ పద్యం/కవిత రచయిత ఎవరు?",
         "ఈ రచన యొక్క భావం ఏమిటి?",
         "ఈ రచన ఏ గ్రంథం నుండి తీసుకోబడింది?"
     ], "type": "interpretive"},
    
    # Story/prose analysis
    {"keywords": ["కథ", "నవల", "గాథ", "జానపద", "సామెత", "ఊరు", "వారు", "వాడు"],
     "questions": [
         "ఈ కథ/రచన యొక్క నీతి ఏమిటి?",
         "ఈ రచన ఏ రకం?",
         "ఈ రచన ఏ భాషలో రాయబడింది?"
     ], "type": "descriptive"},
    
    # Author-specific questions
    {"keywords": ["వేమన", "పోతన", "తిక్కన", "అన్నమయ్య", "శ్రీనాథ", "గురజాడ", "బద్దెన"],
     "questions": [
         "ఈ రచన రచయిత ఎవరు?",
         "ఈ రచయిత ఏ కాలంలో జీవించారు?",
         "ఈ రచయిత ప్రసిద్ధ రచనలు ఏమిటి?"
     ], "type": "factoid"},
    
    # Content-based questions (generic, high match rate)
    {"keywords": ["చదువు", "విద్య", "మంచి", "చెడు", "ధర్మం", "నీతి", "ప్రేమ", "భక్తి"],
     "questions": [
         "ఈ రచన ముఖ్య సందేశం ఏమిటి?",
         "ఈ రచనలో చెప్పిన విలువలు ఏమిటి?"
     ], "type": "interpretive"},
    
    # General literature questions (fallback)
    {"keywords": ["తెలుగు", "రచన", "సాహిత్యం", "కవి"],
     "questions": [
         "ఈ రచన యొక్క శైలి ఏమిటి?",
         "ఈ రచన ఏ ప్రాంతానికి చెందినది?"
     ], "type": "descriptive"}
]


def generate_qa_id(context: str, question: str) -> str:
    """Generate unique QA pair ID."""
    combined = f"{context[:100]}:{question}"
    return hashlib.md5(combined.encode()).hexdigest()[:12]


def find_answer_span(context: str, answer: str) -> Optional[int]:
    """Find the starting position of answer in context."""
    if not answer or not context:
        return None
    idx = context.find(answer)
    return idx if idx >= 0 else None


def extract_entities_telugu(text: str) -> Dict[str, List[str]]:
    """Extract named entities from Telugu text using pattern matching."""
    entities = {
        "numbers": [],
        "amounts": [],
        "dates": [],
        "names": [],
        "places": [],
        "departments": []
    }
    
    # Extract amounts (రూ. N లక్షలు/కోట్లు)
    amount_pattern = r'రూ\.\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(లక్షలు|కోట్లు|వేలు)?'
    for match in re.finditer(amount_pattern, text):
        entities["amounts"].append(match.group(0))
    
    # Extract percentages
    percent_pattern = r'(\d+(?:\.\d+)?)\s*శాతం'
    for match in re.finditer(percent_pattern, text):
        entities["numbers"].append(match.group(0))
    
    # Extract department names (శాఖ pattern) - capture 1-3 Telugu words before శాఖ
    dept_pattern = r'((?:[\u0C00-\u0C7F]+\s+){0,2}[\u0C00-\u0C7F]+\s+శాఖ)'
    for match in re.finditer(dept_pattern, text):
        dept_name = match.group(1).strip()
        if dept_name and len(dept_name) > 3:
            entities["departments"].append(dept_name)
    
    # Extract dates
    date_patterns = [
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
        r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
        r'(జనవరి|ఫిబ్రవరి|మార్చి|ఏప్రిల్|మే|జూన్|జూలై|ఆగస్టు|సెప్టెంబర్|అక్టోబర్|నవంబర్|డిసెంబర్)\s*\d{1,2}'
    ]
    for pattern in date_patterns:
        for match in re.finditer(pattern, text):
            entities["dates"].append(match.group(0))
    
    return entities


def extract_key_phrases(context: str, domain: str) -> List[str]:
    """Extract key phrases from context for question generation."""
    key_phrases = []
    
    # Extract scheme/program names
    scheme_pattern = r'([\u0C00-\u0C7F]+\s+(?:పథకం|యోజన|కార్యక్రమం|సేవ))'
    for match in re.finditer(scheme_pattern, context):
        key_phrases.append(match.group(1))
    
    # Extract department names (full)
    dept_pattern = r'((?:[\u0C00-\u0C7F]+\s+){1,3}శాఖ)'
    for match in re.finditer(dept_pattern, context):
        key_phrases.append(match.group(1).strip())
    
    # Extract names (Telugu proper nouns pattern - often after titles)
    name_pattern = r'(ముఖ్యమంత్రి|మంత్రి|రచయిత|కవి)\s+([\u0C00-\u0C7F]+(?:\s+[\u0C00-\u0C7F]+)?)'
    for match in re.finditer(name_pattern, context):
        key_phrases.append(match.group(2))
    
    # Extract work/book titles
    if domain == "literature":
        work_pattern = r'([\u0C00-\u0C7F]+(?:ము|లు|ం))\s+(?:నుండి|లో|యొక్క)'
        for match in re.finditer(work_pattern, context):
            key_phrases.append(match.group(1))
    
    return list(set(key_phrases))


def generate_context_specific_question(context: str, answer: str, domain: str, 
                                       key_phrase: str = None) -> Optional[str]:
    """Generate a question that is specific to this context."""
    # Build question incorporating context-specific details
    
    if domain == "government":
        # For amounts, ask about specific scheme
        if re.match(r'రూ\.', answer):
            if key_phrase:
                return f"{key_phrase} ద్వారా ఎంత సహాయం అందుతుంది?"
            return None
        
        # For departments
        if 'శాఖ' in answer:
            sentences = context.split('.')
            if sentences:
                action = sentences[0][:50] if sentences[0] else ""
                return f"'{action}...' ఏ శాఖ చేసింది?"
        
        # For process/eligibility
        if 'అవసరం' in answer or 'దరఖాస్తు' in answer:
            if key_phrase:
                return f"{key_phrase} పొందడానికి ఏం చేయాలి?"
    
    elif domain == "literature":
        # For author info
        if key_phrase and any(author in context for author in ['వేమన', 'పోతన', 'తిక్కన', 'అన్నమయ్య']):
            return f"'{context[:30]}...' రచయిత ఎవరు?"
        
        # For content meaning
        if len(answer) > 50:
            return f"'{context[:40]}...' యొక్క భావం ఏమిటి?"
    
    return None


def generate_template_qa(context: str, domain: str, source: str) -> List[QAPair]:
    """Generate QA pairs using domain-specific templates with context-aware questions."""
    qa_pairs = []
    
    # Select templates based on domain
    if domain == "news":
        templates = NEWS_TEMPLATES
    elif domain == "government":
        templates = GOVERNMENT_TEMPLATES
    elif domain == "literature":
        templates = LITERATURE_TEMPLATES
    else:
        templates = NEWS_TEMPLATES  # default
    
    # Extract entities and key phrases for context-specific questions
    entities = extract_entities_telugu(context)
    key_phrases = extract_key_phrases(context, domain)
    
    # Generate context-specific QA pairs
    used_answers = set()  # Track to avoid duplicate answers
    
    for template in templates:
        if "keywords" not in template:
            continue
            
        # Check if any keyword matches
        matched_keyword = None
        for keyword in template["keywords"]:
            if keyword in context:
                matched_keyword = keyword
                break
        
        if not matched_keyword:
            continue
        
        # For each potential answer, create a specific question
        potential_answers = []
        
        # Add amounts
        for amt in entities["amounts"]:
            if amt not in used_answers:
                potential_answers.append(("amount", amt))
        
        # Add departments
        for dept in entities["departments"]:
            if dept not in used_answers:
                potential_answers.append(("department", dept))
        
        # Add sentence answers (for descriptive questions)
        sentences = [s.strip() for s in re.split(r'[.।\n]', context) if len(s.strip()) > 30]
        for sent in sentences[:3]:
            sent_key = sent[:50]
            if sent_key not in used_answers and matched_keyword in sent:
                potential_answers.append(("sentence", sent))
        
        # Generate QA for each unique answer
        for ans_type, answer_text in potential_answers[:2]:  # Max 2 per template
            answer_start = find_answer_span(context, answer_text)
            if answer_start is None:
                continue
            
            # Generate context-specific question
            key_phrase = key_phrases[0] if key_phrases else None
            specific_q = generate_context_specific_question(
                context, answer_text, domain, key_phrase
            )
            
            if specific_q:
                question = specific_q
            else:
                # Fallback: modify template question with context info
                base_questions = template.get("questions", [])
                if base_questions:
                    question = base_questions[0]
                    # Add context prefix to make it specific
                    context_prefix = context[:30].strip()
                    question = f"'{context_prefix}...' - {question}"
                else:
                    continue
            
            used_answers.add(answer_text[:50])
            
            qa_pair = QAPair(
                id=generate_qa_id(context, question + answer_text),
                context=context,
                question=question,
                answers=[{"text": answer_text, "answer_start": answer_start}],
                domain=domain,
                source=source,
                difficulty="medium",
                question_type=template.get("type", "factoid")
            )
            qa_pairs.append(qa_pair)
    
    return qa_pairs


def generate_entity_based_qa(context: str, domain: str, source: str) -> List[QAPair]:
    """Generate QA pairs based on extracted entities with context-specific questions."""
    qa_pairs = []
    entities = extract_entities_telugu(context)
    key_phrases = extract_key_phrases(context, domain)
    
    # Create context identifier
    context_id = context[:25].strip().replace('\n', ' ')
    
    # Amount-based questions - context specific
    for idx, amount in enumerate(entities["amounts"][:2]):
        # Create context-specific question
        if key_phrases:
            question = f"{key_phrases[0]} ద్వారా లభించే మొత్తం ఎంత?"
        else:
            question = f"'{context_id}...' లో పేర్కొన్న మొత్తం ఎంత?"
        
        answer_start = find_answer_span(context, amount)
        if answer_start is not None:
            qa_pairs.append(QAPair(
                id=generate_qa_id(context, question + amount + str(idx)),
                context=context,
                question=question,
                answers=[{"text": amount, "answer_start": answer_start}],
                domain=domain,
                source=source,
                difficulty="easy",
                question_type="factoid"
            ))
    
    # Department-based questions - context specific
    for idx, dept in enumerate(entities["departments"][:1]):
        # Create context-specific question using first sentence
        first_sentence = context.split('.')[0][:40] if '.' in context else context[:40]
        question = f"'{first_sentence}...' ఏ శాఖ చేసింది?"
        
        answer_start = find_answer_span(context, dept)
        if answer_start is not None:
            qa_pairs.append(QAPair(
                id=generate_qa_id(context, question + dept + str(idx)),
                context=context,
                question=question,
                answers=[{"text": dept, "answer_start": answer_start}],
                domain=domain,
                source=source,
                difficulty="easy",
                question_type="factoid"
            ))
    
    return qa_pairs


def generate_sentence_level_qa(context: str, domain: str, source: str) -> List[QAPair]:
    """Generate QA from individual sentences for higher diversity."""
    qa_pairs = []
    sentences = [s.strip() for s in re.split(r'[.।\n]', context) if len(s.strip()) > 20]
    
    for sent_idx, sentence in enumerate(sentences):
        # District/place extraction
        place_match = re.search(r'([\u0C00-\u0C7F]+)\s*జిల్లా', sentence)
        if place_match:
            answer = place_match.group(1) + ' జిల్లా'
            answer_start = context.find(answer)
            if answer_start >= 0:
                q = f"'{sentence[:25]}...' ఏ జిల్లాలో జరిగింది?"
                qa_pairs.append(QAPair(
                    id=generate_qa_id(context, q + answer),
                    context=context, question=q,
                    answers=[{"text": answer, "answer_start": answer_start}],
                    domain=domain, source=source,
                    difficulty="easy", question_type="factoid"
                ))
        
        # Official/person extraction
        official_match = re.search(r'(ముఖ్యమంత్రి|మంత్రి|కలెక్టర్|ఎమ్మెల్యే|ఎంపీ|RDO|తహసీల్దార్|మేయర్)', sentence)
        if official_match:
            answer = official_match.group(1)
            answer_start = context.find(answer)
            if answer_start >= 0:
                q = f"'{sentence[:25]}...' ఎవరు పాల్గొన్నారు/చేసారు?"
                qa_pairs.append(QAPair(
                    id=generate_qa_id(context, q + answer),
                    context=context, question=q,
                    answers=[{"text": answer, "answer_start": answer_start}],
                    domain=domain, source=source,
                    difficulty="easy", question_type="factoid"
                ))
        
        # Date extraction
        date_match = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', sentence)
        if date_match:
            answer = date_match.group(1)
            answer_start = context.find(answer)
            if answer_start >= 0:
                q = f"'{sentence[:25]}...' చివరి తేదీ/తేదీ ఏమిటి?"
                qa_pairs.append(QAPair(
                    id=generate_qa_id(context, q + answer),
                    context=context, question=q,
                    answers=[{"text": answer, "answer_start": answer_start}],
                    domain=domain, source=source,
                    difficulty="easy", question_type="factoid"
                ))
        
        # Number extraction (posts, count)
        num_match = re.search(r'(\d+)\s*(పోస్టులు|మంది|గంటలు|రోజులు|సంవత్సరాలు)', sentence)
        if num_match:
            answer = num_match.group(0)
            answer_start = context.find(answer)
            if answer_start >= 0:
                q = f"'{sentence[:25]}...' ఎన్ని {num_match.group(2)}?"
                qa_pairs.append(QAPair(
                    id=generate_qa_id(context, q + answer),
                    context=context, question=q,
                    answers=[{"text": answer, "answer_start": answer_start}],
                    domain=domain, source=source,
                    difficulty="easy", question_type="factoid"
                ))
        
        # Scheme name extraction
        scheme_match = re.search(r'([\u0C00-\u0C7F]+(?:\s+[\u0C00-\u0C7F]+)?\s+(?:పథకం|యోజన))', sentence)
        if scheme_match:
            answer = scheme_match.group(1)
            answer_start = context.find(answer)
            if answer_start >= 0:
                q = f"'{sentence[:25]}...' ఏ పథకం/యోజన గురించి?"
                qa_pairs.append(QAPair(
                    id=generate_qa_id(context, q + answer),
                    context=context, question=q,
                    answers=[{"text": answer, "answer_start": answer_start}],
                    domain=domain, source=source,
                    difficulty="medium", question_type="factoid"
                ))

        # City/location extraction  (e.g. "హైదరాబాద్లో")
        city_match = re.search(r'([\u0C00-\u0C7F]{3,})లో\s', sentence)
        if city_match and not place_match:  # avoid duplicate with district
            answer = city_match.group(1)
            answer_start = context.find(answer)
            if answer_start >= 0 and len(answer) > 3:
                q = f"'{sentence[:25]}...' ఎక్కడ జరిగింది?"
                qa_pairs.append(QAPair(
                    id=generate_qa_id(context, q + answer),
                    context=context, question=q,
                    answers=[{"text": answer, "answer_start": answer_start}],
                    domain=domain, source=source,
                    difficulty="easy", question_type="factoid"
                ))

        # Party/organization extraction (for news)
        party_match = re.search(r'([\u0C00-\u0C7F]+(?:\s+[\u0C00-\u0C7F]+){0,3}\s+(?:పార్టీ|సంస్థ|కంపెనీ|జట్టు))', sentence)
        if party_match:
            answer = party_match.group(1).strip()
            answer_start = context.find(answer)
            if answer_start >= 0 and len(answer) > 5:
                q = f"'{sentence[:25]}...' ఏ పార్టీ/సంస్థ?"
                qa_pairs.append(QAPair(
                    id=generate_qa_id(context, q + answer),
                    context=context, question=q,
                    answers=[{"text": answer, "answer_start": answer_start}],
                    domain=domain, source=source,
                    difficulty="easy", question_type="factoid"
                ))

        # Percentage extraction (for business news)
        pct_match = re.search(r'(\d+(?:\.\d+)?%)', sentence)
        if pct_match:
            answer = pct_match.group(1)
            answer_start = context.find(answer)
            if answer_start >= 0:
                q = f"'{sentence[:25]}...' వృద్ధి/శాతం ఎంత?"
                qa_pairs.append(QAPair(
                    id=generate_qa_id(context, q + answer),
                    context=context, question=q,
                    answers=[{"text": answer, "answer_start": answer_start}],
                    domain=domain, source=source,
                    difficulty="easy", question_type="factoid"
                ))
    
    return qa_pairs


def generate_synthetic_qa_pairs(context: str, domain: str, source: str, num_pairs: int = 8) -> List[QAPair]:
    """Generate synthetic QA pairs using multiple methods."""
    all_pairs = []
    
    # Method 1: Template-based
    template_pairs = generate_template_qa(context, domain, source)
    all_pairs.extend(template_pairs)
    
    # Method 2: Entity-based
    entity_pairs = generate_entity_based_qa(context, domain, source)
    all_pairs.extend(entity_pairs)
    
    # Method 3: Sentence-level extraction (NEW)
    sentence_pairs = generate_sentence_level_qa(context, domain, source)
    all_pairs.extend(sentence_pairs)
    
    # Deduplicate by (question, answer) pair
    seen = set()
    unique_pairs = []
    for pair in all_pairs:
        key = (pair.question.strip(), pair.answers[0]['text'].strip() if pair.answers else '')
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)
    
    # Final span validation
    validated = []
    for pair in unique_pairs:
        if pair.answers:
            a = pair.answers[0]
            ctx = pair.context
            start = a['answer_start']
            text = a['text']
            if start >= 0 and ctx[start:start+len(text)] == text:
                validated.append(pair)
    
    return validated[:num_pairs]


def generate_literature_qa(item: Dict) -> List[QAPair]:
    """Generate QA pairs for literature using metadata (author, work_title, etc.)."""
    qa_pairs = []
    
    content = item.get("content", "")
    author = item.get("author", "")
    title = item.get("title", "")
    source = item.get("source", "unknown")
    work_title = item.get("work_title", "")
    genre = item.get("genre", "")
    
    if len(content) < 30:
        return []
    
    # Create context identifier from first line
    first_line = content.split('\n')[0][:40] if '\n' in content else content[:40]
    
    # 1. Author question (if author is known)
    if author and author != "అజ్ఞాత కవి" and author != "జానపద సాహిత్యం":
        question = f"'{first_line}...' అనే పద్యం/రచన రచయిత ఎవరు?"
        # The answer needs to be in the content - append author info for QA purposes
        # We'll create a composite context that includes the metadata
        enhanced_context = f"{content}\n\nఈ రచన {author} రచించారు."
        
        answer_start = enhanced_context.find(author)
        if answer_start >= 0:
            qa_pairs.append(QAPair(
                id=generate_qa_id(content, question + author),
                context=enhanced_context,
                question=question,
                answers=[{"text": author, "answer_start": answer_start}],
                domain="literature",
                source=source,
                difficulty="medium",
                question_type="factoid"
            ))
    
    # 2. Work title question
    if work_title:
        question = f"'{first_line}...' ఏ గ్రంథం నుండి తీసుకోబడింది?"
        enhanced_context = f"{content}\n\nఈ భాగం {work_title} అనే గ్రంథం నుండి తీసుకోబడింది."
        
        answer_start = enhanced_context.find(work_title)
        if answer_start >= 0:
            qa_pairs.append(QAPair(
                id=generate_qa_id(content, question + work_title),
                context=enhanced_context,
                question=question,
                answers=[{"text": work_title, "answer_start": answer_start}],
                domain="literature",
                source=source,
                difficulty="medium",
                question_type="factoid"
            ))
    
    # 3. Content meaning question - use a significant line as answer
    lines = [l.strip() for l in content.split('\n') if len(l.strip()) > 15]
    if len(lines) >= 2:
        # Use a middle line as the "key meaning"
        answer_line = lines[len(lines)//2]
        question = f"'{first_line}...' అనే రచనలో ముఖ్య సందేశం ఏమిటి?"
        
        answer_start = content.find(answer_line)
        if answer_start >= 0:
            qa_pairs.append(QAPair(
                id=generate_qa_id(content, question + answer_line),
                context=content,
                question=question,
                answers=[{"text": answer_line, "answer_start": answer_start}],
                domain="literature",
                source=source,
                difficulty="hard",
                question_type="interpretive"
            ))
    
    # 4. Genre question
    if genre:
        genre_telugu = {
            "poetry": "కవిత్వం/పద్యం",
            "epic": "ప్రబంధం/కావ్యం",
            "devotional": "భక్తి గీతం",
            "folk": "జానపద సాహిత్యం",
            "prose": "గద్యం"
        }.get(genre, genre)
        
        question = f"'{first_line}...' ఏ రకమైన సాహిత్య ప్రక్రియ?"
        enhanced_context = f"{content}\n\nఈ రచన {genre_telugu} ప్రక్రియకు చెందినది."
        
        answer_start = enhanced_context.find(genre_telugu)
        if answer_start >= 0:
            qa_pairs.append(QAPair(
                id=generate_qa_id(content, question + genre_telugu),
                context=enhanced_context,
                question=question,
                answers=[{"text": genre_telugu, "answer_start": answer_start}],
                domain="literature",
                source=source,
                difficulty="easy",
                question_type="factoid"
            ))
    
    # 5. Theme/keyword question from content
    themes = {
        "విద్య": "విద్య గురించి", "నీతి": "నీతి గురించి",
        "భక్తి": "భక్తి గురించి", "ప్రేమ": "ప్రేమ గురించి",
        "ధైర్యం": "ధైర్యం గురించి", "ప్రకృతి": "ప్రకృతి గురించి",
        "స్నేహం": "స్నేహం గురించి", "త్యాగం": "త్యాగం గురించి"
    }
    for theme_word, theme_desc in themes.items():
        if theme_word in content:
            # Find the sentence containing the theme
            for sent in content.split('\n'):
                sent = sent.strip()
                if theme_word in sent and len(sent) > 15:
                    answer_start = content.find(sent)
                    if answer_start >= 0:
                        question = f"'{first_line}...' రచనలో {theme_desc} ఏమి చెప్పబడింది?"
                        qa_pairs.append(QAPair(
                            id=generate_qa_id(content, question + sent),
                            context=content,
                            question=question,
                            answers=[{"text": sent, "answer_start": answer_start}],
                            domain="literature",
                            source=source,
                            difficulty="hard",
                            question_type="interpretive"
                        ))
                    break
    
    # 6. Period/century question for poets with explicit period info
    period_match = re.search(r'(\d+\w?\s*శతాబ్దం)', content)
    if period_match:
        answer = period_match.group(1)
        answer_start = content.find(answer)
        if answer_start >= 0:
            question = f"'{first_line}...' రచయిత ఏ శతాబ్దానికి చెందినవారు?"
            qa_pairs.append(QAPair(
                id=generate_qa_id(content, question + answer),
                context=content,
                question=question,
                answers=[{"text": answer, "answer_start": answer_start}],
                domain="literature",
                source=source,
                difficulty="medium",
                question_type="factoid"
            ))
    
    # Final span validation for all literature QA
    validated = []
    for pair in qa_pairs:
        if pair.answers:
            a = pair.answers[0]
            ctx = pair.context
            start = a['answer_start']
            text = a['text']
            if 0 <= start < len(ctx) and ctx[start:start+len(text)] == text:
                validated.append(pair)
    
    return validated


def load_domain_data(domain: str) -> List[Dict]:
    """Load raw domain data from collected JSON files."""
    domain_dir = Path(f"data/domain/{domain}/raw")
    all_items = []
    
    if not domain_dir.exists():
        print(f"Domain directory not found: {domain_dir}")
        return []
    
    for json_file in domain_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract items based on domain structure
            if domain == "news":
                items = data.get("articles", [])
                for item in items:
                    all_items.append({
                        "content": item.get("content", ""),
                        "source": item.get("source", json_file.stem),
                        "title": item.get("title", "")
                    })
            elif domain == "government":
                items = data.get("documents", [])
                for item in items:
                    all_items.append({
                        "content": item.get("content", ""),
                        "source": item.get("source", json_file.stem),
                        "title": item.get("title", "")
                    })
            elif domain == "literature":
                items = data.get("passages", [])
                for item in items:
                    all_items.append({
                        "content": item.get("content", ""),
                        "source": item.get("source", json_file.stem),
                        "title": item.get("title", ""),
                        "author": item.get("author")
                    })
            
            print(f"  Loaded {len(items)} items from {json_file.name}")
        except Exception as e:
            print(f"  Error loading {json_file}: {e}")
    
    return all_items


def generate_for_domain(domain: str, method: str = "template", limit: int = None) -> List[QAPair]:
    """Generate QA pairs for a specific domain."""
    print(f"\n📝 Generating QA pairs for domain: {domain}")
    print(f"   Method: {method}")
    print("-" * 50)
    
    # Load domain data
    items = load_domain_data(domain)
    
    if not items:
        print(f"  No data found for domain: {domain}")
        return []
    
    if limit:
        items = items[:limit]
    
    print(f"  Processing {len(items)} items...")
    
    all_qa_pairs = []
    
    for idx, item in enumerate(items):
        content = item.get("content", "")
        source = item.get("source", "unknown")
        
        if len(content) < 50:  # Skip very short content
            continue
        
        # Generate QA pairs - use domain-specific generation
        if domain == "literature":
            # Use metadata-aware generation for literature
            qa_pairs = generate_literature_qa(item)
        else:
            qa_pairs = generate_synthetic_qa_pairs(content, domain, source)
        all_qa_pairs.extend(qa_pairs)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(items)} items, generated {len(all_qa_pairs)} QA pairs")
    
    print(f"✓ Generated {len(all_qa_pairs)} QA pairs for {domain}")
    return all_qa_pairs


def convert_to_squad(qa_pairs: List[QAPair], domain: str) -> Dict:
    """Convert QA pairs to SQuAD format."""
    # Group by EXACT context (not truncated) to avoid span mismatches
    context_groups = {}
    for pair in qa_pairs:
        ctx_key = hash(pair.context)
        if ctx_key not in context_groups:
            context_groups[ctx_key] = {
                "context": pair.context,
                "qas": []
            }
        context_groups[ctx_key]["qas"].append({
            "id": pair.id,
            "question": pair.question,
            "answers": pair.answers,
            "difficulty": pair.difficulty,
            "question_type": pair.question_type
        })
    
    # Build SQuAD structure
    paragraphs = [
        {
            "context": group["context"],
            "qas": group["qas"]
        }
        for group in context_groups.values()
    ]
    
    return {
        "version": "2.0",
        "domain": domain,
        "generated_at": datetime.now().isoformat(),
        "data": [
            {
                "title": f"Telugu {domain.capitalize()} QA",
                "paragraphs": paragraphs
            }
        ]
    }


def save_qa_pairs(qa_pairs: List[QAPair], domain: str, output_file: str = None):
    """Save generated QA pairs."""
    output_dir = Path(f"data/domain/{domain}/qa_pairs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{domain}_qa_{timestamp}.json"
    
    output_path = output_dir / output_file
    
    # Convert to SQuAD format
    squad_data = convert_to_squad(qa_pairs, domain)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(squad_data, f, ensure_ascii=False, indent=2)
    
    # Also save raw pairs for analysis
    raw_path = output_dir / f"raw_{output_file}"
    raw_data = {
        "total_pairs": len(qa_pairs),
        "domain": domain,
        "pairs": [asdict(p) for p in qa_pairs]
    }
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Saved {len(qa_pairs)} QA pairs to {output_path}")
    print(f"   Raw data: {raw_path}")


def verify_qa_quality(domain: str):
    """Verify quality of generated QA pairs."""
    print(f"\n🔍 Verifying QA pairs for domain: {domain}")
    
    qa_dir = Path(f"data/domain/{domain}/qa_pairs")
    if not qa_dir.exists():
        print(f"No QA pairs found for {domain}")
        return
    
    for json_file in qa_dir.glob("*.json"):
        if json_file.name.startswith("raw_"):
            continue
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        paragraphs = data.get("data", [{}])[0].get("paragraphs", [])
        total_qas = sum(len(p.get("qas", [])) for p in paragraphs)
        
        print(f"\n📄 {json_file.name}")
        print(f"   Contexts: {len(paragraphs)}")
        print(f"   QA pairs: {total_qas}")
        
        # Sample quality check
        if paragraphs:
            sample = paragraphs[0]
            print(f"\n   Sample context: {sample['context'][:100]}...")
            if sample.get("qas"):
                qa = sample["qas"][0]
                print(f"   Sample Q: {qa['question']}")
                print(f"   Sample A: {qa['answers'][0]['text'][:50]}...")


def main():
    parser = argparse.ArgumentParser(description="Generate QA pairs from domain data")
    parser.add_argument("--domain", type=str, choices=["news", "government", "literature"],
                        help="Generate for specific domain")
    parser.add_argument("--all", action="store_true", help="Generate for all domains")
    parser.add_argument("--method", type=str, default="template", 
                        choices=["template", "entity", "llm"],
                        help="Generation method")
    parser.add_argument("--limit", type=int, help="Limit number of items to process")
    parser.add_argument("--verify", action="store_true", help="Verify generated QA pairs")
    parser.add_argument("--output", type=str, help="Output filename")
    
    args = parser.parse_args()
    
    if args.verify:
        for domain in ["news", "government", "literature"]:
            verify_qa_quality(domain)
    elif args.all:
        for domain in ["news", "government", "literature"]:
            qa_pairs = generate_for_domain(domain, args.method, args.limit)
            if qa_pairs:
                save_qa_pairs(qa_pairs, domain, args.output)
    elif args.domain:
        qa_pairs = generate_for_domain(args.domain, args.method, args.limit)
        if qa_pairs:
            save_qa_pairs(qa_pairs, args.domain, args.output)
    else:
        parser.print_help()
        print("\n📋 Domains: news, government, literature")
        print("📋 Methods: template, entity, llm")


if __name__ == "__main__":
    main()
