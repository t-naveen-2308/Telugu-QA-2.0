"""
Generate large-scale synthetic domain data for Telugu QA training.

Target: 4,000+ QA pairs across domains
- Government: 500 documents -> 1,500 QA pairs
- Literature: 200 passages -> 1,000 QA pairs
- News: 400 articles -> 1,500 QA pairs

Usage:
    python scripts/data_collection/generate_scaled_data.py --domain government --count 500
    python scripts/data_collection/generate_scaled_data.py --domain literature --count 200
    python scripts/data_collection/generate_scaled_data.py --domain news --count 400
    python scripts/data_collection/generate_scaled_data.py --all
"""

import json
import random
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

# Diverse government content components
GOV_COMPONENTS = {
    "departments": [
        "వైద్య ఆరోగ్య శాఖ", "పాఠశాల విద్యా శాఖ", "ఉన్నత విద్యా శాఖ",
        "వ్యవసాయ శాఖ", "ఆర్థిక శాఖ", "గృహ నిర్మాణ శాఖ",
        "రవాణా శాఖ", "విద్యుత్ శాఖ", "నీటి పారుదల శాఖ",
        "పంచాయతీ రాజ్ శాఖ", "పట్టణాభివృద్ధి శాఖ", "సామాజిక సంక్షేమ శాఖ",
        "మహిళా శిశు సంక్షేమ శాఖ", "పశు సంవర్ధక శాఖ", "అటవీ శాఖ",
        "పర్యాటక శాఖ", "సాంస్కృతిక శాఖ", "క్రీడల శాఖ",
        "IT మరియు కమ్యూనికేషన్ల శాఖ", "పరిశ్రమల శాఖ"
    ],
    "schemes": [
        ("రైతు బంధు", "రూ.5,000", "రైతులకు పెట్టుబడి సహాయం"),
        ("ఆసరా పెన్షన్", "రూ.2,016", "వృద్ధులు, వితంతువులకు నెలవారీ పెన్షన్"),
        ("కల్యాణ లక్ష్మి", "రూ.1,00,116", "ఆడపిల్లల వివాహానికి సహాయం"),
        ("షాది ముబారక్", "రూ.1,00,116", "మైనారిటీ వివాహ సహాయం"),
        ("అమ్మ ఒడి", "రూ.15,000", "తల్లులకు విద్యా సహాయం"),
        ("జగనన్న విద్యా దీవెన", "రూ.20,000", "విద్యార్థుల ఫీజు రీయింబర్స్మెంట్"),
        ("వైద్య ధరణి", "రూ.5,00,000", "ఉచిత వైద్య చికిత్స"),
        ("గృహ లక్ష్మి", "రూ.2,500", "మహిళలకు నెలవారీ సహాయం"),
        ("చేయూత", "రూ.75,000", "యువతకు ఉపాధి శిక్షణ"),
        ("కేసీఆర్ కిట్", "రూ.13,000", "గర్భిణీ స్త్రీలకు పోషకాహారం"),
        ("దళిత బంధు", "రూ.10,00,000", "దళితులకు ఆర్థిక సహాయం"),
        ("రైతు రుణమాఫీ", "రూ.1,00,000", "రైతుల అప్పుల మాఫీ"),
    ],
    "certificates": [
        ("జనన ధృవీకరణ పత్రం", "7 రోజులు", "రూ.0", "ఆసుపత్రి రికార్డు, తల్లిదండ్రుల ఆధార్"),
        ("మరణ ధృవీకరణ పత్రం", "7 రోజులు", "రూ.0", "ఆసుపత్రి రికార్డు, ఆధార్ కార్డు"),
        ("ఆదాయ ధృవీకరణ పత్రం", "15 రోజులు", "రూ.35", "రేషన్ కార్డు, బ్యాంక్ స్టేట్మెంట్"),
        ("కుల ధృవీకరణ పత్రం", "30 రోజులు", "రూ.50", "పాత కుల సర్టిఫికెట్, రేషన్ కార్డు"),
        ("నివాస ధృవీకరణ పత్రం", "15 రోజులు", "రూ.25", "విద్యుత్ బిల్లు, అద్దె ఒప్పందం"),
        ("ఈబీసీ సర్టిఫికెట్", "15 రోజులు", "రూ.35", "ఆదాయ ధృవీకరణ పత్రం"),
        ("భూమి రికార్డులు", "1 రోజు", "రూ.10", "సర్వే నంబర్, ఆధార్"),
        ("డ్రైవింగ్ లైసెన్స్", "30 రోజులు", "రూ.200", "లెర్నర్ లైసెన్స్, వయస్సు నిరూపణ"),
        ("పాస్‌పోర్ట్", "45 రోజులు", "రూ.1,500", "ఆధార్, పుట్టిన తేదీ ధృవీకరణ"),
        ("వోటర్ ఐడీ", "30 రోజులు", "రూ.0", "చిరునామా నిరూపణ, ఫోటో"),
    ],
    "districts_ts": [
        "హైదరాబాద్", "రంగారెడ్డి", "మేడ్చల్", "సంగారెడ్డి", "మెదక్",
        "నిజామాబాద్", "కామారెడ్డి", "ఆదిలాబాద్", "మంచిర్యాల", "కరీంనగర్",
        "పెద్దపల్లి", "జగిత్యాల", "రాజన్న సిరిసిల్ల", "వరంగల్", "హనుమకొండ",
        "జనగామ", "మహబూబాబాద్", "భద్రాద్రి కొత్తగూడెం", "ఖమ్మం", "సూర్యాపేట",
        "నల్గొండ", "యాదాద్రి భువనగిరి", "మహబూబ్‌నగర్", "నాగర్ కర్నూల్", "వనపర్తి",
        "గద్వాల", "నారాయణపేట", "వికారాబాద్", "మేడక్", "సిద్దిపేట", "కొమురంభీమ్"
    ],
    "districts_ap": [
        "విశాఖపట్నం", "విజయనగరం", "శ్రీకాకుళం", "పార్వతీపురం మన్యం",
        "అనకాపల్లి", "కాకినాడ", "ఏలూరు", "రాజమహేంద్రవరం", "పశ్చిమ గోదావరి",
        "కొణసీమ", "గుంటూరు", "బాపట్ల", "ప్రకాశం", "నెల్లూరు", "తిరుపతి",
        "చిత్తూరు", "అన్నమయ్య", "కడప", "నందయాల్", "కర్నూలు", "అనంతపురం",
        "శ్రీ సత్యసాయి", "కృష్ణా", "ఎన్టీఆర్", "పల్నాడు"
    ],
    "actions": [
        "కొత్త మార్గదర్శకాలను విడుదల చేసింది",
        "నిధులు మంజూరు చేసింది",
        "క్రొత్త పథకాన్ని ప్రారంభించింది",
        "సమావేశం నిర్వహించింది",
        "నియామకాలకు అనుమతి ఇచ్చింది",
        "పరిశీలన జరిపింది",
        "ఉత్తర్వులు జారీ చేసింది",
        "ప్రకటన విడుదల చేసింది",
        "అభివృద్ధి ప్రాజెక్టుకు శంకుస్థాపన చేసింది",
        "క్రొత్త సదుపాయాలను ప్రారంభించింది"
    ],
    "officials": [
        "ముఖ్యమంత్రి", "ఉప ముఖ్యమంత్రి", "మంత్రి", "ప్రధాన కార్యదర్శి",
        "కలెక్టర్", "జిల్లా పరిషత్ చైర్మన్", "ఎమ్మెల్యే", "ఎంపీ",
        "మేయర్", "సర్పంచ్", "తహసీల్దార్", "RDO"
    ]
}

# Literature content components
LIT_COMPONENTS = {
    "classical_poets": [
        ("వేమన", "శతక కవి", "15వ శతాబ్దం", ["వేమన పద్యాలు", "విశ్వదాభిరామ వినురవేమ"]),
        ("బద్దెన భూపాలుడు", "శతక కవి", "13వ శతాబ్దం", ["సుమతీ శతకం"]),
        ("పోతన", "కావ్య కవి", "15వ శతాబ్దం", ["ఆంధ్ర మహాభాగవతము", "భాగవత పద్యాలు"]),
        ("తిక్కన", "కావ్య కవి", "13వ శతాబ్దం", ["ఆంధ్ర మహాభారతము", "నిర్వచనోత్తర రామాయణము"]),
        ("నన్నయ", "ఆదికవి", "11వ శతాబ్దం", ["ఆంధ్ర మహాభారతము (మొదటి భాగం)"]),
        ("అన్నమయ్య", "భక్తి కవి", "15వ శతాబ్దం", ["సంకీర్తనలు", "శృంగార సంకీర్తనలు"]),
        ("త్యాగరాజు", "వాగ్గేయకారుడు", "18వ శతాబ్దం", ["కృతులు", "పంచరత్న కృతులు"]),
        ("క్షేత్రయ్య", "పద కవి", "17వ శతాబ్దం", ["పదాలు", "మువ్వగోపాల పదాలు"]),
        ("శ్రీనాథుడు", "ప్రబంధ కవి", "15వ శతాబ్దం", ["శృంగార నైషధము", "పల్నాటి వీర చరిత్ర"]),
        ("గురజాడ అప్పారావు", "ఆధునిక కవి", "19వ శతాబ్దం", ["కన్యాశుల్కం", "ముత్యాల సరాలు"]),
    ],
    "themes": [
        ("విద్య", ["చదువు", "జ్ఞానం", "బోధన", "నేర్పు"]),
        ("నీతి", ["ధర్మం", "మంచి", "చెడు", "సత్యం"]),
        ("భక్తి", ["దేవుడు", "ప్రార్థన", "పూజ", "మోక్షం"]),
        ("ప్రేమ", ["అనురాగం", "విరహం", "మిలనం", "సంయోగం"]),
        ("స్నేహం", ["మిత్రుడు", "తోడు", "నమ్మకం", "ఆపద"]),
        ("త్యాగం", ["బలిదానం", "సేవ", "పరోపకారం"]),
        ("ధైర్యం", ["శౌర్యం", "వీరత్వం", "యుద్ధం", "విజయం"]),
        ("ప్రకృతి", ["వనం", "నది", "పర్వతం", "సముద్రం"]),
    ],
    "genres": [
        ("పద్యం", "poetry"),
        ("శతకం", "poetry"),
        ("కీర్తన", "devotional"),
        ("కావ్యం", "epic"),
        ("ప్రబంధం", "epic"),
        ("గద్యం", "prose"),
        ("కథ", "prose"),
        ("జానపద గీతం", "folk"),
        ("సామెత", "folk"),
    ],
    "sample_verses": [
        # వేమన పద్యాలు
        ("ఉప్పు కప్పురమ్ము ఒక్క పోలిక నుండు\nచూడ చూడ రుచుల జాడ లేరు\nపురుషులందు పుణ్య పురుషు లెఱుంగరు\nవిశ్వదాభిరామ వినురవేమ", "వేమన", "పద్యం"),
        ("అనగననగ రాగ మతిశయిల్లుచునుండు\nతినగ తినగ వేము తీయనుండు\nసాధనంబున పనులు సమకూరు ధరలోన\nవిశ్వదాభిరామ వినురవేమ", "వేమన", "పద్యం"),
        ("చిత్తశుద్ధి కలిగి చేసిన పుణ్యమ్ము\nకొంచెమైన నదియు కొదువ గాదు\nవిత్తనమ్ము మఱ్ఱి వృక్షమై పెరుగును\nవిశ్వదాభిరామ వినురవేమ", "వేమన", "పద్యం"),
        # సుమతీ శతకం
        ("అధికారము దొరికెనేని\nమదిలో యొక్క కీడు దోచు మానవునికి\nచెదరి మనసున నెగసెనేని\nగదికి నొక్క వేల కొఱతలుండు సుమతీ", "బద్దెన", "శతకం"),
        # పోతన భాగవతం
        ("అల వైకుంఠపురంబులో నగరిలో\nఆ మూలసౌధంబు దాపల నుండె\nపల్వలంబున నీరు గల్గు చోట\nపూజార్హంబుగ పూర్ణచంద్రుడె యొప్పున", "పోతన", "కావ్యం"),
        # అన్నమయ్య కీర్తన
        ("బ్రహ్మమొకటే పరబ్రహ్మమొకటే\nజీవ బ్రహ్మైక్యము సిద్ధమాయెన్\nతారకమంత్రము తరించి చూడగ\nఆరయ మీరాత్మకు అది ముఖ్యమయ్య", "అన్నమయ్య", "కీర్తన"),
        # గురజాడ
        ("దేశమును ప్రేమించుమన్నా\nమంచి యన్నది పెంచుమన్నా\nవందనం మాధరమన్నా\nవేమన", "గురజాడ అప్పారావు", "ఆధునిక కవిత"),
    ],
    "proverbs": [
        "అడగనిదే అమ్మైనా పెట్టదు",
        "ఆకలికి అన్నమే, మోహానికి మన్యమే",
        "ఇంటికన్నా గుడి మేలు",
        "ఉప్పు తిన్న వాడు నీళ్ళు తాగాలి",
        "ఊరు మారినా ఉసురు మారదు",
        "కంటికి నిద్ర, మనసుకు శాంతి",
        "కష్టపడ్డవాడికి ఫలితం తప్పదు",
        "జ్ఞానికి వేల మాటలు, అజ్ఞానికి ఒక్క దెబ్బ",
        "తాళము గట్టి దానం తప్పదు",
        "నీతి లేని బ్రతుకు రీతిలేని పాట",
        "పంట చేలో కలుపు మొక్క",
        "భయము లేని వాడికి బంధువులు ఎందరు",
        "మాట మంచిదైతే మనిషి మంచివాడు",
        "రానివారిని రమ్మని పిలవరాదు",
        "విద్య ఉన్నవాడికి వివేకం ఉంటుంది",
    ]
}


def generate_gov_document(doc_id: int) -> Dict:
    """Generate a diverse government document."""
    doc_type = random.choice(["scheme", "certificate", "order", "press_release", "notification"])
    
    if doc_type == "scheme":
        scheme = random.choice(GOV_COMPONENTS["schemes"])
        dept = random.choice(GOV_COMPONENTS["departments"])
        district = random.choice(GOV_COMPONENTS["districts_ts"] + GOV_COMPONENTS["districts_ap"])
        official = random.choice(GOV_COMPONENTS["officials"])
        
        content = f"""{dept} {scheme[0]} పథకాన్ని {district} జిల్లాలో ప్రారంభించింది.
ఈ పథకం ద్వారా అర్హులైన లబ్ధిదారులకు {scheme[1]} ఆర్థిక సహాయం అందజేయబడుతుంది.
{scheme[2]} అనే లక్ష్యంతో ఈ పథకం రూపొందించబడింది.
{official} ఈ పథకాన్ని అధికారికంగా ప్రారంభించారు.
దరఖాస్తు ప్రక్రియ మీసేవ కేంద్రాల్లో మొదలైంది.
అర్హత ప్రమాణాలు:
- ఆదాయ పరిమితి: రూ.2,00,000 లోపు
- వయస్సు: 18 ఏళ్ళ పైన
- నివాసం: {district} జిల్లా
మరిన్ని వివరాలకు సమీపంలోని ప్రభుత్వ కార్యాలయాన్ని సంప్రదించండి."""
        
        title = f"{scheme[0]} - {district} జిల్లా"
        
    elif doc_type == "certificate":
        cert = random.choice(GOV_COMPONENTS["certificates"])
        dept = random.choice(GOV_COMPONENTS["departments"][:5])
        district = random.choice(GOV_COMPONENTS["districts_ts"])
        
        content = f"""{cert[0]} పొందడం కోసం {district} జిల్లా మీసేవ కేంద్రంలో దరఖాస్తు చేసుకోవచ్చు.
ఈ సర్టిఫికేట్ {cert[1]} లో జారీ చేయబడుతుంది.
ఫీజు: {cert[2]}
అవసరమైన పత్రాలు: {cert[3]}.
ఆన్‌లైన్ ద్వారా కూడా దరఖాస్తు చేసుకోవచ్చు.
వెబ్‌సైట్: meeseva.telangana.gov.in
హెల్ప్‌లైన్: 1800-123-4567
కార్యాలయ సమయం: ఉ.10 - సా.5 (సోమ-శని)"""
        
        title = f"{cert[0]} - {district}"
        
    elif doc_type == "order":
        dept = random.choice(GOV_COMPONENTS["departments"])
        action = random.choice(GOV_COMPONENTS["actions"])
        official = random.choice(GOV_COMPONENTS["officials"])
        district = random.choice(GOV_COMPONENTS["districts_ts"] + GOV_COMPONENTS["districts_ap"])
        amount = random.choice(["రూ.50 లక్షలు", "రూ.1 కోటి", "రూ.5 కోట్లు", "రూ.10 కోట్లు", "రూ.25 కోట్లు"])
        
        content = f"""తెలంగాణ/ఆంధ్రప్రదేశ్ ప్రభుత్వం {dept} {action}.
{official} అధ్యక్షతన {district}లో సమావేశం జరిగింది.
ఈ సందర్భంగా {amount} నిధులు మంజూరు చేయబడ్డాయి.
ఈ నిర్ణయం రాష్ట్ర ప్రజలకు మేలు చేకూర్చుతుందని ప్రభుత్వం భావిస్తోంది.
కొత్త మార్గదర్శకాల ప్రకారం పథకాలు అమలు చేయబడతాయి.
అన్ని జిల్లాల్లో ఈ ఉత్తర్వులు వర్తిస్తాయి."""
        
        title = f"{dept} - ప్రభుత్వ ఉత్తర్వులు"
        
    elif doc_type == "press_release":
        dept = random.choice(GOV_COMPONENTS["departments"])
        official = random.choice(GOV_COMPONENTS["officials"])
        action = random.choice(GOV_COMPONENTS["actions"])
        district = random.choice(GOV_COMPONENTS["districts_ts"] + GOV_COMPONENTS["districts_ap"])
        
        content = f"""{official} {district}లో {dept} కార్యక్రమంలో పాల్గొన్నారు.
ఈ సందర్భంగా అనేక అభివృద్ధి పథకాలను ప్రారంభించారు.
{action} అని {official} ప్రకటించారు.
రాష్ట్ర అభివృద్ధి కోసం ప్రభుత్వం కట్టుబడి ఉందని తెలిపారు.
ప్రజల సమస్యలను పరిష్కరించడం ప్రభుత్వ ప్రాథమిక బాధ్యత అని పేర్కొన్నారు."""
        
        title = f"ప్రెస్ రిలీజ్ - {district} - {dept}"
        
    else:  # notification
        dept = random.choice(GOV_COMPONENTS["departments"])
        posts = random.choice([50, 100, 200, 500, 1000, 2000])
        last_date = (datetime.now() + timedelta(days=random.randint(15, 45))).strftime("%d-%m-%Y")
        
        content = f"""{dept} నియామకాల నోటిఫికేషన్ విడుదల చేసింది.
మొత్తం {posts} పోస్టులు భర్తీ చేయబడతాయి.
దరఖాస్తు చివరి తేదీ: {last_date}
అర్హత: సంబంధిత డిగ్రీ/డిప్లొమా
వయస్సు: 18-44 సంవత్సరాలు
దరఖాస్తు వెబ్‌సైట్: recruitment.telangana.gov.in
పరీక్ష తేదీ తదుపరి ప్రకటించబడుతుంది."""
        
        title = f"{dept} - నియామకాల నోటిఫికేషన్"
    
    return {
        "id": f"gov_{doc_id:05d}",
        "title": title,
        "content": content,
        "doc_type": doc_type,
        "department": GOV_COMPONENTS["departments"][doc_id % len(GOV_COMPONENTS["departments"])],
        "source": "Synthetic Government Data",
        "date_scraped": datetime.now().isoformat(),
        "date_published": (datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d")
    }


def generate_lit_passage(passage_id: int) -> Dict:
    """Generate a diverse literature passage."""
    passage_type = random.choice(["verse", "proverb", "story", "interpretation"])
    
    if passage_type == "verse" and GOV_COMPONENTS:
        verse_data = random.choice(LIT_COMPONENTS["sample_verses"])
        poet_data = next((p for p in LIT_COMPONENTS["classical_poets"] if p[0] == verse_data[1]), None)
        
        if poet_data:
            content = verse_data[0]
            author = verse_data[1]
            genre = verse_data[2]
            work_title = poet_data[3][0] if poet_data[3] else genre
            period = poet_data[2]
            
            # Add metadata as natural text
            content += f"\n\nఈ {genre} {author} రచించారు. {author} {period}కు చెందిన ప్రసిద్ధ తెలుగు కవి."
        else:
            content = verse_data[0]
            author = verse_data[1]
            genre = verse_data[2]
            work_title = genre
            
    elif passage_type == "proverb":
        proverbs = random.sample(LIT_COMPONENTS["proverbs"], min(5, len(LIT_COMPONENTS["proverbs"])))
        theme = random.choice(LIT_COMPONENTS["themes"])
        
        content = f"తెలుగు సామెతలు - {theme[0]} గురించి:\n\n"
        for p in proverbs:
            content += f"• {p}\n"
        content += f"\nఈ సామెతలు {theme[0]} యొక్క ప్రాముఖ్యతను తెలుపుతాయి. తెలుగు జానపద సాహిత్యంలో సామెతలు ముఖ్యమైన స్థానం కలిగి ఉంటాయి."
        
        author = "జానపద సాహిత్యం"
        genre = "సామెత"
        work_title = "తెలుగు సామెతలు"
        
    elif passage_type == "story":
        theme = random.choice(LIT_COMPONENTS["themes"])
        
        stories = [
            f"""ఒక గ్రామంలో ఇద్దరు స్నేహితులు ఉండేవారు. ఒకరు చాలా కష్టపడి పని చేసేవారు, మరొకరు సోమరిపోతు.
కొన్నేళ్ళ తర్వాత, కష్టపడి పని చేసినవాడు సంపన్నుడయ్యాడు.
సోమరిపోతు అతన్ని చూసి బాధపడ్డాడు.
కానీ స్నేహితుడు అతనికి సహాయం చేసి, కష్టపడి పని చేయడం నేర్పించాడు.
{theme[0]} యొక్క విలువ తెలిపే ఈ కథ {theme[1][0]} మరియు {theme[1][1]} ప్రాముఖ్యతను చూపిస్తుంది.""",
            
            f"""పూర్వం ఒక రాజ్యంలో ఒక మంచి రాజు ఉండేవాడు. ఆయన ప్రజలను తన పిల్లలలా చూసుకునేవాడు.
ఒకరోజు పేద రైతు రాజు దగ్గరకు వచ్చాడు.
రాజు అతని సమస్య విని వెంటనే సహాయం చేశాడు.
ఈ విధంగా {theme[0]} మరియు {theme[1][0]} విలువలను పాటించిన రాజు చరిత్రలో నిలిచిపోయాడు.""",
            
            f"""ఒక చిన్న పిల్లవాడు రోజూ పాఠశాలకు వెళ్ళేవాడు. అతను చాలా తెలివైనవాడు.
ఒకరోజు గురువుగారు అతన్ని మెచ్చుకున్నారు.
అప్పటి నుండి అతను మరింత కష్టపడి చదివాడు.
{theme[1][0]} ద్వారా {theme[0]} సాధించవచ్చని ఈ కథ చెబుతోంది."""
        ]
        
        content = random.choice(stories)
        author = "జానపద కథ"
        genre = "కథ"
        work_title = f"తెలుగు జానపద కథలు - {theme[0]}"
        
    else:  # interpretation
        poet = random.choice(LIT_COMPONENTS["classical_poets"])
        theme = random.choice(LIT_COMPONENTS["themes"])
        
        content = f"""{poet[0]} {poet[2]}కు చెందిన ప్రసిద్ధ తెలుగు కవి.
{poet[1]} అనే బిరుదు కలిగిన ఈ కవి {poet[3][0]} అనే గొప్ప రచన చేశారు.
{poet[0]} రచనల్లో {theme[0]} యొక్క ప్రాముఖ్యత ప్రతిబింబిస్తుంది.
{theme[1][0]} మరియు {theme[1][1]} విలువలను ఆయన తన రచనల్లో చక్కగా చిత్రీకరించారు.
ఈ కవి తెలుగు సాహిత్యానికి అమూల్యమైన సేవ చేశారు."""
        
        author = poet[0]
        genre = "వ్యాఖ్యానం"
        work_title = poet[3][0] if poet[3] else "రచనా సంపుటి"
    
    return {
        "id": f"lit_{passage_id:05d}",
        "title": f"{genre} - {author}",
        "content": content,
        "author": author,
        "genre": genre,
        "work_title": work_title,
        "source": "Synthetic Telugu Literature",
        "date_scraped": datetime.now().isoformat()
    }


# ── News content components ──────────────────────────────────────────────

NEWS_COMPONENTS = {
    "categories": ["రాజకీయాలు", "క్రీడలు", "వ్యాపారం", "సినిమా", "సాంకేతికత", "జాతీయం", "అంతర్జాతీయం", "నేరాలు"],
    "cities_ts": [
        "హైదరాబాద్", "వరంగల్", "కరీంనగర్", "నిజామాబాద్", "ఖమ్మం",
        "నల్గొండ", "మహబూబ్‌నగర్", "ఆదిలాబాద్", "సిద్దిపేట", "సూర్యాపేట"
    ],
    "cities_ap": [
        "విశాఖపట్నం", "విజయవాడ", "తిరుపతి", "గుంటూరు", "నెల్లూరు",
        "కర్నూలు", "రాజమహేంద్రవరం", "కాకినాడ", "అనంతపురం", "ఏలూరు"
    ],
    "politics": {
        "parties": ["తెలుగు దేశం పార్టీ", "భారతీయ జనతా పార్టీ", "వైఎస్ఆర్ కాంగ్రెస్ పార్టీ", "భారత రాష్ట్ర సమితి", "కాంగ్రెస్ పార్టీ", "జనసేన పార్టీ"],
        "events": [
            "ఎన్నికల ప్రచారం ప్రారంభించారు", "బహిరంగ సభ నిర్వహించారు",
            "సమావేశంలో పాల్గొన్నారు", "ప్రకటన విడుదల చేశారు",
            "ధర్నా నిర్వహించారు", "పార్లమెంటులో ప్రసంగించారు",
            "అభివృద్ధి పనులను ప్రారంభించారు", "జనసభలో ప్రసంగించారు"
        ],
        "positions": ["ముఖ్యమంత్రి", "ప్రతిపక్ష నాయకుడు", "మంత్రి", "ఎమ్మెల్యే", "ఎంపీ", "పార్టీ అధ్యక్షుడు"]
    },
    "sports": {
        "games": [
            ("క్రికెట్", ["బ్యాట్స్‌మన్", "బౌలర్", "విక్కెట్", "రన్స్", "ఓవర్లు", "సెంచరీ", "హాఫ్ సెంచరీ", "టెస్ట్ మ్యాచ్"]),
            ("కబడ్డి", ["రైడర్", "డిఫెండర్", "పాయింట్లు", "ప్రో కబడ్డి", "మ్యాచ్"]),
            ("బ్యాడ్మింటన్", ["షటిల్", "సెట్", "ఫైనల్", "సెమీ ఫైనల్", "టోర్నమెంట్"]),
            ("ఫుట్‌బాల్", ["గోల్", "పెనాల్టీ", "హాఫ్ టైమ్", "ఫైనల్ మ్యాచ్"]),
            ("వాలీబాల్", ["సెట్", "పాయింట్లు", "జట్టు", "జాతీయ స్థాయి"]),
            ("హాకీ", ["గోల్", "పెనాల్టీ కార్నర్", "హాఫ్ టైమ్", "ఒలింపిక్స్"])
        ],
        "events": [
            "విజయం సాధించింది", "ఫైనల్‌కు చేరింది", "గోల్డ్ మెడల్ గెలిచింది",
            "కొత్త రికార్డు నెలకొల్పింది", "సెమీ ఫైనల్‌లో ఓడిపోయింది",
            "టోర్నమెంట్‌లో పాల్గొంది", "జాతీయ జట్టుకు ఎంపికైంది"
        ],
        "players": [
            "విరాట్ కోహ్లి", "రోహిత్ శర్మ", "కేఎల్ రాహుల్",
            "పీవీ సింధు", "సయీనా నెహ్వాల్", "ఎంసీ మేరీకోమ్",
            "సునీల్ ఛెత్రీ", "నీరజ్ చోప్రా", "మీరాబాయి చాను"
        ]
    },
    "business": {
        "sectors": ["IT", "ఫార్మా", "రియల్ ఎస్టేట్", "వ్యవసాయం", "ఆటోమొబైల్", "బ్యాంకింగ్", "స్టార్టప్"],
        "terms": [
            ("షేర్ మార్కెట్", "సెన్సెక్స్", "నిఫ్టీ"),
            ("జీడీపీ", "వృద్ధి రేటు", "శాతం"),
            ("ఎగుమతులు", "దిగుమతులు", "వాణిజ్య లోటు"),
            ("పెట్టుబడి", "FDI", "కోట్లు"),
        ],
        "events": [
            "కొత్త ఫ్యాక్టరీ ప్రారంభించింది",
            "వేల కొత్త ఉద్యోగాలు కల్పిస్తామని ప్రకటించింది",
            "లాభాలు రికార్డు స్థాయికి చేరాయి",
            "కొత్త ఉత్పత్తిని విడుదల చేసింది",
            "స్టాక్ ధర పెరిగింది",
            "కొత్త పథకాన్ని ప్రారంభించింది"
        ]
    },
    "cinema": {
        "roles": ["నటుడు", "నటి", "దర్శకుడు", "నిర్మాత", "సంగీత దర్శకుడు"],
        "actors": ["మహేష్ బాబు", "ప్రభాస్", "జూనియర్ ఎన్టీఆర్", "అల్లు అర్జున్", "రామ్ చరణ్",
                   "సమంత", "రష్మిక", "పూజా హెగ్డే", "అనుపమ", "కీర్తి సురేష్"],
        "events": [
            "కొత్త సినిమా విడుదలైంది",
            "బాక్సాఫీస్ వద్ద రికార్డులు సృష్టించింది",
            "కొత్త ప్రాజెక్ట్ ప్రకటించారు",
            "100 కోట్ల క్లబ్‌లో చేరింది",
            "అవార్డు అందుకున్నారు",
            "సినిమా షూటింగ్ ప్రారంభమైంది"
        ]
    },
    "crime": {
        "types": ["దొంగతనం", "మోసం", "ఆన్‌లైన్ మోసం", "ప్రమాదం", "హత్య", "అక్రమ రవాణా"],
        "actions": [
            "పోలీసులు అరెస్ట్ చేశారు", "కేసు నమోదు చేశారు",
            "దర్యాప్తు జరుపుతున్నారు", "నిందితులను పట్టుకున్నారు",
            "ఆస్తులు స్వాధీనం చేశారు", "FIR నమోదు చేశారు"
        ]
    },
    "tech": {
        "topics": ["ఐఫోన్", "యాండ్రాయిడ్", "5G", "AI", "స్టార్‌లింక్", "ఎలక్ట్రిక్ వాహనాలు", "చంద్రయాన్", "ISRO"],
        "events": [
            "కొత్త ఫోన్ విడుదల చేసింది",
            "టెక్నాలజీ ఎక్స్‌పోలో ప్రదర్శించారు",
            "ధర తగ్గించింది",
            "కొత్త అప్‌డేట్ విడుదలైంది",
            "రాకెట్ ప్రయోగం విజయవంతమైంది",
            "ఉపగ్రహాన్ని ప్రయోగించారు"
        ]
    }
}


def generate_news_article(article_id: int) -> Dict:
    """Generate a diverse synthetic Telugu news article."""
    category = random.choice(NEWS_COMPONENTS["categories"])
    city = random.choice(NEWS_COMPONENTS["cities_ts"] + NEWS_COMPONENTS["cities_ap"])
    date_str = (datetime.now() - timedelta(days=random.randint(1, 180))).strftime("%d-%m-%Y")

    if category == "రాజకీయాలు":
        party = random.choice(NEWS_COMPONENTS["politics"]["parties"])
        event = random.choice(NEWS_COMPONENTS["politics"]["events"])
        position = random.choice(NEWS_COMPONENTS["politics"]["positions"])
        second_party = random.choice([p for p in NEWS_COMPONENTS["politics"]["parties"] if p != party])
        content = (
            f"{city}లో {party} {position} {event}.\n"
            f"ఈ సందర్భంగా రాష్ట్ర అభివృద్ధిపై ప్రధానంగా మాట్లాడారు.\n"
            f"ప్రజల సమస్యలను పరిష్కరించడమే తమ లక్ష్యమని {position} పేర్కొన్నారు.\n"
            f"ఈ కార్యక్రమంలో వేలాది మంది పార్టీ కార్యకర్తలు పాల్గొన్నారు.\n"
            f"ఈ సమావేశం {date_str}న జరిగింది.\n"
            f"{second_party} ఈ వ్యాఖ్యలపై తీవ్రంగా విమర్శించింది."
        )
        title = f"{party} - {city}లో {event}"

    elif category == "క్రీడలు":
        game_data = random.choice(NEWS_COMPONENTS["sports"]["games"])
        game_name = game_data[0]
        game_terms = game_data[1]
        sport_event = random.choice(NEWS_COMPONENTS["sports"]["events"])
        player = random.choice(NEWS_COMPONENTS["sports"]["players"])
        score = random.choice(["3-1", "2-0", "156 రన్స్", "21-18", "85 పాయింట్లు"])
        content = (
            f"{game_name}: {player} అద్భుతమైన ప్రదర్శనతో {sport_event}.\n"
            f"ఈ {game_terms[0]} {city}లో జరిగిన {game_terms[-1]}లో {score} తేడాతో గెలిచారు.\n"
            f"{player} ఈ టోర్నమెంట్‌లో {random.choice(game_terms[1:-1])} విభాగంలో ఉత్తమంగా రాణించారు.\n"
            f"ఈ విజయంతో భారత్ ర్యాంకింగ్‌లో ముందుకు దూసుకెళ్ళింది.\n"
            f"ఈ పోటీ {date_str}న జరిగింది."
        )
        title = f"{game_name}: {player} {sport_event}"

    elif category == "వ్యాపారం":
        sector = random.choice(NEWS_COMPONENTS["business"]["sectors"])
        biz_event = random.choice(NEWS_COMPONENTS["business"]["events"])
        amount = random.choice(["రూ.500 కోట్లు", "రూ.1,200 కోట్లు", "రూ.3,000 కోట్లు", "రూ.50 కోట్లు", "రూ.10,000 కోట్లు"])
        pct = random.choice(["12%", "8.5%", "15%", "22%", "5.3%"])
        jobs = random.choice(["1,000", "5,000", "10,000", "2,500", "500"])
        content = (
            f"{city}లో {sector} రంగంలో ప్రముఖ సంస్థ {biz_event}.\n"
            f"ఈ ప్రాజెక్ట్ కోసం {amount} పెట్టుబడి పెట్టనున్నట్లు ప్రకటించింది.\n"
            f"ఈ ప్రాజెక్ట్ ద్వారా {jobs} మందికి ఉద్యోగాలు కల్పించబడతాయి.\n"
            f"గత ఏడాదితో పోలిస్తే {sector} రంగం {pct} వృద్ధి నమోదు చేసింది.\n"
            f"ఈ ప్రకటన {date_str}న చేయబడింది."
        )
        title = f"{sector}: {city}లో {biz_event}"

    elif category == "సినిమా":
        actor = random.choice(NEWS_COMPONENTS["cinema"]["actors"])
        role = random.choice(NEWS_COMPONENTS["cinema"]["roles"])
        cinema_event = random.choice(NEWS_COMPONENTS["cinema"]["events"])
        collection = random.choice(["రూ.100 కోట్లు", "రూ.250 కోట్లు", "రూ.50 కోట్లు", "రూ.500 కోట్లు"])
        director = random.choice(["రాజమౌళి", "త్రివిక్రమ్", "సుకుమార్", "కొరటాల శివ", "ప్రశాంత్ నీల్"])
        content = (
            f"ప్రముఖ {role} {actor} {cinema_event}.\n"
            f"దర్శకుడు {director} తీసిన ఈ సినిమా {collection} కలెక్షన్లు సాధించింది.\n"
            f"ఈ సినిమా {date_str}న విడుదలైంది.\n"
            f"ప్రేక్షకులు మరియు విమర్శకులు ఈ సినిమాకు మంచి రేటింగ్ ఇచ్చారు.\n"
            f"తెలుగు సినీ పరిశ్రమలో ఈ సినిమా కొత్త రికార్డులు సృష్టించింది."
        )
        title = f"{actor} {cinema_event}"

    elif category == "నేరాలు":
        crime_type = random.choice(NEWS_COMPONENTS["crime"]["types"])
        action = random.choice(NEWS_COMPONENTS["crime"]["actions"])
        suspects = random.choice(["ఇద్దరు నిందితులను", "ముగ్గురు నిందితులను", "ఒక నిందితుడిని", "నలుగురు నిందితులను"])
        amount = random.choice(["రూ.5 లక్షలు", "రూ.10 లక్షలు", "రూ.50 లక్షలు", "రూ.1 కోటి"])
        station = random.choice(["బంజారాహిల్స్", "మియాపూర్", "కూకట్‌పల్లి", "మాదాపూర్", "ఎల్బీ నగర్", "ఆమీర్‌పేట్"])
        content = (
            f"{city} {station} పోలీస్ స్టేషన్ పరిధిలో {crime_type} కేసులో {action}.\n"
            f"{suspects} {action}.\n"
            f"నిందితుల నుండి {amount} విలువైన ఆస్తులు స్వాధీనం చేశారు.\n"
            f"ఈ కేసు {date_str}న నమోదైంది.\n"
            f"పోలీసు అధికారులు దర్యాప్తు కొనసాగిస్తున్నారు."
        )
        title = f"{city}: {crime_type} కేసులో {action}"

    elif category == "సాంకేతికత":
        tech = random.choice(NEWS_COMPONENTS["tech"]["topics"])
        tech_event = random.choice(NEWS_COMPONENTS["tech"]["events"])
        price = random.choice(["రూ.15,999", "రూ.29,999", "రూ.49,999", "రూ.79,999", "రూ.1,09,999"])
        content = (
            f"{tech} రంగంలో కొత్త పరిణామం: {tech_event}.\n"
            f"ఈ ఉత్పత్తి ధర {price} గా నిర్ణయించబడింది.\n"
            f"నిపుణులు ఈ పరిణామాన్ని సానుకూలంగా అభిప్రాయపడ్డారు.\n"
            f"భారతదేశంలో ఈ టెక్నాలజీ వేగంగా వ్యాప్తి చెందుతోంది.\n"
            f"ఈ ప్రకటన {date_str}న చేయబడింది."
        )
        title = f"{tech}: {tech_event}"

    elif category == "జాతీయం":
        topic = random.choice(["బడ్జెట్", "ఎన్నికలు", "వాతావరణం", "విద్య", "ఆరోగ్యం", "రక్షణ"])
        content = (
            f"కేంద్ర ప్రభుత్వం {topic} విషయంలో కొత్త ప్రకటన చేసింది.\n"
            f"{city}లో ఈ అంశంపై సమావేశం జరిగింది.\n"
            f"దేశవ్యాప్తంగా ఈ నిర్ణయం ప్రభావం చూపుతుందని నిపుణులు భావిస్తున్నారు.\n"
            f"పార్లమెంటులో ఈ అంశంపై చర్చ జరిగింది.\n"
            f"ఈ ప్రకటన {date_str}న చేయబడింది."
        )
        title = f"జాతీయం: {topic} పై కొత్త ప్రకటన"

    else:  # అంతర్జాతీయం
        country = random.choice(["అమెరికా", "చైనా", "రష్యా", "జపాన్", "ఇంగ్లాండ్", "ఆస్ట్రేలియా", "జర్మనీ"])
        topic = random.choice(["ఆర్థిక సంక్షోభం", "శాంతి చర్చలు", "వాణిజ్య ఒప్పందం", "వాతావరణ మార్పు", "అంతరిక్ష పరిశోధన"])
        content = (
            f"{country}లో {topic} అంశంపై కొత్త పరిణామం చోటు చేసుకుంది.\n"
            f"అంతర్జాతీయ నాయకులు ఈ అంశంపై చర్చలు జరిపారు.\n"
            f"భారతదేశం ఈ విషయంలో తన వైఖరిని స్పష్టం చేసింది.\n"
            f"ఐక్యరాజ్యసమితి ఈ పరిణామాన్ని పరిశీలిస్తోంది.\n"
            f"ఈ సంఘటన {date_str}న జరిగింది."
        )
        title = f"అంతర్జాతీయం: {country}లో {topic}"

    return {
        "id": f"news_{article_id:05d}",
        "title": title,
        "content": content,
        "category": category,
        "source": "Synthetic Telugu News",
        "city": city,
        "date_scraped": datetime.now().isoformat(),
        "date_published": date_str
    }


def save_data(items: List[Dict], domain: str, filename: str):
    """Save generated data."""
    output_dir = Path(f"data/domain/{domain}/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / filename
    
    if domain == "government":
        key = "documents"
    elif domain == "news":
        key = "articles"
    else:
        key = "passages"
    
    data = {
        "metadata": {
            f"total_{key}": len(items),
            "generated_at": datetime.now().isoformat(),
            "source": "Synthetic Generation"
        },
        key: items
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved {len(items)} items to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate scaled synthetic data")
    parser.add_argument("--domain", choices=["government", "literature", "news"], help="Domain to generate")
    parser.add_argument("--count", type=int, default=500, help="Number of items to generate")
    parser.add_argument("--all", action="store_true", help="Generate for all domains")
    
    args = parser.parse_args()
    
    if args.all or args.domain == "government":
        count = args.count if args.domain == "government" else 500
        print(f"\n📝 Generating {count} government documents...")
        docs = [generate_gov_document(i) for i in range(count)]
        save_data(docs, "government", "gov_scaled.json")
    
    if args.all or args.domain == "literature":
        count = args.count if args.domain == "literature" else 200
        print(f"\n📝 Generating {count} literature passages...")
        passages = [generate_lit_passage(i) for i in range(count)]
        save_data(passages, "literature", "lit_scaled.json")
    
    if args.all or args.domain == "news":
        count = args.count if args.domain == "news" else 400
        print(f"\n📝 Generating {count} news articles...")
        articles = [generate_news_article(i) for i in range(count)]
        save_data(articles, "news", "news_scaled.json")
    
    if args.all:
        print("\n✅ Scaled data generation complete!")
        print("   Next: Run QA generation with --all flag")


if __name__ == "__main__":
    main()
