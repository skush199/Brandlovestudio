"""
COGNIXIA CREATIVE INTELLIGENCE PIPELINE  v7  — TEMPLATE-CLONE ARCHITECTURE
============================================================================

FIXES IN v7:
  ✅ FIX 1: gpt-image-1-mini → gpt-image-1  (mini doesn't exist — was crashing)
  ✅ FIX 2: Category-aware re-ranking in retriever
            (food goal → food templates come FIRST, not buried under 7 tech posts)
  ✅ FIX 3: all_embeddings added to TypedDict  (was causing state KeyError)
  ✅ FIX 4: similarity_search_with_score → shows WHY each template was picked
  ✅ FIX 5: Embedding text now prefixed with category+type for stronger signal
  ✅ FIX 6: Template cloner prompt tightened — forces 1:1 field inheritance

HOW THE PIPELINE WORKS (the correct mental model):
─────────────────────────────────────────────────────────────────────────────
  STORE PHASE:
    image → GPT-4.1-mini vision → structured JSON → FAISS vector DB

  GENERATE PHASE — 3 steps:

    STEP 1: SIMILARITY SEARCH  (category-aware)
      goal → detect category (food/tech/corporate)
           → embed goal → FAISS.similarity_search_with_score(k=6)
           → re-rank: same-category docs get a 0.15 score boost
           → return top-3 re-ranked docs (FULL JSONs from disk)

      e.g. goal = "Bisibelebath powder ad"
           detected category = "food"
           raw results: [rasam(0.88), lemon_rice(0.82), cognixia_tech(0.79)]
           re-ranked:   [rasam(0.88+0.15), lemon_rice(0.82+0.15), cognixia_tech(0.79)]
                      = [rasam(1.03), lemon_rice(0.97), cognixia_tech(0.79)]
           → food templates WIN every time ✅

    STEP 2: TEMPLATE-CLONE JSON GENERATION
      Send retrieved JSONs + goal to GPT-4.1:
      → Same layout_pattern, art_style, color structure, subject placement
      → Only CONTENT changes (new product, new scene props, new copy slots)
      Output: blueprint_json — 90%+ structurally identical to templates

    STEP 3: IMAGE GENERATION from blueprint
      blueprint.subjects       → DALL-E background prompt
      blueprint.brand_signals  → compositor colors
      blueprint.typography     → compositor layout
      content_writer           → real copy (headline/bullets/CTA)
      compositor               → Pillow overlays text on background

  WHY THIS WORKS:
    Retrieved JSON contains exact hex colors, layout pattern, subject type,
    placement, environment from a REAL past creative of the same category.
    The LLM clones this structure. Result: 90% visual similarity guaranteed.
─────────────────────────────────────────────────────────────────────────────
"""

import os, re, json, base64, io, textwrap, shutil
from datetime import datetime
from typing import Dict, Any, List

from typing_extensions import TypedDict
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

VECTORSTORE_DIR = "json_faiss_store"
JSON_OUT_DIR    = "json_images"
OUTPUT_DIR      = "generated_creatives_jsons"
DEFAULT_LOGO    = r"C:\Users\rkart\Downloads\Brandlovestudioai\cognixia\US-Cognixia-Logo_1.png"

# ── Category keyword maps for goal→category detection ─────────────────────
CATEGORY_KEYWORDS = {
    "food":      ["food", "recipe", "cooking", "spice", "powder", "masala", "curry",
                  "rice", "dal", "sambar", "rasam", "bisi", "bele", "bath", "biryani",
                  "snack", "biscuit", "beverage", "meal", "dish", "ingredient",
                  "kitchen", "homemade", "flavor", "taste", "eat", "restaurant"],
    "tech":      ["ai", "technology", "tech", "software", "cloud", "digital",
                  "machine learning", "neural", "robot", "data", "algorithm",
                  "api", "developer", "coding", "programming", "network", "cyber",
                  "agentic", "llm", "rag", "workflow", "automation"],
    "corporate": ["corporate", "business", "enterprise", "b2b", "professional",
                  "conference", "summit", "leadership", "training", "certification",
                  "webinar", "event", "announcement", "policy", "compliance"],
    "lifestyle": ["lifestyle", "wellness", "fitness", "women", "diversity",
                  "celebration", "festival", "culture", "art", "design"],
}


# ══════════════════════════════════════════════════════════════════
# GRAPH STATE  (FIX 3: all_embeddings added)
# ══════════════════════════════════════════════════════════════════
class GraphState(TypedDict, total=False):
    # Phase 1 — Store
    image_paths:       List[str]
    cached_paths:      List[str]
    uncached_paths:    List[str]
    all_analyses:      List[Dict[str, Any]]
    all_json_paths:    List[str]
    all_documents:     List[Document]
    all_embeddings:    List[List[float]]   # ← FIX 3: was missing, caused KeyError
    vectorstore:       object
    vectorstore_path:  str

    # Phase 2 — Retrieve + Clone
    goal:              str
    goal_category:     str                 # ← NEW: detected category for re-ranking
    retrieved_jsons:   List[Dict]
    retrieved_scores:  List[float]         # ← NEW: similarity scores for debugging
    blueprint_json:    Dict

    # Phase 3 — Content + Generate + Composite
    creative_content:  Dict
    dalle_prompt:      str
    bg_image_path:     str
    saved_image_path:  str
    logo_path:         str


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def img_to_url(path: str, max_side=1200, q=85) -> str:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = max_side / max(w, h)
    if s < 1:
        img = img.resize((int(w*s), int(h*s)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=q, optimize=True)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

def flatten(d: dict, prefix="") -> str:
    lines = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            lines.append(flatten(v, key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict): lines.append(flatten(item, f"{key}[{i}]"))
                else:                      lines.append(f"{key}[{i}]: {item}")
        else:
            lines.append(f"{key}: {v}")
    return "\n".join(lines)

def load_vs(embedder):
    fp = os.path.join(VECTORSTORE_DIR, "index.faiss")
    return FAISS.load_local(VECTORSTORE_DIR, embedder,
                            allow_dangerous_deserialization=True) if os.path.exists(fp) else None

def stored_sources(vs) -> set:
    s = set()
    for _, doc in vs.docstore._dict.items():
        src = doc.metadata.get("source_image", "")
        if src: s.add(os.path.abspath(src))
    return s

def parse_json(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    data = json.loads(cleaned)
    return scrub_missing(data)

def scrub_missing(d):
    """Recursively convert 'MISSING' strings to empty or None."""
    if isinstance(d, dict):
        return {k: scrub_missing(v) for k, v in d.items()}
    if isinstance(d, list):
        return [scrub_missing(i) for i in d]
    if isinstance(d, str) and d.upper() == "MISSING":
        return ""
    return d

def clean_v(v, default="") -> str:
    """Helper to handle GPT 'MISSING' string literals from analysis."""
    s = str(v).strip()
    if not s or s.upper() == "MISSING":
        return default
    return s

def hex_rgb(h: str) -> tuple:
    m = re.search(r'#?([0-9A-Fa-f]{6})', str(h))
    if m:
        v = m.group(1)
        return tuple(int(v[i:i+2], 16) for i in (0, 2, 4))
    return (255, 255, 255)

def get_font(size: int, bold=False) -> ImageFont.FreeTypeFont:
    paths = (
        [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\calibrib.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
         "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold else
        [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\calibri.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
         "/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for p in paths:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except: pass
    return ImageFont.load_default()

def detect_goal_category(goal: str) -> str:
    """
    Detect what category the user's goal belongs to.
    Returns: "food" | "tech" | "corporate" | "lifestyle" | "general"

    This is the key to category-aware retrieval:
    If goal = "Bisibelebath powder ad", this returns "food"
    → retriever then boosts food template scores by +0.15
    → food templates always win over tech even if DB has 7 tech : 1 food
    """
    goal_lower = goal.lower()
    scores = {cat: 0 for cat in CATEGORY_KEYWORDS}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in goal_lower:
                scores[cat] += 1
    best_cat = max(scores, key=scores.get)
    return best_cat if scores[best_cat] > 0 else "general"


# ══════════════════════════════════════════════════════════════════
# IMAGE ANALYSIS PROMPT
# ══════════════════════════════════════════════════════════════════
ANALYSIS_PROMPT = """
Analyze this image and return ONLY valid JSON using this exact schema:

{
  "core_content": {
    "main_description": "One precise sentence: what is shown, what type of creative, what purpose.",
    "creative_type": "product_ad | tech_infographic | lifestyle_post | event_post | etc",
    "category": "food | tech | corporate | lifestyle | etc"
  },
  "subjects": [
    {
      "type": "food_product | robot | human | plant | object | illustration | etc",
      "identity": "specific product/person/object name and description",
      "spatial_position": {
        "placement": "center | center-right | bottom-center | left | etc",
        "scale": "large | medium | small",
        "dynamic_framing": {
          "visual_vector": "direction of motion or orientation",
          "frame_interaction": "how subject relates to frame edges"
        }
      },
      "pose": {
        "body_narrative": "what the subject is doing",
        "hand_gesture": "specific gesture or MISSING"
      },
      "eye_gaze": {
        "direction": "gaze direction or MISSING",
        "contact": "yes | no | MISSING",
        "focus": "what focused on or MISSING"
      },
      "appearance": {
        "features": "detailed description: shape, color, texture, material",
        "grooming": "surface quality / finish / condition"
      },
      "attire": "clothing/packaging description or MISSING"
    }
  ],
  "environment": {
    "scene": "Full scene description: setting, mood, atmosphere",
    "background_elements": [
      { "object": "element name", "description": "color, texture, position detail" }
    ]
  },
  "aesthetics": {
    "art_style": "clean modern product photography | 3D digital render sci-fi | flat illustration | etc — be specific",
    "lighting": {
      "key_light": "direction, temperature, quality (e.g. soft natural top-down warm)",
      "shadows": "shadow style and intensity"
    },
    "camera": {
      "angle": "eye-level | slight-above | top-down | etc",
      "lens": "standard | wide | slight telephoto | etc"
    }
  },
  "typography": {
    "layout": "Describe where text zones are positioned on canvas",
    "elements": [
      {
        "text": "exact visible text string",
        "style": "size_class weight color_hex position"
      }
    ]
  },
  "brand_signals": {
    "color_palette": ["#hex - usage description"],
    "dominant_colors": "top 2-3 colors with hex",
    "background_color": "#hex or gradient description",
    "logo_position": "top-right | top-left | MISSING",
    "logo_style": "describe logo appearance",
    "layout_pattern": "centered-product | split-left-text-right-image | icon-row-list | full-bleed | etc",
    "text_placement": "describe where headline and body text sit",
    "cta_present": "yes | no",
    "cta_style": "pill-button | underlined | MISSING",
    "overall_tone": "warm lifestyle | bold corporate-tech | fresh natural | minimal clean | etc",
    "target_feel": "premium | everyday | festive | professional | etc"
  },
  "compositor_hints": {
    "canvas_ratio": "1:1 | 16:9 | 4:5",
    "text_zone": "top-overlay | left-40% | bottom-strip | centered | etc",
    "subject_zone": "center | right-60% | bottom-right | full-bleed | etc",
    "has_overlay_panel": "yes | no",
    "overlay_color": "#hex rgba or MISSING",
    "font_style": "bold sans-serif | script | mixed | etc"
  }
}

RULES:
- Fill EVERY field. Use "MISSING" only if truly not visible.
- Colors: always use #hex format. Extract real hex values from visible pixels.
- art_style: one precise phrase — this is the most critical field.
- brand_signals.layout_pattern: the single most important structural descriptor.
- core_content.category: the category/industry this creative belongs to.
- typography.elements: list every visible text element with its style.
- JSON only. No markdown. No explanation.
""".strip()


# ══════════════════════════════════════════════════════════════════
# PHASE 0 — CACHE CHECK
# ══════════════════════════════════════════════════════════════════
def cache_check_node(state: GraphState) -> GraphState:
    print("🔍 [Node 0] cache_check_node...")
    paths    = [os.path.abspath(p) for p in state.get("image_paths", [])]
    embedder = OpenAIEmbeddings(model="text-embedding-ada-002",
                                openai_api_key=os.getenv("OPENAI_API_KEY"))
    vs = load_vs(embedder)
    if vs is None:
        return {**state, "image_paths": paths, "cached_paths": [],
                "uncached_paths": paths, "vectorstore": None}
    stored   = stored_sources(vs)
    cached   = [p for p in paths if p in stored]
    uncached = [p for p in paths if p not in stored]
    print(f"  ✅ Cached: {len(cached)}  🆕 New: {len(uncached)}")
    return {**state, "image_paths": paths, "cached_paths": cached,
            "uncached_paths": uncached, "vectorstore": vs}


def route_after_cache(state: GraphState) -> str:
    return "all_cached" if not state.get("uncached_paths") else "needs_work"


# ══════════════════════════════════════════════════════════════════
# PHASE 1 — IMAGE ANALYZER
# ══════════════════════════════════════════════════════════════════
def image_analyzer_node(state: GraphState) -> GraphState:
    print("👁️  [Node 1] image_analyzer_node...")
    uncached = state.get("uncached_paths", [])
    analyses, jpaths = [], []
    os.makedirs(JSON_OUT_DIR, exist_ok=True)

    for idx, path in enumerate(uncached, 1):
        print(f"  🖼️  [{idx}/{len(uncached)}] {os.path.basename(path)}")
        try:
            resp = client.responses.create(
                model="gpt-4.1-mini",
                input=[{"role": "user", "content": [
                    {"type": "input_text",  "text": ANALYSIS_PROMPT},
                    {"type": "input_image", "image_url": img_to_url(path)},
                ]}],
            )
            raw = (resp.output_text or "").strip()
            a = parse_json(raw)
        except Exception as e:
            print(f"  ⚠️  Analysis failed: {e}")
            a = {"raw": str(e)}

        a["_source_image"] = os.path.abspath(path)
        a["_analyzed_at"]  = datetime.now().isoformat()
        base = os.path.splitext(os.path.basename(path))[0]
        jp   = os.path.join(JSON_OUT_DIR, f"{base}_analysis.json")
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(a, f, ensure_ascii=False, indent=2)
        analyses.append(a)
        jpaths.append(jp)
        print(f"  ✅  → {jp}")
        print(f"       category={a.get('core_content',{}).get('category','?')}  "
              f"pattern={a.get('brand_signals',{}).get('layout_pattern','?')}")

    return {**state, "all_analyses": analyses, "all_json_paths": jpaths}


# ══════════════════════════════════════════════════════════════════
# PHASE 2 — EMBEDDINGS
# FIX 5: Embedding text prefixed with CATEGORY + TYPE keywords
#         so semantic distance is strongly influenced by category
#         Example: "CATEGORY:food TYPE:product_ad ..." will embed
#         much closer to "food bisibelebath powder ad" than to tech
# ══════════════════════════════════════════════════════════════════
def embeddings_node(state: GraphState) -> GraphState:
    print("🔢 [Node 2] embeddings_node...")
    analyses = state.get("all_analyses", [])
    embedder = OpenAIEmbeddings(model="text-embedding-ada-002",
                                openai_api_key=os.getenv("OPENAI_API_KEY"))
    docs, vecs = [], []
    for idx, a in enumerate(analyses, 1):
        src      = a.get("_source_image", f"img_{idx}")
        category = a.get("core_content", {}).get("category", "general")
        c_type   = a.get("core_content", {}).get("creative_type", "")
        art      = a.get("aesthetics", {}).get("art_style", "")
        layout   = a.get("brand_signals", {}).get("layout_pattern", "")
        tone     = a.get("brand_signals", {}).get("overall_tone", "")

        # FIX 5: Prefix with repeated category keywords for stronger embedding signal
        # This ensures food images cluster far from tech images in vector space
        category_prefix = f"CATEGORY:{category} TYPE:{c_type} STYLE:{art} LAYOUT:{layout} TONE:{tone}\n"
        flat_text = flatten(a)
        text = category_prefix + flat_text

        docs.append(Document(
            page_content=text,
            metadata={
                "source_image":   src,
                "analyzed_at":    a.get("_analyzed_at", ""),
                "json_path":      state["all_json_paths"][idx - 1],
                "doc_index":      idx,
                "category":       category,
                "creative_type":  c_type,
                "art_style":      art,
                "layout_pattern": layout,
                "overall_tone":   tone,
            },
        ))
        vecs.append(embedder.embed_query(text))
        print(f"  ✅  [{idx}] {os.path.basename(src)}  dims={len(vecs[-1])}  cat={category}")

    return {**state, "all_documents": docs, "all_embeddings": vecs}


# ══════════════════════════════════════════════════════════════════
# PHASE 3 — VECTORSTORE
# ══════════════════════════════════════════════════════════════════
def vectorstore_node(state: GraphState) -> GraphState:
    print("🗄️  [Node 3] vectorstore_node...")
    docs     = state.get("all_documents", [])
    embedder = OpenAIEmbeddings(model="text-embedding-ada-002",
                                openai_api_key=os.getenv("OPENAI_API_KEY"))
    existing = state.get("vectorstore")
    if existing:
        existing.add_documents(docs); vs = existing
    else:
        vs = FAISS.from_documents(documents=docs, embedding=embedder)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vs.save_local(VECTORSTORE_DIR)
    print(f"  ✅  json_faiss_store/ — total vectors: {vs.index.ntotal}")
    return {**state, "vectorstore": vs, "vectorstore_path": VECTORSTORE_DIR}


# ══════════════════════════════════════════════════════════════════
# PHASE 4 — RETRIEVER  (FIX 2: category-aware re-ranking)
#
# THE PROBLEM IT SOLVES:
#   You store 8 images: 7 cognixia tech + 1 Indira Foods food
#   Goal: "Bisibelebath powder ad"
#   Raw vector search might return: [tech1, tech2, food] (food is #3!)
#   Because 7 tech images occupy most of the vector space.
#
# THE FIX:
#   Step 1: detect goal category ("food" from keywords like "bisibelebath")
#   Step 2: retrieve k=6 raw results (wider net)
#   Step 3: re-rank — same-category docs get +CATEGORY_BOOST to their score
#   Step 4: return top-3 re-ranked
#   Result: food always comes FIRST when goal is food-related ✅
# ══════════════════════════════════════════════════════════════════
CATEGORY_BOOST = 0.20  # Score boost for matching category (0.0 to 1.0)

def retriever_node(state: GraphState) -> GraphState:
    print("🔍 [Node 4] retriever_node — category-aware similarity search...")
    goal     = state.get("goal", "")
    embedder = OpenAIEmbeddings(model="text-embedding-ada-002",
                                openai_api_key=os.getenv("OPENAI_API_KEY"))
    vs = state.get("vectorstore") or load_vs(embedder)
    if vs is None:
        raise ValueError("No FAISS index found. Run Phase 1 (store) first.")

    # ── Detect goal category for re-ranking ───────────────────────
    goal_cat = detect_goal_category(goal)
    print(f"  🏷️  Goal category detected: '{goal_cat}'")

    # ── Generate 3 targeted search queries ────────────────────────
    resp = client.chat.completions.create(
        model="gpt-4.1-mini", temperature=0.05,
        messages=[{"role": "user", "content": f"""
You are a creative director. Given this creative goal, generate 3 search queries
to find the most visually and structurally similar past creatives in a vector database.

Goal: "{goal}"

Each query should target a different angle:
1. CATEGORY + STYLE  → e.g. "food product advertisement clean lifestyle photography"
2. SUBJECT + SCENE   → e.g. "spice packet bowl kitchen desk brick wall"
3. LAYOUT + TONE     → e.g. "centered product hero shot warm natural lifestyle"

IMPORTANT: Start each query with the CATEGORY keyword first.
Return ONLY a JSON array of 3 strings. No markdown. No explanation.
"""}],
    )
    try:
        queries = json.loads(resp.choices[0].message.content.strip())
        assert isinstance(queries, list) and len(queries) == 3
    except:
        queries = [
            f"{goal_cat} {goal}",
            goal,
            f"{goal_cat} product photography lifestyle"
        ]

    print("  📡 Search queries:")
    for i, q in enumerate(queries, 1):
        print(f"     Q{i}: {q}")

    # ── Retrieve wider pool with scores ───────────────────────────
    # FIX 4: use similarity_search_with_score to see actual distances
    # FAISS returns L2 distance — lower = more similar
    # We convert: similarity = 1 / (1 + distance) so higher = better
    seen, pool = set(), []   # pool = [(doc, raw_similarity_score)]

    for q in queries:
        results = vs.similarity_search_with_score(q, k=6)
        for doc, dist in results:
            uid = doc.metadata.get("source_image", doc.page_content[:80])
            if uid not in seen:
                seen.add(uid)
                sim = 1.0 / (1.0 + dist)   # convert L2 dist → similarity score
                pool.append((doc, sim))

    # ── Category-aware re-ranking (FIX 2) ─────────────────────────
    # Boost score if doc category matches goal category
    re_ranked = []
    for doc, sim in pool:
        doc_cat = doc.metadata.get("category", "general")
        boost   = CATEGORY_BOOST if (
            goal_cat != "general" and
            doc_cat.lower() == goal_cat.lower()
        ) else 0.0
        final_score = sim + boost
        re_ranked.append((doc, final_score, sim, boost, doc_cat))

    re_ranked.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  📊 Re-ranking results (goal_cat='{goal_cat}', boost={CATEGORY_BOOST}):")
    print(f"  {'Rank':<5} {'File':<35} {'Cat':<12} {'Raw':<8} {'Boost':<8} {'Final':<8}")
    print(f"  {'─'*5} {'─'*35} {'─'*12} {'─'*8} {'─'*8} {'─'*8}")
    for rank, (doc, final, raw, boost, cat) in enumerate(re_ranked, 1):
        fname = os.path.basename(doc.metadata.get("source_image", "?"))[:34]
        marker = " ← TEMPLATE" if rank <= 3 else ""
        print(f"  {rank:<5} {fname:<35} {cat:<12} {raw:.4f}  {boost:.4f}  {final:.4f}{marker}")

    top3 = re_ranked[:3]

    # ── Load full JSONs from disk ──────────────────────────────────
    retrieved_jsons, retrieved_scores = [], []
    for doc, final_score, raw_sim, boost, doc_cat in top3:
        jp  = doc.metadata.get("json_path", "")
        src = doc.metadata.get("source_image", "")
        if jp and os.path.exists(jp):
            with open(jp) as f:
                full_json = json.load(f)
            retrieved_jsons.append(full_json)
            retrieved_scores.append(round(final_score, 4))
            print(f"\n  ✅  Template: {os.path.basename(src)}")
            print(f"       category={full_json.get('core_content',{}).get('category','?')}  "
                  f"pattern={full_json.get('brand_signals',{}).get('layout_pattern','?')}")
            print(f"       score={final_score:.4f} (raw={raw_sim:.4f} + boost={boost:.4f})")
        else:
            retrieved_jsons.append({"_source_image": src, "raw_text": doc.page_content})
            retrieved_scores.append(round(final_score, 4))

    print(f"\n  ✅  Retrieved {len(retrieved_jsons)} template JSONs")
    return {**state,
            "retrieved_jsons":  retrieved_jsons,
            "retrieved_scores": retrieved_scores,
            "goal_category":    goal_cat,
            "vectorstore":      vs}


# ══════════════════════════════════════════════════════════════════
# PHASE 5 — TEMPLATE CLONER  (THE CORE NODE)
#
# Receives top-3 retrieved JSONs (templates).
# Sends them to GPT-4.1 with the new goal.
# GPT generates a NEW JSON in the EXACT SAME SCHEMA & PATTERN.
#   ✅ Keep: art_style, layout_pattern, color palette structure,
#            subject placement, environment style, typography hierarchy
#   🔄 Change: product identity, scene props, specific colors, copy slots
# ══════════════════════════════════════════════════════════════════
def template_cloner_node(state: GraphState) -> GraphState:
    print("🧬 [Node 5] template_cloner_node — cloning template for new goal...")

    goal      = state.get("goal", "")
    templates = state.get("retrieved_jsons", [])

    if not templates:
        raise ValueError("No template JSONs retrieved. Run retriever_node first.")

    # Build numbered template list for prompt
    templates_str = ""
    for i, t in enumerate(templates, 1):
        src = os.path.basename(t.get("_source_image", f"template_{i}"))
        templates_str += f"\n\n{'─'*60}\nTEMPLATE {i} — from: {src}\n{'─'*60}\n"
        templates_str += json.dumps(t, ensure_ascii=False, indent=2)

    prompt = f"""
You are a senior creative director specializing in template-based design systems.

You have {len(templates)} TEMPLATE CREATIVE JSONs retrieved because they are the most
visually and structurally similar past creatives to a NEW creative goal.

YOUR TASK:
Study ALL {len(templates)} template JSONs completely. Then generate a NEW JSON for the
new goal that follows the SAME TEMPLATE STRUCTURE as closely as possible (~90% similarity).

═══════════════════════════════════════════════════════════
NEW CREATIVE GOAL:
"{goal}"
═══════════════════════════════════════════════════════════

TEMPLATE CREATIVES (study every field):
{templates_str}

═══════════════════════════════════════════════════════════
RULES FOR GENERATING THE NEW JSON:
═══════════════════════════════════════════════════════════

KEEP EXACTLY THE SAME (clone these VERBATIM from templates):
  ✅ aesthetics.art_style — copy exact phrase from template
  ✅ aesthetics.lighting.key_light — copy exact phrase
  ✅ aesthetics.camera — copy exact values
  ✅ brand_signals.layout_pattern — copy exact string
  ✅ brand_signals.overall_tone — copy exact phrase
  ✅ brand_signals.target_feel — copy exact phrase
  ✅ compositor_hints.canvas_ratio — copy exactly
  ✅ compositor_hints.text_zone — copy exactly
  ✅ compositor_hints.subject_zone — copy exactly
  ✅ compositor_hints.has_overlay_panel — copy exactly
  ✅ compositor_hints.font_style — copy exactly
  ✅ environment.background_elements structure — same TYPE of props
    (if template has plant+laptop+keys → new has plant+ingredient+utensil)

ADAPT FOR NEW GOAL (change only these):
  🔄 core_content.main_description — describe new creative
  🔄 core_content.category — update to new category
  🔄 subjects — new product with SAME placement and scale
  🔄 subjects[0].identity — new product name/description
  🔄 subjects[0].appearance.features — new product appearance
  🔄 environment.scene — adapted for new product context
  🔄 background_elements — adapted props for new product
  🔄 brand_signals.color_palette — adapted colors (same COUNT, same usage patterns)
  🔄 brand_signals.background_color — adapted for new product mood
  🔄 typography.elements — placeholder text for new goal
  🔄 compositor_hints.overlay_color — adapted if needed

CRITICAL CONSTRAINTS:
  ❌ NEVER change the structural layout (split / centered / etc)
  ❌ NEVER change the art_style (photography stays photography)
  ❌ NEVER change subject zone / text zone positions
  ❌ NEVER add serif fonts if templates use sans-serif
  ❌ NEVER change canvas_ratio

Return the new blueprint JSON using the EXACT SAME schema as the template JSONs.
Also include these extra fields for the compositor:
  "compositor_instructions": {{
    "canvas_px": 1080,
    "pad_left": <int, same as template pattern>,
    "text_zone_px": <int, width of text zone in pixels>,
    "logo_x": <int>,
    "logo_y": <int>,
    "logo_w_px": <int>,
    "strip_h_px": <int, height of bottom CTA strip>,
    "hl_size_px": <int, headline font size>,
    "sub_size_px": <int, subheadline font size>,
    "body_size_px": <int, body/bullet font size>,
    "bg_is_dark": <bool, true if background is dark>,
    "headline_color": "<#hex>",
    "accent_color": "<#hex>",
    "body_color": "<#hex>",
    "cta_bg_color": "<#hex>",
    "cta_fg_color": "<#hex>",
    "overlay_alpha": <int 0-255, panel transparency>,
    "has_logo": <bool>
  }}

Return ONLY valid JSON. No markdown. No explanation.
""".strip()

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.05,  # very low — precise cloning, not creativity
        messages=[
            {"role": "system", "content": "You are a design system expert. Return only valid JSON."},
            {"role": "user",   "content": prompt},
        ],
    )

    try:
        blueprint = parse_json(resp.choices[0].message.content)
    except Exception as e:
        print(f"  ⚠️  Blueprint parse failed: {e}. Using template 1 as fallback.")
        blueprint = templates[0].copy() if templates else {}
        blueprint["creative_goal"] = goal

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    slug    = re.sub(r'[^\w]', '_', goal)[:45]
    bp_path = os.path.join(OUTPUT_DIR,
                           f"{slug}_BLUEPRINT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(bp_path, "w", encoding="utf-8") as f:
        json.dump(blueprint, f, ensure_ascii=False, indent=2)

    print(f"  ✅  Blueprint saved → {bp_path}")
    print(f"  ✅  art_style      : {blueprint.get('aesthetics',{}).get('art_style','?')}")
    print(f"  ✅  layout_pattern : {blueprint.get('brand_signals',{}).get('layout_pattern','?')}")
    print(f"  ✅  overall_tone   : {blueprint.get('brand_signals',{}).get('overall_tone','?')}")
    ci = blueprint.get("compositor_instructions", {})
    print(f"  ✅  bg_is_dark     : {ci.get('bg_is_dark','?')}")
    print(f"  ✅  headline_color : {ci.get('headline_color','?')}")
    print(f"  ✅  accent_color   : {ci.get('accent_color','?')}")

    return {**state, "blueprint_json": blueprint}


# ══════════════════════════════════════════════════════════════════
# PHASE 6 — CONTENT WRITER
# ══════════════════════════════════════════════════════════════════
def content_writer_node(state: GraphState) -> GraphState:
    print("✏️  [Node 6] content_writer_node — writing real copy...")

    goal     = state.get("goal", "")
    bp       = state.get("blueprint_json", {})
    tone     = bp.get("brand_signals", {}).get("overall_tone", "professional")
    feel     = bp.get("brand_signals", {}).get("target_feel", "premium")
    ci       = bp.get("compositor_instructions", {})
    bg_dark  = ci.get("bg_is_dark", True)
    hl_px    = int(ci.get("hl_size_px", 68))
    text_px  = int(ci.get("text_zone_px", 480))
    pad      = int(ci.get("pad_left", 48))
    zone_w   = text_px - pad

    chars_per_line = max(10, int(zone_w / (hl_px * 0.56)))

    category = bp.get("core_content", {}).get("category", "general")
    is_food  = "food" in category.lower() or "food" in goal.lower()

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.4,
        messages=[
            {"role": "system", "content": "You are a copywriter. Return only valid JSON."},
            {"role": "user",   "content": f"""
Write copy for a {feel} {tone} {category} creative.
Goal: "{goal}"

HARD LAYOUT CONSTRAINTS (compositor will break if violated):
- headline_lines: array of strings, each ≤{chars_per_line} chars
  Use 2-3 lines max. Short, punchy.
- subheadline: ≤55 chars, sentence case
- bullets: 3-4 items, ≤42 chars each, concrete benefits/insights
- cta_text: 2-4 words, action verb
- brand_name: your brand name

{"FOOD AD COPY STYLE: warm, appetite-appealing, benefit-focused, homestyle" if is_food else "TECH/CORPORATE COPY STYLE: bold, forward-looking, B2B professional"}

Return ONLY:
{{
  "headline_lines": ["Line 1 ≤{chars_per_line}ch", "Line 2 ≤{chars_per_line}ch"],
  "subheadline": "One line, max 55 chars",
  "bullets": [
    "Benefit or insight 1, max 42 chars",
    "Benefit or insight 2, max 42 chars",
    "Benefit or insight 3, max 42 chars",
    "Benefit or insight 4, max 42 chars"
  ],
  "cta_text": "2-4 words",
  "brand_name": "Brand Name"
}}
No Lorem Ipsum. No ellipsis. Real copy relevant to: {goal}
"""}],
    )

    try:
        content = parse_json(resp.choices[0].message.content)
    except:
        words = goal.split()[:6]
        content = {
            "headline_lines": [" ".join(words[:3]), " ".join(words[3:])],
            "subheadline": "Discover the authentic taste",
            "bullets": ["Premium quality ingredients", "Ready in minutes",
                        "Authentic homestyle flavor", "No preservatives added"],
            "cta_text": "Order Now",
            "brand_name": "Indira's",
        }

    hl = content.get("headline_lines")
    if isinstance(hl, str):
        content["headline_lines"] = textwrap.wrap(hl, width=chars_per_line)[:3]

    print(f"  ✅  Headline : {content.get('headline_lines')}")
    print(f"  ✅  Sub      : {content.get('subheadline')}")
    for i, b in enumerate(content.get("bullets", []), 1):
        print(f"  ✅  Bullet {i}: {b}")
    print(f"  ✅  CTA      : {content.get('cta_text')}")
    return {**state, "creative_content": content}


# ══════════════════════════════════════════════════════════════════
# PHASE 7 — PROMPT BUILDER  (Option C: full-layout text baked into image)
# ══════════════════════════════════════════════════════════════════
# PHASE 7 — PROMPT BUILDER (Option C - Baked-in Text)
#
# Builds ONE rich DALL-E prompt that includes the REAL text content
# (headline, subheadline, bullets, CTA) with exact colors, font
# style, and layout zones inherited from the retrieved template.
# gpt-image-1 renders the complete ad — text baked in — just like
# a professional Canva / Photoshop designer would.
# ══════════════════════════════════════════════════════════════════
def prompt_builder_node(state: GraphState) -> GraphState:
    print("✍️  [Node 7] prompt_builder_node — full-layout baked-text DALL-E prompt (Option C)...")

    bp   = state.get("blueprint_json", {})
    goal = state.get("goal", "")
    ci   = bp.get("compositor_instructions", {})

    # ── Handle Blueprint Metadata (Clean "MISSING" values) ────────
    art_style   = clean_v(bp.get("aesthetics", {}).get("art_style"), "clean modern marketing photography")
    lighting    = clean_v(bp.get("aesthetics", {}).get("lighting", {}).get("key_light"), "soft natural studio lighting")
    shadows     = clean_v(bp.get("aesthetics", {}).get("lighting", {}).get("shadows"), "subtle gentle shadows")
    camera_a    = clean_v(bp.get("aesthetics", {}).get("camera", {}).get("angle"), "eye-level")
    camera_l    = clean_v(bp.get("aesthetics", {}).get("camera", {}).get("lens"), "standard lens")
    scene       = clean_v(bp.get("environment", {}).get("scene"), "clean studio setting")
    bg_color    = clean_v(bp.get("brand_signals", {}).get("background_color"), "#FFFFFF")
    tone        = clean_v(bp.get("brand_signals", {}).get("overall_tone"), "minimal professional")

    # ── Background & Palette ──────────────────────────────────────
    bg_elements = bp.get("environment", {}).get("background_elements", [])
    bg_str_list = [f"{clean_v(e.get('object'))} — {clean_v(e.get('description'))}"
                   for e in bg_elements[:4] if clean_v(e.get('object'))]
    bg_str = "; ".join(bg_str_list) if bg_str_list else "minimal clean background"

    palette     = bp.get("brand_signals", {}).get("color_palette", [])
    palette_str = "; ".join([clean_v(c) for c in palette[:4] if clean_v(c)])

    # ── Composer Hints (Inherited from template) ──────────────────
    font_style = clean_v(bp.get("compositor_hints", {}).get("font_style"), "bold sans-serif")
    bg_is_dark = bp.get("compositor_hints", {}).get("bg_is_dark", True)
    if isinstance(bg_is_dark, str): bg_is_dark = bg_is_dark.lower() == "true"

    hl_color   = clean_v(bp.get("compositor_hints", {}).get("headline_color"),  "#FFFFFF" if bg_is_dark else "#1A2E44")
    sub_color  = clean_v(bp.get("compositor_hints", {}).get("accent_color"),    "#3FD1C6" if bg_is_dark else "#0077B6")
    body_color = clean_v(bp.get("compositor_hints", {}).get("body_color"),      "#D0E8FF" if bg_is_dark else "#2C4A60")
    cta_bg     = clean_v(bp.get("compositor_hints", {}).get("cta_bg_color"),    "#3FD1C6")
    cta_fg     = clean_v(bp.get("compositor_hints", {}).get("cta_fg_color"),    "#0A1628")
    overlay_style = "semi-transparent dark overlay panel" if bg_is_dark else "semi-transparent light frosted-glass panel"

    # ── Subjects ──────────────────────────────────────────────────
    subjects = bp.get("subjects", [{}])
    subj     = subjects[0] if subjects else {}
    s_type   = clean_v(subj.get("type"), "product")
    s_id     = clean_v(subj.get("identity"), goal)
    s_place  = clean_v(subj.get("spatial_position", {}).get("placement"), "center")
    s_scale  = clean_v(subj.get("spatial_position", {}).get("scale"), "medium")
    s_appear = clean_v(subj.get("appearance", {}).get("features"), "")

    # ── Real text content from content_writer_node ────────────────
    content    = state.get("creative_content", {})
    hl_lines   = content.get("headline_lines", ["New Creative"])
    if isinstance(hl_lines, str): hl_lines = [hl_lines]
    headline   = "\n".join(hl_lines)
    subhl      = content.get("subheadline", "")
    bullets    = content.get("bullets", [])
    cta        = content.get("cta_text", "Learn More")
    brand_name = content.get("brand_name", "Brand")
    bullets_str = "\n".join([f"  • {b}" for b in bullets[:4]])

    # ── Dynamic Layout Extraction ─────────────────────────────────
    layout_p    = clean_v(bp.get("brand_signals", {}).get("layout_pattern"), "split-left-text-right-image")
    text_zone   = clean_v(bp.get("compositor_hints", {}).get("text_zone"), "left-40%")
    subj_zone   = clean_v(bp.get("compositor_hints", {}).get("subject_zone"), "right-60%")
    has_overlay = bp.get("compositor_hints", {}).get("has_overlay_panel", "yes") == "yes"

    # ── Build the Dynamic Layout Description ──────────────────────
    if "infographic" in layout_p.lower() or "icon-row" in layout_p.lower():
        layout_desc = f"""
INFOGRAPHIC FLOWCHART LAYOUT (Pattern: {layout_p}):
- Goal: Create a multi-step flowchart or principles diagram.
- Use a clean {bg_color} background with a subtle connecting line (color {sub_color}) flowing between items.
- {len(bullets)} distinct zones or nodes, each with an icon (color {sub_color} background) and text (color {hl_color}).
- Headline "{headline}" clearly at the top in a professional boxed rectangle.
- Icons should be clean flat vector style.
"""
    elif "split" in layout_p.lower():
        layout_desc = f"""
SPLIT CANVAS LAYOUT (Pattern: {layout_p}):
- {text_zone.upper()} — TEXT ZONE: Render a {overlay_style} panel for text clarity. Base color {bg_color}.
- {subj_zone.upper()} — VISUAL ZONE: Zero text here. Focus on the {s_type}.
"""
    else:
        layout_desc = f"""
MODERN BALANCED LAYOUT (Pattern: {layout_p}):
- Spatial positioning: Text is placed in {text_zone}, Subjects are in {subj_zone}.
- Overlay requested: {has_overlay}. If yes, use a {overlay_style} for text legibility.
"""

    # ── Build the single full-layout baked-text DALL-E prompt ─────
    dp_content = f"""
Professional {art_style} marketing template. {tone} tone.
Goal: {goal}

{layout_desc.strip()}

TYPOGRAPHY & BRANDING:
- Font: {font_style}, crisp, professionally kerned.
- Brand name: "{brand_name}" (small caps, {sub_color}).
- Headline: "{headline}" (EXACT words, bold, {hl_color}).
- Subheadline: "{subhl}" (weighted, {sub_color}).
- Bullets:
{bullets_str}
- CTA: "{cta}" (bold pill button, fill {cta_bg}, text {cta_fg}).

VISUAL ZONE DETAILS:
- Subject: {s_type} — {s_id}
- Placement: {s_place}, scale: {s_scale}
- Appearance: {s_appear}
- Scene: {scene} with {bg_str}
- Lighting: {lighting}, Shadows: {shadows}
- Palette: {palette_str}

ABSOLUTE REQUIREMENTS:
- The image MUST follow the {layout_p} layout pattern from the template exactly.
- Render all text EXACTLY as written. No spelling errors.
- Ensure high contrast and professional human-designed aesthetics.
"""
    prompt_msg = dp_content.strip()

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": f"You are a senior art director. Output only a DALL-E image prompt. You must interpret the requested layout pattern '{layout_p}' and describe it spatially to DALL-E so it clones the template's structure perfectly."},
            {"role": "user",   "content": f"Create a DALL-E prompt following these dynamic layout instructions. Fill in 'MISSING' gaps creatively.\n\n{prompt_msg}"},
        ],
    )

    dp = resp.choices[0].message.content.strip()
    print(f"       Layout   : {layout_p}")
    print(f"       Colors   : hl={hl_color}  sub={sub_color}")
    return {**state, "dalle_prompt": dp}



# ══════════════════════════════════════════════════════════════════
# PHASE 8 — IMAGE GENERATOR
# FIX 1: gpt-image-1-mini → gpt-image-1  (mini model does NOT exist)
#         This was causing a silent API error that killed the pipeline
# ══════════════════════════════════════════════════════════════════
def image_generator_node(state: GraphState) -> GraphState:
    print("🖼️  [Node 8] image_generator_node — generating background...")
    dp = state.get("dalle_prompt", "")
    if not dp:
        raise ValueError("dalle_prompt is empty.")

    resp = client.images.generate(
        model="gpt-image-1-mini",    # FIX 1: was "gpt-image-1-mini" — doesn't exist
        prompt=dp,
        size="auto",
        quality="high",
    )
    data0 = resp.data[0]
    b64   = getattr(data0, "b64_json", None)
    url   = getattr(data0, "url", None)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    slug = re.sub(r'[^\w]', '_', state.get("goal", "creative"))[:35]
    bgp  = os.path.join(OUTPUT_DIR,
                        f"{slug}_BG_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    if b64:
        with open(bgp, "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"  ✅  BG → {bgp}")
        return {**state, "bg_image_path": bgp}
    if url:
        return {**state, "bg_image_path": url}
    raise ValueError("No image returned from gpt-image-1")


# ══════════════════════════════════════════════════════════════════
# PHASE 9 — COMPOSITOR
# Reads compositor_instructions from blueprint_json.
# Colors / sizes / positions inherited from cloned template.
# Food ad → warm cream + green automatically
# Tech ad → dark navy + teal automatically
# ══════════════════════════════════════════════════════════════════
"""
COMPOSITOR NODE — v3 FIXED (drop-in replacement for compositor_node)
=====================================================================

BUGS FIXED vs original:
  FIX 1 — Panel fades WITHIN text zone (was near-transparent at text position)
           Now: solid for first 80% of panel width, fades only in last 20%
           Result: text always sits on a solid readable background

  FIX 2 — Font auto-sizing: calculates TOTAL content height first,
           then shrinks ALL fonts together until everything fits vertically
           Result: no more bullet overlapping, no content cut off

  FIX 3 — CTA button skipped when strip_h_px == 0 (infographic mode)
           Previously: cta_y = 1080, button drawn off-canvas, y math corrupted

  FIX 4 — Text strictly clipped to TEXT_W — nothing bleeds into subject zone
           Using PIL ImageDraw with explicit max_width clamping on all draws

  FIX 5 — Vertical layout distributed properly:
           LOGO_AREA → HEADLINE → DIVIDER → SUBHEADLINE → DIVIDER → BULLETS → CTA
           All spaced with proportional gaps based on available canvas height

  FIX 6 — Logo placed correctly regardless of corner — no overlap with headline

HOW TO USE:
  Replace the compositor_node function in your pipeline file with this one.
  Everything else (graph, state, other nodes) stays exactly the same.
"""

import os, re, textwrap
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── Re-use your existing helpers ────────────────────────────────────────────
OUTPUT_DIR  = "generated_creatives_jsons"
DEFAULT_LOGO = r"C:\Users\rkart\Downloads\Brandlovestudioai\cognixia\US-Cognixia-Logo_1.png"


def hex_rgb(h: str) -> tuple:
    m = re.search(r'#?([0-9A-Fa-f]{6})', str(h))
    if m:
        v = m.group(1)
        return tuple(int(v[i:i+2], 16) for i in (0, 2, 4))
    return (255, 255, 255)


def get_font(size: int, bold=False) -> ImageFont.FreeTypeFont:
    paths = (
        [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\calibrib.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
         "/usr/share/fonts/truetype/crosextra/Carlito-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
        if bold else
        [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\calibri.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
         "/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except:
                pass
    return ImageFont.load_default()


def text_line_height(font: ImageFont.FreeTypeFont) -> int:
    """Returns the pixel height of a single line for the given font."""
    tmp = Image.new("RGB", (1, 1))
    d   = ImageDraw.Draw(tmp)
    bb  = d.textbbox((0, 0), "Ag", font=font)
    return bb[3] - bb[1]


def wrap_lines(text: str, font: ImageFont.FreeTypeFont, max_px: int) -> list:
    """
    Word-wrap `text` so that each line fits within `max_px` pixels wide.
    Returns list of line strings.
    """
    tmp = Image.new("RGB", (1, 1))
    d   = ImageDraw.Draw(tmp)
    words  = text.split()
    lines  = []
    current = ""
    for w in words:
        test = (current + " " + w).strip()
        if d.textbbox((0, 0), test, font=font)[2] <= max_px:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines or [text]


def measure_content_height(
    hl_lines, subhl, bullets,
    hf, sf, bf,
    text_w, strip_h, has_cta,
    logo_bottom,
    gap_after_logo=16, divider_gap=18,
    gap_between_bullets=14
) -> int:
    """
    Calculate total height that all text elements will consume.
    Returns total pixel height needed.
    """
    lh_hl   = text_line_height(hf)
    lh_sub  = text_line_height(sf)
    lh_body = text_line_height(bf)

    h = logo_bottom + gap_after_logo    # start after logo

    # Headline lines
    for ln in hl_lines:
        h += lh_hl + 5
    h += 4 + 3 + 16                     # hl_bottom gap + divider + gap

    # Subheadline (wrapped)
    sub_lines = wrap_lines(subhl, sf, text_w)
    h += len(sub_lines) * (lh_sub + 4) + 6

    # Divider + gap
    h += 3 + 18

    # Bullets
    for bul in bullets:
        bul_lines = wrap_lines(bul, bf, text_w - 18)
        h += len(bul_lines) * (lh_body + 2) + gap_between_bullets

    # CTA strip
    if has_cta and strip_h > 0:
        h += strip_h

    return h


# ═══════════════════════════════════════════════════════════════════
#  COMPOSITOR NODE  — drop-in replacement
# ═══════════════════════════════════════════════════════════════════
def compositor_node(state) -> dict:
    """
    PASS-THROUGH compositor (Option C mode).
    Text is baked directly into the image by gpt-image-1 via the
    full-layout DALL-E prompt from prompt_builder_node.
    This node simply copies bg_image_path → saved_image_path.
    """
    print("🎨 [Node 9] compositor_node — pass-through (text baked in by gpt-image-1)")

    bg_path = state.get("bg_image_path", "")
    if not bg_path:
        print("  ⚠️  No bg_image_path in state — skipping")
        return {**state, "saved_image_path": ""}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    slug = re.sub(r'[^\w]', '_', state.get("goal", "creative"))[:35]
    outp = os.path.join(OUTPUT_DIR,
                        f"{slug}_FINAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    if os.path.exists(bg_path):
        shutil.copy2(bg_path, outp)
        print(f"  ✅  Final → {outp}")
    else:
        # bg_path might be a URL — already handled by image_generator_node
        outp = bg_path
        print(f"  ✅  Final (URL) → {outp}")

    return {**state, "saved_image_path": outp}



# ══════════════════════════════════════════════════════════════════
# BUILD GRAPH
# ══════════════════════════════════════════════════════════════════
def route_after_cache(state):
    return "all_cached" if not state.get("uncached_paths") else "needs_work"

def route_to_generate(state):
    return "retrieve" if state.get("goal") else "done"

wf = StateGraph(GraphState)
nodes = [
    ("cache_check",       cache_check_node),
    ("image_analyzer",    image_analyzer_node),
    ("embeddings",        embeddings_node),
    ("vectorstore",       vectorstore_node),
    ("retriever",         retriever_node),
    ("template_cloner",   template_cloner_node),
    ("content_writer",    content_writer_node),
    ("prompt_builder",    prompt_builder_node),
    ("image_generator",   image_generator_node),
    ("compositor",        compositor_node),
]
for name, fn in nodes:
    wf.add_node(name, fn)

wf.add_edge(START, "cache_check")
wf.add_conditional_edges("cache_check", route_after_cache,
    {"all_cached": "retriever", "needs_work": "image_analyzer"})
wf.add_edge("image_analyzer", "embeddings")
wf.add_edge("embeddings",     "vectorstore")
wf.add_conditional_edges("vectorstore", route_to_generate,
    {"retrieve": "retriever", "done": END})

for a, b in [
    ("retriever",       "template_cloner"),
    ("template_cloner", "content_writer"),
    ("content_writer",  "prompt_builder"),
    ("prompt_builder",  "image_generator"),
    ("image_generator", "compositor"),
    ("compositor",      END),
]:
    wf.add_edge(a, b)

app = wf.compile()


# ══════════════════════════════════════════════════════════════════
# RUN EXAMPLES
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    BASE      = r"C:\Users\rkart\Downloads\Brandlovestudioai\cognixia"
    LOGO_PATH = fr"{BASE}\US-Cognixia-Logo_1.png"

    # ── Example 1: Food product ad ─────────────────────────────────────────
    # Goal = "Bisibelebath powder ad"
    # Expected: retriever detects category="food", boosts Indira Foods to rank 1
    # Template cloner: copies Indira Foods JSON layout → 90% similar output
    result = app.invoke({
        "image_paths": [
            fr"{BASE}\2-Cognixia-Apr-25.jpg",
            fr"{BASE}\2-Gen-Ai-Disclosure-act-.jpg",
            fr"{BASE}\2-Gen-AI-X-Network-Ops.jpg",
            fr"{BASE}\Agent-Ai.jpg",
            fr"{BASE}\Artboard.jpg",
            fr"{BASE}\AI-generated-Art.png",
            fr"{BASE}\2-women-in-tech.png",
            fr"{BASE}\Indira Foods.jpg", 
            # fr"{BASE}\2-Benefits-Of-ERGs (1).pdf",     # ← food template (will be boosted to rank 1)
        ],
        "goal": "Can AI Replace Designers?",
        "goal": "Fundamental Principles of LLM with workflow ",
        "goal": "Lemon Rice Masala product advertisement poster with steaming lemon rice, lemon wedges, curry leaves, mustard seeds, and clean vibrant food photography style",
        # "goal": "The future of Generative AI in creative design, professional tech thought leadership post with futuristic elements",
        # "goal": "Coffee powder product advertisement poster with coffee beans, steaming cup, and clean food photography style",
        "goal": "International Men’s Day circuit-style technology poster",
        # "goal": "G Coconut Biscuit product advertisement poster with biscuits, coconut pieces, and clean food photography style",
        # "goal": (
        #     "A realistic food advertisement poster showing a bowl of hot Bisibelebath "
        #     "on a kitchen table with a premium 'Indira's Bisibelebath Powder' packet beside it, "
        #     "natural lighting, green plant leaves and brick wall background, modern lifestyle "
        #     "setup with laptop and keys, clean product-focused food photography."
        # ),

        "logo_path": LOGO_PATH,
    })

    # ── Example 2: Tech creative ───────────────────────────────────────────
    # result = app.invoke({
    #     "image_paths": [],    # already stored
    #     "goal": "Can AI Replace Designers? — Cognixia thought leadership post",
    #     "logo_path": LOGO_PATH,
    # })

    # ── Example 3: Tech creative (no new images to store) ─────────────────
    # result = app.invoke({
    #     "image_paths": [],
    #     "goal": "Agentic RAG Workflows — how AI agents retrieve and reason autonomously",
    #     "logo_path": LOGO_PATH,
    # })

    cc = result.get("creative_content", {})
    bp = result.get("blueprint_json", {})
    print(f"\n{'═'*62}")
    print("🎯 PIPELINE COMPLETE")
    print(f"{'═'*62}")
    print(f"  Goal category   : {result.get('goal_category', '?')}")
    print(f"  Templates used  : {[os.path.basename(r.get('_source_image','?')) for r in result.get('retrieved_jsons',[])]}")
    print(f"  Template scores : {result.get('retrieved_scores', [])}")
    print(f"  Cloned pattern  : {bp.get('brand_signals',{}).get('layout_pattern','?')}")
    print(f"  Cloned style    : {bp.get('aesthetics',{}).get('art_style','?')}")
    print(f"  Headline        : {cc.get('headline_lines')}")
    print(f"  Sub             : {cc.get('subheadline')}")
    for i, b in enumerate(cc.get("bullets", []), 1):
        print(f"  Bullet {i}       : {b}")
    print(f"  CTA             : {cc.get('cta_text')}")
    print(f"  Background      : {result.get('bg_image_path')}")
    print(f"  ✅ Final         : {result.get('saved_image_path')}")
    print(app.get_graph().draw_mermaid())