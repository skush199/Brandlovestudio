
import os
import json
import base64
import io
from datetime import datetime
from typing_extensions import TypedDict
from typing import Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from langgraph.graph import StateGraph, START, END

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# Graph State
# -----------------------------
class GraphState(TypedDict, total=False):
    image_path: str
    image_analysis: Dict[str, Any]
    analysis_json_path: str

    dalle_prompt: str
    saved_image_path: str

# -----------------------------
# Helper: compress + base64 data URL
# -----------------------------
def image_to_data_url(image_path: str, max_side: int = 1200, quality: int = 85) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:
        img = img.resize((int(w * scale), int(h * scale)))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# -----------------------------
# Node 1: OpenAI Analyze + Save JSON
# -----------------------------
def openai_analyze_and_save_node(state: GraphState) -> GraphState:
    print("👁️ Running openai_analyze_and_save_node...")

    image_path = state.get("image_path")
    if not image_path:
        raise ValueError("state['image_path'] is required")

    data_url = image_to_data_url(image_path)

    prompt = """
Analyze the image and return ONLY valid JSON with this structure:

{
  "core_content": {"main_description": "..."},
  "subjects": [
    {
      "type": "...",
      "identity": "...",
      "spatial_position": {
        "placement": "...",
        "scale": "...",
        "dynamic_framing": {"visual_vector": "...", "frame_interaction": "..."}
      },
      "pose": {"body_narrative": "...", "hand_gesture": "..."},
      "eye_gaze": {"direction": "...", "contact": "...", "focus": "..."},
      "appearance": {"features": "...", "grooming": "..."},
      "attire": "..."
    }
  ],
  "environment": {
    "scene": "...",
    "background_elements": [{"object": "...", "description": "..."}]
  },
  "aesthetics": {
    "art_style": "...",
    "lighting": {"key_light": "...", "shadows": "..."},
    "camera": {"angle": "...", "lens": "..."}
  },
  "typography": {
    "layout": "...",
    "elements": [{"text": "...", "style": "..."}]
  }
}

Rules:
- Use what is visible in the image.
- If no text is readable: typography.layout="MISSING" and typography.elements=[]
- If uncertain: use "N/A"
- Output JSON only. No markdown. No extra text.
""".strip()

    resp = client.responses.create(
        model="gpt-4.1",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
    )

    raw = (resp.output_text or "").strip()

    # Parse JSON safely (fallback stores raw if model returns non-JSON)
    try:
        analysis = json.loads(raw)
    except Exception:
        analysis = {"raw": raw}

    # Save JSON to file
    os.makedirs("json_images", exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join("json_images", f"{base}_analysis.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved analysis JSON: {out_path}")

    return {**state, "image_analysis": analysis, "analysis_json_path": out_path}

# -----------------------------
# Node 2: JSON -> DALL·E prompt -> Generate image
# -----------------------------
def dalle_generate_from_json_node(state: GraphState) -> GraphState:
    print("🖼️ Running dalle_generate_from_json_node...")

    analysis = state.get("image_analysis")
    if not analysis:
        # allow reading from saved json if analysis not present
        json_path = state.get("analysis_json_path")
        if not json_path or not os.path.exists(json_path):
            raise ValueError("No image_analysis in state and analysis_json_path missing.")
        with open(json_path, "r", encoding="utf-8") as f:
            analysis = json.load(f)

    # Build a strong generation prompt from analysis (use GPT-4.1 text)
    prompt_builder = f"""
You are a senior DALL·E prompt engineer.

Given this image analysis JSON, write ONE production-grade DALL·E prompt to recreate a NEW image
that matches the same scene composition and design language, but not a pixel-perfect copy.

ANALYSIS JSON:
{json.dumps(analysis, ensure_ascii=False)}

Rules:
- Specify background, main subjects, lighting, camera angle, and typography layout (if present).
- Keep it concise but specific (1 paragraph).
- Output ONLY the final DALL·E prompt text (no quotes, no bullets).
""".strip()

    prompt_resp = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You output only a single final DALL·E prompt paragraph."},
            {"role": "user", "content": prompt_builder},
        ],
    )

    dalle_prompt = prompt_resp.choices[0].message.content.strip()
    state["dalle_prompt"] = dalle_prompt

    # Generate image (DALL·E 3)
    img_resp = client.images.generate(
        model="gpt-image-1",
        prompt=dalle_prompt,
        size="1536x1024",
        quality="high",
        # style="vivid",  # optional for dalle-3
    )

    data0 = img_resp.data[0]
    b64 = getattr(data0, "b64_json", None)
    url = getattr(data0, "url", None)

    os.makedirs("json_images", exist_ok=True)
    base = os.path.splitext(os.path.basename(state.get("image_path", "image")))[0]  
    out_path = os.path.join(
    "json_images",
    f"{base}_dalle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
)

    if b64:
        img_bytes = base64.b64decode(b64)
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        print(f"✅ DALL·E image saved at: {out_path}")
        return {**state, "saved_image_path": out_path}

    if url:
        # Some environments return URLs instead of base64. We'll store the URL.
        print(f"✅ DALL·E image URL: {url}")
        return {**state, "saved_image_path": url}

    raise ValueError(f"DALL·E generation failed: no b64_json or url returned. data0={data0}")

# -----------------------------
# Build graph: analyze -> dalle_generate
# -----------------------------
workflow = StateGraph(GraphState)
workflow.add_node("openai_analyze_save", openai_analyze_and_save_node)
workflow.add_node("dalle_generate_from_json", dalle_generate_from_json_node)

workflow.add_edge(START, "openai_analyze_save")
workflow.add_edge("openai_analyze_save", "dalle_generate_from_json")
workflow.add_edge("dalle_generate_from_json", END)

app = workflow.compile()

# -----------------------------
# Run: result = app.invoke(...)
# -----------------------------
if __name__ == "__main__":
    result = app.invoke({"image_path": "/Users/vishruth/Downloads/2-women-in-tech.png"})

    print("\n🎯 Analysis JSON saved at:", result.get("analysis_json_path"))
    print("🎯 DALL·E prompt:\n", result.get("dalle_prompt"))
    print("🎯 Generated image saved at:", result.get("saved_image_path"))