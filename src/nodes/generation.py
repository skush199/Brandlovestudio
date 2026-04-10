from __future__ import annotations

import os
import re
import json
import base64
from datetime import datetime

import requests
from openai import OpenAI

from nodes.common import GraphState, logger

def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", (text or "").strip())


def _truncate_text(text: str, max_chars: int) -> str:
    text = _normalize_whitespace(text)
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_break = max(truncated.rfind("\n"), truncated.rfind(". "), truncated.rfind("; "))
    if last_break > max_chars * 0.6:
        truncated = truncated[: last_break + 1]
    return truncated.rstrip() + "\n...[truncated]"


def _extract_section(prompt_text: str, section_name: str) -> str:
    pattern = rf"\[{re.escape(section_name)}\]\s*(.*?)(?=\n\[[^\n\]]+\]|\Z)"
    match = re.search(pattern, prompt_text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def _build_compact_image_prompt(
    base_prompt: str,
    goal: str,
    blog_summary: str,
    text_style: str,
    max_length: int = 28000,
) -> str:
    main_message = _truncate_text(_extract_section(base_prompt, "MAIN MESSAGE"), 3500)
    visual_rules = _truncate_text(_extract_section(base_prompt, "VISUAL RULES"), 5000)
    communication_strategy = _truncate_text(
        _extract_section(base_prompt, "COMMUNICATION STRATEGY"), 3500
    )
    brand_guardrails = _truncate_text(
        _extract_section(base_prompt, "BRAND GUARDRAILS"), 3000
    )
    visual_refs = _truncate_text(
        _extract_section(base_prompt, "CREATIVE VISUAL CONTENT DESCRIPTIONS - STRICT REFERENCE"),
        5000,
    )
    layout_rules = _truncate_text(
        _extract_section(base_prompt, "LAYOUT ENFORCEMENT - MANDATORY"), 1500
    )
    final_rules = _truncate_text(
        _extract_section(base_prompt, "FINAL EXECUTION RULES"), 1000
    )

    compact_parts = [
        "Create a photorealistic branded social post image.",
    ]

    if goal:
        compact_parts.append(f"[GOAL]\n{goal.strip()}")

    if main_message:
        compact_parts.append(f"[MAIN MESSAGE]\n{main_message}")

    if communication_strategy:
        compact_parts.append(f"[COMMUNICATION STRATEGY]\n{communication_strategy}")

    if visual_rules:
        compact_parts.append(f"[VISUAL RULES]\n{visual_rules}")

    if brand_guardrails:
        compact_parts.append(f"[BRAND GUARDRAILS]\n{brand_guardrails}")

    if visual_refs:
        compact_parts.append(f"[VISUAL REFERENCES]\n{visual_refs}")

    if text_style:
        compact_parts.append(
            "[TYPOGRAPHY]\n"
            f"Use this exact typography style consistently for all visible text: {text_style}"
        )

    if blog_summary:
        compact_parts.append(
            "[TEXT TO DISPLAY]\n"
            "Use this as the visible main caption/body copy if space allows:\n"
            f"{blog_summary.strip()}"
        )

    if layout_rules:
        compact_parts.append(f"[LAYOUT RULES]\n{layout_rules}")

    if final_rules:
        compact_parts.append(f"[FINAL RULES]\n{final_rules}")

    compact_parts.append(
        "[NON-NEGOTIABLE]\n"
        "- Keep all text inside safe margins.\n"
        "- Do not crop headline, CTA, footer, or logo.\n"
        "- Do not use cartoon, anime, vector, flat illustration, or 3D cartoon style.\n"
        "- Include only visual elements supported by the reference creatives.\n"
        "- Keep composition clean, readable, and premium."
    )

    prompt = "\n\n".join(part for part in compact_parts if part).strip()

    if len(prompt) > max_length:
        prompt = prompt[: max_length - 20].rstrip() + "\n...[truncated]"

    return prompt


def get_model_output_path(
    model_name: str, prefix: str = "image", ext: str = "png"
) -> str:
    safe_model_name = re.sub(r"[^a-zA-Z0-9._-]", "_", model_name.strip())
    model_dir = os.path.join("samples_4", safe_model_name)
    os.makedirs(model_dir, exist_ok=True)

    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
    return os.path.join(model_dir, filename)

def generate_blog_node(state: GraphState) -> GraphState:
    print("📝 Running Generate Blog Node...")
    logger.log_workflow("generate_blog_node")
    logger.log_api_call("OpenAI Chat", "Generating blog from goal + DB answers")

    goal = (state.get("goal") or "").strip()
    db_answers = state.get("db_answers") or {}
    merged_prompt = (state.get("prompt") or "").strip()

    if not goal:
        raise ValueError("Missing 'goal' in state for blog generation.")

    if not db_answers:
        print("  ⚠️ No db_answers in state, loading from chat_node output...")
        chat_output_path = "db_answers.json"
        if os.path.exists(chat_output_path):
            try:
                with open(chat_output_path, "r", encoding="utf-8") as f:
                    db_answers = json.load(f)
                print(f"  ✅ Loaded db_answers from {chat_output_path}")
            except Exception as e:
                print(f"  ⚠️ Error loading db_answers: {e}")

    answers_text = "\n\n".join(
        [
            f"{db_name.upper()} ANSWER:\n{answer}"
            for db_name, answer in db_answers.items()
            if answer and str(answer).strip()
        ]
    )

    system_msg = """
    You are a senior content strategist.

    Write:
    1. a concise, brand-aligned blog
    2. a compact blog summary

    Rules:
    - Use only the campaign goal, DB answers, and merged prompt guidance
    - Do not use external knowledge
    - Do not invent unsupported claims
    - The blog summary must be under 100 words
    - The blog summary must contain only the most important words, phrases, and ideas from the blog
    - Keep the summary dense, clear, and useful for future reuse
    """

    user_msg = f"""
    GOAL:
    {goal}

    ANSWERS FROM ALL DBS:
    {answers_text}

    MERGED PROMPT GUIDANCE:
    {merged_prompt}

    Return EXACTLY this format. Do NOT repeat sections:

    BLOG:
    <content>

    BLOG SUMMARY:
    - EXACTLY 2 lines
    - Line 1: catchy, engaging hook sentence (human-friendly)
    - Line 2: keyword-dense summary (important words, phrases for reuse)
    - No repetition
    - No bullet points
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    full_output = response.choices[0].message.content.strip()

    blog_text = full_output
    blog_summary_text = ""

    if "BLOG SUMMARY:" in full_output:
        parts = full_output.split("BLOG SUMMARY:", 1)
        blog_text = parts[0].strip()
        blog_summary_text = parts[1].strip()

        words = blog_summary_text.split()
        line1 = " ".join(words[:15])
        line2 = " ".join(words[15:30])

        blog_summary_text = f"{line1}\n{line2}"

    if blog_text.startswith("BLOG:"):
        blog_text = blog_text[len("BLOG:") :].strip()

    print("\n📝 GENERATED BLOG:\n")
    print(blog_text)

    print("\n🧾 BLOG SUMMARY:\n")
    print(blog_summary_text)

    return {"blog_text": blog_text, "blog_summary": blog_summary_text}

def generate_image(state: GraphState) -> GraphState:
    print("🖼 Running Generate Image Node...")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    base_prompt = (state.get("prompt") or "").strip()
    if not base_prompt:
        raise ValueError(
            "Missing 'prompt' in state. generate_prompt must run before generate_image."
        )

    brand_data = state.get("brand_data", {})
    text_style = ""

    if brand_data:
        typography = brand_data.get("Typography", {})
        if typography:
            text_style = typography.get("Text_style", "")

    if not text_style:
        jiraaf_path = "Jiraaf_data.json"
        if os.path.exists(jiraaf_path):
            try:
                with open(jiraaf_path, "r", encoding="utf-8") as f:
                    jiraaf_data = json.load(f)
                typography = jiraaf_data.get("Typography", {})
                if typography:
                    text_style = typography.get("Text_style", "")
            except Exception as e:
                print(f"  ⚠️ Error reading Jiraaf_data.json for typography: {e}")

    goal = state.get("goal", "")
    blog_summary = (state.get("blog_summary") or "").strip()

    prompt = _build_compact_image_prompt(
        base_prompt=base_prompt,
        goal=goal,
        blog_summary=blog_summary,
        text_style=text_style,
        max_length=28000,
    )

    print(f"  🧾 Final image prompt length: {len(prompt)}")

    model_name = state.get("image_model") or "gpt-image-1-mini"

    response = client.images.generate(
        model=model_name, prompt=prompt, size="1024x1024", n=1
    )

    data0 = response.data[0]
    filename = get_model_output_path(model_name=model_name, prefix="image", ext="png")

    b64 = getattr(data0, "b64_json", None)
    if b64:
        image_bytes = base64.b64decode(b64)
        with open(filename, "wb") as f:
            f.write(image_bytes)

        print(f"✅ Image saved at: {filename}")
        state["saved_image_path"] = filename
        state["image_model"] = model_name
        return state

    url = getattr(data0, "url", None)
    if url:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(filename, "wb") as f:
            f.write(r.content)

        print(f"✅ Image downloaded & saved at: {filename}")
        state["saved_image_path"] = filename
        state["image_model"] = model_name
        return state

    raise ValueError(
        f"Image generation failed. No b64_json or url returned. Raw: {data0}"
    )

def image_feedback(state: GraphState) -> GraphState:
    print("\n🖼 Generated Image:", state.get("saved_image_path", "N/A"))
    print("Are you satisfied with this image?")
    print("Type 'y' for yes OR give feedback to modify it.\n")

    user_input = input("Your response: ")
    state["user_feedback"] = user_input
    return state

def edit_image(state: GraphState) -> GraphState:
    print("✏️ Running Image Correction Node...")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    existing_image_path = state.get("saved_image_path")
    feedback = state.get("user_feedback")

    if not existing_image_path:
        raise ValueError("No existing image found to edit.")

    if not feedback:
        raise ValueError("No feedback provided for editing.")
    edit_prompt = f"""
    Modify the EXISTING image.
    STRICT RULES:
    - Keep entire layout unchanged
    - Do NOT redesign
    - Do NOT move elements
    - Do NOT change colors
    - Do NOT remove any existing elements unless explicitly instructed.
    - Do NOT replace any existing text unless explicitly instructed.
    - If new text is requested, ADD it into the image without disturbing any existing content.
    - Place new text in an appropriate empty space while maintaining the same style.
    - Preserve typography, spacing, and visual hierarchy.

    USER REQUEST:

    {feedback}

    Apply ONLY the requested modification.
    Everything else must remain exactly the same.
    """
    # response = client.images.edit(

    #     model="gpt-image-1",
    #     image=open(existing_image_path, "rb"),
    #     prompt=edit_prompt,
    #     size="1024x1024",
    #     quality="low"

    # )
    model_name = state.get("image_model") or "gpt-image-1-mini"

    response = client.images.edit(
        model=model_name,
        image=open(existing_image_path, "rb"),
        prompt=edit_prompt,
        size="1024x1024",
        # quality="high",
    )

    item = response.data[0]
    image_base64 = (
        getattr(item, "b64_json", None)
        if hasattr(item, "b64_json")
        else item.get("b64_json")
    )

    if image_base64:
        image_bytes = base64.b64decode(image_base64)

    else:
        image_url = (
            getattr(item, "url", None) if hasattr(item, "url") else item.get("url")
        )
        r = requests.get(image_url, timeout=60)
        r.raise_for_status()
        image_bytes = r.content

    # filename = f"generated_images/edited_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    filename = get_model_output_path(model_name=model_name, prefix="edited", ext="png")

    with open(filename, "wb") as f:
        f.write(image_bytes)

    print(f"✅ Edited Image saved at: {filename}")

    state["saved_image_path"] = filename
    return state
