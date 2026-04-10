from __future__ import annotations

import os
import json

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

from nodes.common import image_desc_store

def _flatten(results) -> str:
    """Flatten a list of retrieved doc dicts into a plain string."""
    if isinstance(results, list):
        return "\n".join(
            r.get("content", "") if isinstance(r, dict) else str(r) for r in results
        )
    return str(results)

def _b(brand: dict, key: str) -> str:
    """Safe brand value lookup — handles list values from JSON and returns {key} when absent."""
    value = brand.get(key)
    
    # Handle list values (JSON stores them as arrays like ["Jiraaf"])
    if isinstance(value, list):
        if value:
            return str(value[0]).strip()
        return f"{{{key}}}"
    
    # Handle None or empty string
    if value is None or value == "":
        return f"{{{key}}}"
    
    return str(value).strip()

def _call_openai(system_msg: str, user_content: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content.strip()

def generate_prompt_main(state) -> dict:
    """
    Generates a DALL-E prompt grounded in MAIN CONTENT documents only.
    Result stored in state["prompt_main"].
    """
    print("🎨 Running Generate Prompt (MAIN) Node...")

    SYSTEM_GOAL = """

    You are a senior brand content strategist for {Brand_Name}.
    Your responsibility is to retrieve, interpret, and synthesize the core content and messaging direction for a branded marketing poster using ONLY the retrieved main-content knowledge and the brand context provided by the user.
    You must follow these non-negotiable rules at all times:

    1. Retrieve and synthesize ONLY the message and content direction aligned with the provided brand context.
    2. Every output field MUST reflect the emotion: {Primary_Emotion} in tone and message framing.
    3. NEVER include wording, themes, or claims that trigger the emotion: {Avoided_Emotion}.
    4. All retrieved content MUST serve the goal: {goal} and align with the mission: {Brand_Mission}.
    5. Apply {What_To_Do} as behavioral guardrails at all times.
    6. Apply {What_Not_To_Do} as hard restrictions with no exceptions.
    7. Focus only on communication, message, and content direction.
    8. Do not generate visual design guidance.
    9. Do not generate colors, layout, or typography instructions.
    10. Do not invent unsupported claims.
    11. Do not generate the final image prompt.
    12. If any field is unavailable, return exactly: "MISSING".
    
    Your output must strictly follow this exact structure:
    
    1. Primary Campaign Theme
    2. Core Audience Message
    3. Headline Direction
    4. Supporting Copy Direction
    5. CTA Intent
    6. Key Value Proposition
    7. Important Keywords/Phrases
    8. Emotional Messaging Direction
    9. What Must Be Avoided In Messaging

    """

    USER_GOAL = """

    ## BRAND CONTEXT
    {Brand_Name} is a brand operating with the mission of {Brand_Mission} and a long-term vision of {Brand_Vision}. The brand promises its customers {Brand_Promise} and is positioned in the market as {Market_Positioning}, standing apart through its key differentiators: {Key_Differentiators}. The brand primarily serves a {Audience_Type} audience, with the persona playing the role of {Persona_Role}, driven by goals such as {Persona_Goals} and facing pain points including {Fear_And_Pain_Points}. The current strategic goal guiding this retrieval is {goal}. In all communications, the brand must lead with the emotion of {Primary_Emotion} and must strictly avoid evoking {Avoided_Emotion}. The brand always follows these behavioral principles — {What_To_Do} — and must never engage in the following — {What_Not_To_Do}.
    
    ## RETRIEVED MAIN CONTENT
    {retrieved_main}
    
    ## TASK
    Using only the retrieved main content and the brand context above, extract and synthesize the poster messaging direction.

    """
    brand = state.get("brand_data", {})
    goal = state.get("goal", "")

    retrieved_main = _flatten(state.get("retrieved_docs", {}).get("main", []))

    system_msg = SYSTEM_GOAL.format(
        Brand_Name=_b(brand, "Brand_Name"),
        Brand_Mission=_b(brand, "Brand_Mission"),
        Brand_Vision=_b(brand, "Brand_Vision"),
        Brand_Promise=_b(brand, "Brand_Promise"),
        Market_Positioning=_b(brand, "Market_Positioning"),
        Key_Differentiators=_b(brand, "Key_Differentiators"),
        Audience_Type=_b(brand, "Audience_Type"),
        Persona_Role=_b(brand, "Persona_Role"),
        Persona_Goals=_b(brand, "Persona_Goals"),
        Fear_And_Pain_Points=_b(brand, "Fear_And_Pain_Points"),
        Primary_Emotion=_b(brand, "Primary_Emotion"),
        Avoided_Emotion=_b(brand, "Avoided_Emotion"),
        What_To_Do=_b(brand, "What_To_Do"),
        What_Not_To_Do=_b(brand, "What_Not_To_Do"),
        goal=goal,
    )

    user_content = USER_GOAL.format(
        retrieved_main=retrieved_main,
        Brand_Name=_b(brand, "Brand_Name"),
        Brand_Mission=_b(brand, "Brand_Mission"),
        Brand_Vision=_b(brand, "Brand_Vision"),
        Brand_Promise=_b(brand, "Brand_Promise"),
        Market_Positioning=_b(brand, "Market_Positioning"),
        Key_Differentiators=_b(brand, "Key_Differentiators"),
        Audience_Type=_b(brand, "Audience_Type"),
        Persona_Role=_b(brand, "Persona_Role"),
        Persona_Goals=_b(brand, "Persona_Goals"),
        Fear_And_Pain_Points=_b(brand, "Fear_And_Pain_Points"),
        Primary_Emotion=_b(brand, "Primary_Emotion"),
        Avoided_Emotion=_b(brand, "Avoided_Emotion"),
        What_To_Do=_b(brand, "What_To_Do"),
        What_Not_To_Do=_b(brand, "What_Not_To_Do"),
        goal=goal,
    )

    print("\n" + "=" * 60)
    print("PROMPT MAIN - SYSTEM_GOAL:")
    print("=" * 60)
    print(system_msg)
    print("\n" + "=" * 60)
    print("PROMPT MAIN - USER_GOAL:")
    print("=" * 60)
    print(user_content)

    prompt = _call_openai(system_msg, user_content)

    print(f"===============main================================{prompt}")

    # Return ONLY the key this node writes — avoids concurrent-write conflict in fan-out
    return {"prompt_main": prompt}

def generate_prompt_metadata(state) -> dict:
    """
    Generates visual execution rules grounded in METADATA / CREATIVE SAMPLE documents only.
    Result stored in state["prompt_metadata"].
    Uses gpt-4.1-mini and extracts ONLY visible text-color and layout patterns.
    """
    print("🎨 Running Generate Prompt (METADATA) Node...")

    SYSTEM_GOAL = """
You are the VISUAL LAYOUT + EXECUTION RULES agent.
Model: gpt-4.1-mini

Your job is to study the retrieved creative samples and extract the visual system they follow.

You must infer and preserve:
- where headline usually appears
- where body text usually appears
- where CTA usually appears
- where footer usually appears
- where logo usually appears
- where the main subject/person usually appears
- how empty space is reserved for text
- how text avoids faces/subjects
- how the layout flows from top to bottom / left to right
- what side is text-heavy vs image-heavy
- how sections are stacked

Non-negotiable rules:
1. Use ONLY the user-provided sample creatives and retrieved metadata context.
2. Do NOT invent layout patterns that are not supported by repeated visual evidence.
3. Preserve role-based text color hierarchy from the samples.
4. Preserve role-based layout hierarchy from the samples.
5. Focus on relative layout zones, not exact pixels.
6. Preserve the same reading flow and same visual balance as the samples.
7. Preserve the same side dominance if evident (for example: text-left / subject-right).
8. Preserve safe empty areas for text if the samples imply them.
9. Do NOT output explanations.
10. If unclear, return MISSING rather than guessing.

Return exactly in this structure:

[TEXT COLOR SYSTEM]
Headline Text Color:
Body Text Color:
Highlight / Emphasis Text Color:
Large Numeral Text Color:
CTA Text Color:
CTA Background / Accent Color:
Footer Text Color:
Footer Background Color:
Divider / Accent Line Color:

[POSITIONAL LAYOUT MAP]
Headline Position:
Subheadline Position:
Body Text Position:
CTA Position:
Footer Position:
Logo Position:
Primary Subject Position:
Secondary Subject Position:
Icon / Illustration Position:
Empty / Safe Text Zones:
Text Alignment Pattern:
Reading Flow Pattern:
Section Stacking Pattern:
Text-heavy Side:
Image-heavy Side:

[CREATIVE CONTENT DESCRIPTIONS - CRITICAL]
Based on the image descriptions provided, extract:
1. What visual content is shown in each creative (people, objects, settings, scenes)
2. What type of content each creative represents (lifestyle shot, product showcase, infographic, testimonial, brand story, etc.)
3. What message or meaning each creative is trying to convey
4. What graphs, charts, icons, or visual elements are present and what purpose they serve
5. Visual style (photograph, illustration, minimalist, bold, elegant, etc.)
6. Mood and tone of each creative
7. How the visual content aligns with the brand message

[LAYOUT SUPPORT]
Text Placement Pattern:
CTA Placement Pattern:
Footer Placement Pattern:
Typography Hierarchy:
Contrast Pattern:
Face/Subject Avoidance Pattern:
Spacing / Margin Pattern:
Grid / Section Structure:
Balance Pattern:

[IMAGE COUNT]
Number of Subjects/People:
Subject Positioning Pattern:

[IMAGE LAYOUT]
Layout Pattern:
Subject Placement:
Background Style:

[ENFORCEMENT RULES]
- Follow the sample text-color hierarchy exactly where supported.
- Follow the sample positional layout map exactly where supported.
- Keep headline, body, CTA, footer, logo, and subject in the same relative zones as the samples.
- Do not move CTA/footer/logo to a new side unless the samples are unclear.
- Do not place text over faces or key subjects.
- Preserve empty space reserved for text.
- Preserve the same reading flow and section stacking pattern.
- CRITICAL: Include similar types of visual content, graphs, charts, icons, or visual elements as shown in the sample creatives
- If the samples contain infographics, include appropriate data visualizations
- If the samples show lifestyle imagery, include relevant lifestyle elements
- Match the overall visual style (photograph vs illustration, minimalist vs detailed) of the samples
"""

    USER_GOAL = """
## BRAND CONTEXT
{Brand_Name} is a brand operating with the mission of {Brand_Mission} and a long-term vision of {Brand_Vision}. The brand promises its customers {Brand_Promise} and is positioned in the market as {Market_Positioning}, standing apart through its key differentiators: {Key_Differentiators}. The brand primarily serves a {Audience_Type} audience.

## RETRIEVED SAMPLE CREATIVES (METADATA)
{retrieved_metadata}

## TASK
From the provided sample creatives, extract ONLY visibly supported visual execution and layout rules.

Return exactly:
1) Headline text color
2) Body text color
3) Highlight/emphasis text color
4) Large numeral text color
5) CTA text color
6) CTA background/accent color
7) Footer text color
8) Footer background color
9) Divider/accent line color

10) Headline position
11) Subheadline position
12) Body text position
13) CTA position
14) Footer position
15) Logo position
16) Primary subject position
17) Secondary subject position
18) Icon / illustration position
19) Empty / safe text zones
20) Text alignment pattern
21) Reading flow pattern
22) Section stacking pattern
23) Text-heavy side
24) Image-heavy side

25) Text placement pattern
26) CTA placement pattern
27) Footer placement pattern
28) Typography hierarchy
29) Contrast pattern
30) Face/Subject avoidance pattern
31) Spacing / margin pattern
32) Grid / section structure
33) Balance pattern

34) Number of subjects/people
35) Subject positioning pattern
36) Layout pattern
37) Subject placement
38) Background style

39) Creative Content Descriptions (from image analysis)
40) Content Type & Meaning for each creative
41) Graphs/Charts/Visual Elements present
42) Visual Style Summary
43) Mood and Tone

Rules:
- Use only what is visible in the samples
- Do not guess
- If unclear, return MISSING
- Prioritize repeated patterns across multiple samples
- Preserve relative positioning, not exact pixels
- Infer layout only from repeated visible evidence
"""
    brand = state.get("brand_data", {})
    goal = state.get("goal", "")

    retrieved_metadata = _flatten(state.get("retrieved_docs_metadata", []))

    system_msg = SYSTEM_GOAL.format(
        Brand_Name=_b(brand, "Brand_Name"),
        Brand_Mission=_b(brand, "Brand_Mission"),
        Brand_Vision=_b(brand, "Brand_Vision"),
        Brand_Promise=_b(brand, "Brand_Promise"),
        Market_Positioning=_b(brand, "Market_Positioning"),
        Key_Differentiators=_b(brand, "Key_Differentiators"),
        Audience_Type=_b(brand, "Audience_Type"),
        Persona_Role=_b(brand, "Persona_Role"),
        Persona_Goals=_b(brand, "Persona_Goals"),
        Fear_And_Pain_Points=_b(brand, "Fear_And_Pain_Points"),
        Primary_Emotion=_b(brand, "Primary_Emotion"),
        Avoided_Emotion=_b(brand, "Avoided_Emotion"),
        What_To_Do=_b(brand, "What_To_Do"),
        What_Not_To_Do=_b(brand, "What_Not_To_Do"),
        goal=goal,
    )

    user_content = USER_GOAL.format(
        retrieved_metadata=retrieved_metadata,
        Brand_Name=_b(brand, "Brand_Name"),
        Brand_Mission=_b(brand, "Brand_Mission"),
        Brand_Vision=_b(brand, "Brand_Vision"),
        Brand_Promise=_b(brand, "Brand_Promise"),
        Market_Positioning=_b(brand, "Market_Positioning"),
        Key_Differentiators=_b(brand, "Key_Differentiators"),
        Audience_Type=_b(brand, "Audience_Type"),
        Persona_Role=_b(brand, "Persona_Role"),
        Persona_Goals=_b(brand, "Persona_Goals"),
        Fear_And_Pain_Points=_b(brand, "Fear_And_Pain_Points"),
        Primary_Emotion=_b(brand, "Primary_Emotion"),
        Avoided_Emotion=_b(brand, "Avoided_Emotion"),
        What_To_Do=_b(brand, "What_To_Do"),
        What_Not_To_Do=_b(brand, "What_Not_To_Do"),
        goal=goal,
    )

    print("\n" + "=" * 60)
    print("PROMPT METADATA - SYSTEM_GOAL:")
    print("=" * 60)
    print(system_msg)
    print("\n" + "=" * 60)
    print("PROMPT METADATA - USER_GOAL:")
    print("=" * 60)
    print(user_content)

    prompt = _call_openai_visual(system_msg, user_content)

    print(f"====================Metadata prompt========================{prompt}")

    # Return ONLY the key this node writes — avoids concurrent-write conflict in fan-out
    return {"prompt_metadata": prompt}

def _call_openai_visual(system_msg: str, user_content: str) -> str:
    """Visual execution rules agent - uses gpt-4.1-mini for strict text-color extraction."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content.strip()

def generate_prompt_strategy(state) -> dict:
    """
    Generates a DALL-E prompt grounded in STRATEGY documents only.
    Result stored in state["prompt_strategy"].
    """
    print("🎨 Running Generate Prompt (STRATEGY) Node...")

    SYSTEM_GOAL = """
    You are a senior brand strategist for {Brand_Name}.

    Your responsibility is to retrieve, interpret, and synthesize a comprehensive communication strategy for a branded marketing poster using ONLY the retrieved strategy knowledge and the brand context provided by the user.

    You must follow these non-negotiable rules at all times:

    1. Retrieve ALL strategy elements strictly aligned with the brand context above.
    2. Every output field MUST reflect the emotion: {Primary_Emotion} in strategic framing and communication direction.
    3. NEVER include content that triggers the emotion: {Avoided_Emotion}.
    4. All retrieved content MUST serve the goal: {goal} and align with the mission: {Brand_Mission}.
    5. Apply {What_To_Do} as behavioral and communication guardrails.
    6. Apply {What_Not_To_Do} as hard strategic restrictions.
    7. Focus only on communication strategy, positioning, audience insight, persuasion, and CTA direction.
    8. Do not generate design, layout, or color guidance.
    9. Do not generate the final image prompt.
    10. Do not invent unsupported strategy.
    11. If any field is unavailable, return exactly: "MISSING".

    Your output must strictly follow this exact structure:

    1. Target Audience Interpretation
    2. Persona Motivation
    3. Persona Pain Points
    4. Persona Goals
    5. Strategic Communication Angle
    6. Core Positioning Angle
    7. Key Differentiators To Highlight
    8. Core Value Proposition
    9. Emotional Persuasion Direction
    10. CTA Strategy
    11. Trust / Aspiration / Urgency / Education Balance
    12. What Strategic Themes Must Be Emphasized
    13. What Strategic Themes Must Be Avoided
    """

    USER_GOAL = """
    ## BRAND CONTEXT

    {Brand_Name} is a brand operating with the mission of {Brand_Mission} and a long-term vision of {Brand_Vision}. The brand promises its customers {Brand_Promise} and is positioned in the market as {Market_Positioning}, standing apart through its key differentiators: {Key_Differentiators}. The brand primarily serves a {Audience_Type} audience, with the persona playing the role of {Persona_Role}, driven by goals such as {Persona_Goals} and facing pain points including {Fear_And_Pain_Points}. The current strategic goal guiding this retrieval is {goal}. In all communications, the brand must lead with the emotion of {Primary_Emotion} and must strictly avoid evoking {Avoided_Emotion}. The brand always follows these behavioral principles — {What_To_Do} — and must never engage in the following — {What_Not_To_Do}.

    ## RETRIEVED STRATEGY CONTEXT

    {retrieved_strategy}

    ## TASK

    Using only the retrieved strategy context and the brand context above, extract and synthesize the communication strategy for the poster.
    """
    # OUTPUT_FORMAT variable removed — structure is already specified in SYSTEM_GOAL above.

    brand = state.get("brand_data", {})
    goal = state.get("goal", "")

    retrieved_strategy = _flatten(state.get("retrieved_docs_strategy", []))

    system_msg = SYSTEM_GOAL.format(
        Brand_Name=_b(brand, "Brand_Name"),
        Brand_Mission=_b(brand, "Brand_Mission"),
        Brand_Vision=_b(brand, "Brand_Vision"),
        Brand_Promise=_b(brand, "Brand_Promise"),
        Market_Positioning=_b(brand, "Market_Positioning"),
        Key_Differentiators=_b(brand, "Key_Differentiators"),
        Audience_Type=_b(brand, "Audience_Type"),
        Persona_Role=_b(brand, "Persona_Role"),
        Persona_Goals=_b(brand, "Persona_Goals"),
        Fear_And_Pain_Points=_b(brand, "Fear_And_Pain_Points"),
        Primary_Emotion=_b(brand, "Primary_Emotion"),
        Avoided_Emotion=_b(brand, "Avoided_Emotion"),
        What_To_Do=_b(brand, "What_To_Do"),
        What_Not_To_Do=_b(brand, "What_Not_To_Do"),
        goal=goal,
    )

    user_content = USER_GOAL.format(
        retrieved_strategy=retrieved_strategy,
        Brand_Name=_b(brand, "Brand_Name"),
        Brand_Mission=_b(brand, "Brand_Mission"),
        Brand_Vision=_b(brand, "Brand_Vision"),
        Brand_Promise=_b(brand, "Brand_Promise"),
        Market_Positioning=_b(brand, "Market_Positioning"),
        Key_Differentiators=_b(brand, "Key_Differentiators"),
        Audience_Type=_b(brand, "Audience_Type"),
        Persona_Role=_b(brand, "Persona_Role"),
        Persona_Goals=_b(brand, "Persona_Goals"),
        Fear_And_Pain_Points=_b(brand, "Fear_And_Pain_Points"),
        Primary_Emotion=_b(brand, "Primary_Emotion"),
        Avoided_Emotion=_b(brand, "Avoided_Emotion"),
        What_To_Do=_b(brand, "What_To_Do"),
        What_Not_To_Do=_b(brand, "What_Not_To_Do"),
        goal=goal,
    )

    print("\n" + "=" * 60)
    print("PROMPT STRATEGY - SYSTEM_GOAL:")
    print("=" * 60)
    print(system_msg)
    print("\n" + "=" * 60)
    print("PROMPT STRATEGY - USER_GOAL:")
    print("=" * 60)
    print(user_content)

    prompt = _call_openai(system_msg, user_content)
 
    print(f"===========================strategy prompt===================={prompt}")

    # Return ONLY the key this node writes — avoids concurrent-write conflict in fan-out
    return {"prompt_strategy": prompt}

def generate_prompt_brand(state) -> dict:
    print("🎨 Running Generate Prompt (BRAND) Node...")

    SYSTEM_GOAL = """
    You are a senior brand identity strategist for {Brand_Name}.

    Your responsibility is to retrieve, interpret, and synthesize the brand identity rules required for safe and consistent poster generation using ONLY the retrieved brand knowledge and the brand context provided by the user.

    You must follow these non-negotiable rules at all times:

    1. Retrieve and synthesize ONLY the brand identity guidance aligned with the brand context above.
    2. Every output field MUST reflect the emotion: {Primary_Emotion} in tone, style, and identity expression.
    3. NEVER include identity or style directions that trigger {Avoided_Emotion}.
    4. All retrieved content MUST serve the goal: {goal} and align with the mission: {Brand_Mission}, vision: {Brand_Vision}, and promise: {Brand_Promise}.
    5. Apply {What_To_Do} as brand governance rules.
    6. Apply {What_Not_To_Do} as hard brand restrictions.
    7. Focus on brand identity, personality, tone, style guardrails, and forbidden style directions.
    8. Do not generate campaign strategy unless explicitly supported.
    9. Do not invent unsupported brand attributes.
    10. Do not generate the final image prompt.
    11. If any field is unavailable, return exactly: "MISSING".

    Your output must strictly follow this exact structure:

        1. Brand Personality
        2. Brand Tone of Voice
        3. Brand Emotional Direction
        4. Brand Promise Expression
        5. Market Positioning Expression
        6. Key Differentiators To Emphasize
        7. Audience Impression The Brand Should Create
        8. Visual Identity Cues
        9. Typography Guidance
        10. Brand-Safe Style Keywords
        11. Layout System Rules
        12. Logo Placement / Safe Area Rules
        13. CTA Placement Preference
        14. Footer Structure Rules
        15. Grid / Alignment / Whitespace Rules
        16. What The Brand Must Always Communicate
        17. What The Brand Must Never Communicate
        18. What To Do
        19. What Not To Doo
    """

    USER_GOAL = """
    ## BRAND CONTEXT

    {Brand_Name} is a brand operating with the mission of {Brand_Mission} and a long-term vision of {Brand_Vision}. The brand promises its customers {Brand_Promise} and is positioned in the market as {Market_Positioning}, standing apart through its key differentiators: {Key_Differentiators}. The brand primarily serves a {Audience_Type} audience, with the persona playing the role of {Persona_Role}, driven by goals such as {Persona_Goals} and facing pain points including {Fear_And_Pain_Points}. The current strategic goal guiding this retrieval is {goal}. In all communications, the brand must lead with the emotion of {Primary_Emotion} and must strictly avoid evoking {Avoided_Emotion}. The brand always follows these behavioral principles — {What_To_Do} — and must never engage in the following — {What_Not_To_Do}.

    ## RETRIEVED BRAND CONTEXT

    {retrieved_brand}

    ## TASK

    Using only the retrieved brand context and the brand details above, extract and synthesize the brand identity rules for poster generation.
    """

    # OUTPUT_FORMAT variable removed — structure is already specified in SYSTEM_GOAL above.
    brand = state.get("brand_data", {})
    goal = state.get("goal", "")

    retrieved_brand = _flatten(state.get("retrieved_docs_brand", []))

    # system_msg = SYSTEM_GOAL.format(**brand, goal=goal)
    system_msg = SYSTEM_GOAL.format(
        Brand_Name=_b(brand, "Brand_Name"),
        Brand_Mission=_b(brand, "Brand_Mission"),
        Brand_Vision=_b(brand, "Brand_Vision"),
        Brand_Promise=_b(brand, "Brand_Promise"),
        Market_Positioning=_b(brand, "Market_Positioning"),
        Key_Differentiators=_b(brand, "Key_Differentiators"),
        Audience_Type=_b(brand, "Audience_Type"),
        Persona_Role=_b(brand, "Persona_Role"),
        Persona_Goals=_b(brand, "Persona_Goals"),
        Fear_And_Pain_Points=_b(brand, "Fear_And_Pain_Points"),
        Primary_Emotion=_b(brand, "Primary_Emotion"),
        Avoided_Emotion=_b(brand, "Avoided_Emotion"),
        What_To_Do=_b(brand, "What_To_Do"),
        What_Not_To_Do=_b(brand, "What_Not_To_Do"),
        goal=goal,
    )

    user_content = USER_GOAL.format(
        retrieved_brand=retrieved_brand,
        Brand_Name=_b(brand, "Brand_Name"),
        Brand_Mission=_b(brand, "Brand_Mission"),
        Brand_Vision=_b(brand, "Brand_Vision"),
        Brand_Promise=_b(brand, "Brand_Promise"),
        Market_Positioning=_b(brand, "Market_Positioning"),
        Key_Differentiators=_b(brand, "Key_Differentiators"),
        Audience_Type=_b(brand, "Audience_Type"),
        Persona_Role=_b(brand, "Persona_Role"),
        Persona_Goals=_b(brand, "Persona_Goals"),
        Fear_And_Pain_Points=_b(brand, "Fear_And_Pain_Points"),
        Primary_Emotion=_b(brand, "Primary_Emotion"),
        Avoided_Emotion=_b(brand, "Avoided_Emotion"),
        What_To_Do=_b(brand, "What_To_Do"),
        What_Not_To_Do=_b(brand, "What_Not_To_Do"),
        goal=goal,
    )

    print("\n" + "=" * 60)
    print("PROMPT BRAND - SYSTEM_GOAL:")
    print("=" * 60)
    print(system_msg)
    print("\n" + "=" * 60)
    print("PROMPT BRAND - USER_GOAL:")
    print("=" * 60)
    print(user_content)

    prompt = _call_openai(system_msg, user_content)

    print(f"==========generate brand prompt========={prompt}")

    # Return ONLY the key this node writes — avoids concurrent-write conflict in fan-out
    return {"prompt_brand": prompt}

def merge_prompts(state) -> dict:
    print("🔀 Running Merge Prompts Node...")

    text_style = ""
    brand_data = state.get("brand_data", {})

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

    typography_section = ""
    if text_style:
        typography_section = f"""
[TYPOGRAPHY REQUIREMENTS - MANDATORY]
- Typography style from brand JSON: {text_style}
- Use this exact typography style for all text in the image
- Apply this to headline, subheadline, body text, CTA buttons, footer, and numerals
- Do not substitute it with a different typography style
- Keep all text visually consistent with this exact typography style
""".strip()

        print(f"  🔤 Adding font style from JSON to merged prompt: {text_style}")

    image_descriptions = state.get("image_descriptions", {})

    if not image_descriptions:
        print("  ⚠️ No image_descriptions in state, loading from persistent store...")
        image_descriptions = image_desc_store.load()

    print(
        f"  🔍 DEBUG: image_descriptions in state at merge_prompts: {len(image_descriptions)} descriptions"
    )

    goal = state.get("goal", "")

    if image_descriptions and goal:
        print(f"  🎯 Filtering top 5 image descriptions by goal relevance...")
        embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
        goal_embedding = embedder.embed_query(goal)

        desc_with_scores = []
        for img_path, desc in image_descriptions.items():
            desc_embedding = embedder.embed_query(desc)
            similarity = cosine_similarity([goal_embedding], [desc_embedding])[0][0]
            desc_with_scores.append((img_path, desc, similarity))

        desc_with_scores.sort(key=lambda x: x[2], reverse=True)
        top_5 = desc_with_scores[:5]

        image_descriptions = {img_path: desc for img_path, desc, _ in top_5}
        print(f"  ✅ Filtered to top 5 relevant descriptions")

    visual_content_section = ""
    if image_descriptions:
        print(
            f"  🖼 Image descriptions found in state: {len(image_descriptions)} images"
        )
        desc_parts = []
        for img_path, desc in image_descriptions.items():
            img_name = os.path.basename(img_path)
            desc_parts.append(f"--- Image: {img_name} ---\n{desc}")
        visual_content_section = f"""
[CREATIVE VISUAL CONTENT DESCRIPTIONS - STRICT REFERENCE]
The following are the top 5 goal-relevant sample creatives. You MUST follow these EXACTLY:

{chr(10).join(desc_parts)}

STRICT ANTI-HALLUCINATION RULES:
- ONLY include visual elements explicitly described above (graphs, charts, icons, people, objects, layouts)
- DO NOT invent, assume, or add any graphics, icons, charts, or elements NOT present in these descriptions
- DO NOT hallucinate visual elements that were not described
- Match the exact visual style of these samples (photograph/illustration/minimalist/infographic)
- If samples show bar charts, include bar charts; if they show lifestyle photography, include lifestyle shots
- Follow the exact composition and layout patterns described

ANTI-CUTOFF RULES:
- Ensure all text elements are fully visible within the canvas
- Do not let headline, body text, CTA, or footer get clipped at edges
- Maintain proper padding from all sides
- Keep all visual elements fully contained within the frame
""".strip()
    else:
        print("  ⚠️ No image descriptions found in state!")

    merged = f"""
[MAIN MESSAGE]
{state.get("prompt_main", "")}

[VISUAL RULES]
{state.get("prompt_metadata", "")}

[COMMUNICATION STRATEGY]
{state.get("prompt_strategy", "")}

[BRAND GUARDRAILS]
{state.get("prompt_brand", "")}

{visual_content_section}

{typography_section}

[LAYOUT ENFORCEMENT - MANDATORY]
- Follow the positional layout map extracted from metadata.
- Keep headline, body, CTA, footer, logo, and subject in the same relative zones as the sample creatives.
- Preserve the same reading flow and section stacking pattern.
- Preserve the same text-heavy side and image-heavy side where supported.
- Preserve safe empty areas for text.
- Preserve spacing, visual balance, and subject/text separation.
- Do not reposition major elements unless metadata says the position is unclear.

[FINAL EXECUTION RULES]
- Maintain brand tone and messaging.
- Keep layout clean and readable.
- Avoid cartoon, anime, vector illustration styles.
- Maintain safe margins for all text.
- Use the typography style exactly as provided in the brand JSON.
""".strip()

    db_answers = state.get("db_answers", {})
    print(f"====================Merge Prompt==================={merged}")
    return {"prompt": merged, "db_answers": db_answers}

def generate_prompt_with_placeholders(variables: dict) -> str:
    # Build the prompt using the variables as placeholders
    prompt = f"""
    Brand Tone: {variables["brand_tone"]}
    
    Persona:
    {variables["persona"]}

    Do's and Don'ts:
    {variables["dos_and_donts"]}

    Brand Mission:
    {variables["brand_mission"]}
    
    Brand Vision:
    {variables["brand_vision"]}

    Word Bank:
    {variables["word_bank"]}
    
    Your task is to create content that aligns with the above brand context. Please ensure that:
    - The tone is consistent with the brand tone.
    - The content reflects the persona's goals and pain points.
    - All do's and don'ts are respected.
    - The brand mission and vision are emphasized.
    - The word bank is used appropriately to strengthen the message.
    """
    return prompt

def write_prompt_to_txt(prompt: str, filename: str):
    with open(filename, "w") as f:
        f.write(prompt)
    print(f"Prompt written to {filename}")
