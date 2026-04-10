
from dotenv import load_dotenv

from edges.workflow import app
from nodes.common import logger


load_dotenv()


if __name__ == "__main__":
    with open("log.txt", "w", encoding="utf-8") as f:
        f.write("")

    logger.start()
    result = app.invoke(
        {
            "brand_assets_files": [
                #"Screenshot 2026-03-16 at 9.11.35 PM.png",
            ],
            "creatives_files": [
                # "Countries Inflation Rate (1).png",
                "GDP growth-01 (1).png",
                # "foreign-01 (1).png",
                # "Quick commerce (1).png",
                "FD to bonds-01.png",
                # "FD to bonds-02.png",
                # "FD to bonds-03.png",
                # "FD to bonds-04.png",
                # "FD to bonds-05.png",
                # "FD to bonds-06.png",
                # "FD to bonds-07.png",
                # "FD to bonds-08.png",
                # "FD to bonds-09.png",
            ],
            "strategy_decks_files": [
                # "Jiraaf & Altgraaf Pitch - Red & Blue Digital.pptx",
            ],
            # "brand_assets_files": [
            #     "LOGO_Niroggi_LOGO ON WHITE.png",
            #     "LOGO_Niroggi_LOGO WHITE.png"
            # ],
            # "creatives_files": [
            #     "CAROUSEL-Niroggi-2.jpg",
            #     "CAROUSEL-Niroggi-3.jpg",
            #     "CAROUSEL-Niroggi-4.jpg",
            #     "CAROUSEL-Niroggi-5.jpg"
            #     "CAROUSEL-Niroggi-6.jpg",
            #     "CAROUSEL-Niroggi-7.jpg",
            #     "CAROUSEL-Niroggi-8.jpg",
            #     "CAROUSEL-Niroggi-9.jpg",
            # ],
            # "strategy_decks_files": [
            #     "Niroggi - Brand Strategy Routes.pptx",
            # ],
            # # 1️⃣ Screen Time Awareness
            # "goal": """
            # Create an Instagram carousel encouraging parents to swap kids’ screen time with real-world activities.
            # """,
            #  2️⃣ Family Activity Ideas
            # "goal": """
            # Create an Instagram carousel showing simple activities families can use to replace screen time.
            # """,
            #  # 3️⃣ Home-Cooked Food Research
            # "goal": """
            # Create an Instagram carousel explaining research on how home-cooked meals improve healthy eating.
            # """,
            # 4️⃣ Screens and Junk Food
            # "goal": """
            # Create an Instagram post explaining how screen time increases junk food consumption in teens.
            # """,
            # 5️⃣ Kids Sleep Awareness
            # "goal": """
            # Create an Instagram awareness post about how screen habits affect children’s sleep.
            # """,
            #     #  6️⃣ Real-World Family Moments Campaign
            #     "goal": """
            #    Create an Instagram carousel encouraging families to replace digital time with real-world moments.
            #     """,
            # "goal": """
            # Create an Instagram carousel encouraging mindful eating habits for children.
            # """,
            # "goal": """
            # create a instagram post for
            # How Tariff are increasing costs for everyday goods
            # """,
            # ___________________________________________________________________
            # "goal": """
            # Create a carousel slide explaining the importance of balancing physical activity with mental wellness.
            # Use this as slide 2 in a wellness education series.
            # Format it as a social media carousel slide for Instagram
            # """,
            "goal": """
            Create a linkedin post with a horizontal bar chart comparing current vs projected quick commerce market sizes across countries.
            """,
            "question": """        
            From the provided brand content, extract:
            1) Primary theme and message direction
            2) Key brand messaging and positioning
            3) Target audience characteristics
            4) Campaign themes and concepts
            5) Communication style and tone guidelines
        
            Focus on: brand identity, messaging strategy, audience insights from the Jiraaf brand materials.
            """,
            "question_metadata": """
            From the provided Jiraaf creative samples (images/PDFs), extract ONLY what is visibly present.
  
            Return:
            1) Top 5 dominant background colors (hex where possible) + brief usage note
            2) Top 3 text colors used (hex where possible)
            3) Accent color suggestions consistent with the creatives (hex where possible)
            4) Key visual labels/themes detected (e.g., icons, shapes, objects, motifs, charts, graphs)
            5) Layout patterns observed (e.g., minimal, split layout, gradient, cards, timeline, bar charts)
            6) Typography hints (headline vs body style, weight, spacing, casing) — if unclear, say MISSING
            7) Visual content descriptions (what graphs, charts, people, objects are shown)
        
            Rules: Do not guess. If not present, return MISSING. Focus on what makes the visual unique.
            """,
            "question_strategy": """
            You are extracting complete brand strategy and guidelines from the provided content.
 
            Return the output in structured bullet points.
 
            ----------------------------------------
            SECTION 1: BRAND CORE (Internal Identity)
            ----------------------------------------
            Extract ONLY if explicitly present:
 
            - Brand Name
            - Brand Description
            - Brand Mission
            - Brand Vision
            - Brand Value Proposition
            - Key Differentiator
            - Market Position
 
            - Brand Tone Attributes
            - Primary Emotion
            - Secondary Emotion
            - Avoided Emotion
            - Sentence Style / Length (if mentioned)
 
            ----------------------------------------
            SECTION 2: AUDIENCE & PERSONA
            ----------------------------------------
 
            - Target Audience (who they are)
            - Persona (traits, behavior)
            - Goals
            - Motivations
            - Pain Points
            - Content Complexity (if mentioned)
 
            Rules:
            - Do NOT generalize audience
            - Do NOT merge multiple segments
 
            ----------------------------------------
            SECTION 3: BRAND EXPRESSION & DESIGN
            ----------------------------------------
 
            - Typography (ONLY if explicitly present, else skip)
            - Color Palette (ONLY if explicitly present)
            - Visual Identity and Design Consistency
            - Visual Themes (charts, icons, etc.)
 
            ----------------------------------------
            SECTION 4: CONTENT & COMMUNICATION STRATEGY
            ----------------------------------------
 
            - Communication Style and Content Approach
            - Messaging Themes and Strategic Focus
            - Content Formats Used (reels, webinars, etc.)
            - Platform-Specific Behavior:
            - Instagram
            - LinkedIn
            - YouTube
 
            - Social Media Challenges (if mentioned)
            - Strategy (overall content/marketing direction)
 
            ----------------------------------------
            SECTION 5: BRAND RULES & LANGUAGE SYSTEM
            ----------------------------------------
 
            - Do’s (behavioral rules)
            - Don’ts
            - Positive Word Bank
            - Negative Word Bank
            - Replaceable Words (if mentioned)
 
            ----------------------------------------
            SECTION 6: BUSINESS & MARKET CONTEXT
            ----------------------------------------
 
            - Business Problem or Opportunity
            - Competitive Landscape (ONLY if tied to brand)
            - Competitor Brands (if mentioned)
            - Compliance / Regulatory Constraints
 
            ----------------------------------------
            SECTION 7: OBJECTIVES & GROWTH
            ----------------------------------------
 
            - Marketing and Content Objectives
            - Growth Opportunities or Strategic Directions
 
            ----------------------------------------
            STRICT RULES:
 
            - Use ONLY information explicitly present in the content
            - Do NOT infer, assume, or generalize
            - Do NOT mix multiple brands
            - Preserve exact meaning (no polishing or improving)
            - Avoid generic words like “engaging”, “innovative”
            - If a field is not present, SKIP it (do NOT invent)
            - Keep output as concise bullet points
            - Only use information from the document. Do not hallucinate. If something is missing, skip it. give as same the keywords
            """,
            "question_brand": """
            Build {Brand_Name} brand guardrails STRICTLY from the provided {Brand_Name} assets. Do not use external knowledge. If any item is not explicitly supported by the assets, write MISSING (do not hallucinate).

            Return:
            1) Brand essence + audience (primary/secondary)
            2) Tone keywords + avoid list
            3) Visual style direction (photo vs illustration, minimal vs maximal, modern vs classic)
            4) Color palette with usage rules (primary/secondary/accent + contrast guidance)
            5) Typography hierarchy rules (headline/body/caption) — include font names only if present; otherwise MISSING
            6) Layout system rules (alignment, grid, whitespace, section stacking, reading flow)
            7) Logo placement and safe area rules
            8) CTA placement preference and treatment
            9) Footer structure rules
            10) Offer/content rules (hero message, CTA, any pricing/offer patterns if present)
            11) Do / Don’t guardrails (emotional + visual + copy + layout)
            12) Recommended template types (3–6) with one rule each
            """,
            "documents": [],
            "chunks": [],
            "chunks_sources": [],
            "images": [],
            "embeddings": [],
            "vectorstore": None,
            "retrieved_docs": [],
            "retrieved_docs_metadata": [],
            "retrieved_docs_strategy": [],
            "retrieved_docs_brand": [],
            "file_metadata": [],
            "target_db": "main",
            "metadata_docs": [],
            "strategy_docs": [],
            # new ones -----------------
            "prompt": "",
            "user_feedback": "",
            "saved_image_path": "",  # FIX: required by GraphState
            "brand_data": {},  # FIX: populated by load_brand_data_node
            "image_model": "gpt-image-1-mini",
            "blog_summary": "",
        }
    )


    print("\n🎯 Final Generated Image Path:")
    print(result.get("saved_image_path"))
    print("\n📝 Final Generated Blog:")
    print(result.get("blog_text", ""))


    # Print retrieved results cleanly
    print("\n" + "=" * 50)
    print("📊 RETRIEVAL RESULTS")
    print("=" * 50)

    retrieved = result.get("retrieved_docs", {})
    if isinstance(retrieved, dict):
        for db_name, docs in retrieved.items():
            print(f"\n📂 {db_name.upper()} DB ({len(docs)} results):")
            for i, doc in enumerate(docs[:2], 1):
                print(f"  --- {db_name} Result {i} ---")
                print(f"  Score: {doc.get('score', 'N/A')}")
                content = doc.get("content", "")
                if db_name == "metadata":
                    print(f"  Content: {content}")
                else:
                    print(f"  Content: {content[:300]}...")
    else:
        print(f"\n📄 Total chunks found: {len(retrieved)}")
        print(f"📁 Sources: {result.get('chunks_sources', [])}...")

    logger.finish()

    print(app.get_graph().draw_mermaid())
