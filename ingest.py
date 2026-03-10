import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_processor import GoogleVisionOCRProcessor

# Brand details (fill like this)
brand_data = {
    # Page1----------------------------------------
    # "Logo":"",
    "Brand Name": "",
    "Brand Description": "",
    "Industry/Sub-Industry": "",
    "Target Geography": "",
    "Audience Type": "",
    "Brand Color Palette": "",
    # upload color palete
    "Typography": "",
    # "Font_Upload":[],
    # page2------------------------------------------
    "Brand Mission": "",
    "Brand Vision": "",
    "Brand Promise": "",
    "Market Positioning": "",
    "Key Differentiators": "",
    "Social Media Challenges": "",
    "Role of Digital Platforms": "",
    "Business Problem or Opportunity": "",
    "Human Problem": "",
    "Insight": "",
    "Brand Advantage": "",
    "Strategy": "",
    "Industry Category": "",
    "Market Maturity": "",
    "Compliance Sensitivity": "",
    "Competotor_brand_name": "",
    "Website_Url": "",
    "Social Media Profiles": "",
    # ----------page 3-----------------
    "Core Tone Attributes": "",
    "Primary Emotion": "",
    "Secondary Emotion": "",
    "Avoided Emotion": "",
    "Sentence Length": "",
    "Perspective": "",
    "Emoji Usage": "",
    # ---------------page 4-----------------------------------------------Audience persona
    "Persona Role": "",
    "Goals": "",
    "Motivations": "",
    "Fear And Pain Points": "",
    "Objections": "",
    "content consumption behaviour": "",
    "Audience Insights": "",
    "Age Range": "",
    "Gender": "",
    "Location": "",
    "Education Level": "",
    "Employment Status": "",
    "Professional Background": "",
    "Household Size": "",
    "Langauge Prefrence": "",
    "Income Level": "",
    "Family Status or Life Stage": "",
    "Socio-economic Segment": "",
    "Digital Access": "",
    # ---------------Page5---------------------------------------
    "What To Do": "",
    "Positive Word Bank": "",
    # "Upload Positive Bank":[],
    "Replacable Words": "",
    # "Upload Replacable Bank":[],
    "What Not To Do": "",
    "Negative Word Bank": "",
    # "Upload Negative Bank":[],
    # ---------------Page6----------------------------
    "Restricted Topics": "",
    "Restricted Claims": "",
    "Blocked Words/Phrases": "",
    # -----------------page 7---------------------
    "Select Your End Goals": "",
    # -------------page8------------------------------------
    "brand_mood": "",
    "Visual identity": "",
    # "refrence Creatives" : [],
    # "Mood Boards":[],
    # ------------------page 9-----------------------------
    # "Upload Template":[],
    # "Upload Documentation":[],
    # ------------------page 10----------------------
    "Add Metadata": "",
}

# Example: update fields later

# Page1-----------------------------
# brand_data["Logo"] = r""
brand_data["Brand Name"] = "AuraSkin"
brand_data["Brand Description"] = (
    "Premium Ayurvedic skincare brand combining traditional herbal science with modern dermatology."
)
brand_data["Industry/Sub-Industry"] = "Beauty & Personal Care / Skincare"
brand_data["Target Geography"] = "India (Primary: Bangalore, Mumbai, Delhi)"
brand_data["Audience Type"] = "B2C"

brand_data["Brand Color Palette"] = {
    "Primary": {"name": "Camel", "hex": "#C49A6C"},
    "Secondary": {"name": "Light Black", "hex": "#2E2E2E"},
    "Accent": {"name": "Soft Beige", "hex": "#EAD7C0"},
    "Defined": [
        {"name": "crimson", "hex": "#E41D2D"},
        {"name": "dodger blue", "hex": "#1E90FF"},
    ],
}

brand_data["Typography"] = {
    "Typography": {"Text_style": "Playfair Display + Lato", "Size": 16}
}
# brand_data["Font_Upload"] = [
#     r"",
#     r"",
# ]

# page2------------------------------------
brand_data["Brand Mission"] = (
    "To make Ayurveda accessible through clinically proven skincare."
)
brand_data["Brand Vision"] = (
    "To become India’s most trusted herbal skincare brand by 2030."
)
brand_data["Brand Promise"] = "Safe, natural, dermatologist-tested skincare."
brand_data["Market Positioning"] = "Premium Niche"

brand_data["Key Differentiators"] = [
    "Ayurvedic ingredient sourcing transparency",
    "Clinical dermatology validation",
    "Sustainable packaging",
    "Women-led brand",
    "Cruelty-free",
]

brand_data["Social Media Challenges"] = [
    "Low engagement on reels",
    "Inconsistent tone",
    "Poor CTR",
]

brand_data["Role of Digital Platforms"] = (
    "Instagram builds awareness, website builds trust/conversion, and CRM/email supports retention."
)

brand_data["Business Problem or Opportunity"] = (
    "Improve engagement, CTR, and brand recall by producing premium, consistent content at scale."
)
brand_data["Human Problem"] = (
    "Consumers fear harmful ingredients and don’t trust skincare claims without transparency."
)
brand_data["Insight"] = (
    "Urban millennials engage more when Ayurveda is explained with premium storytelling + credible validation, not miracle promises."
)
brand_data["Brand Advantage"] = (
    "Traditional Ayurveda + modern dermatology validation, backed by transparent ingredient sourcing."
)
brand_data["Strategy"] = (
    "Educate urban millennials about Ayurveda through premium storytelling and proof-based messaging."
)

brand_data["Industry Category"] = "Beauty & Personal Care"
brand_data["Market Maturity"] = "Growing"
brand_data["Compliance Sensitivity"] = "Medium"

brand_data["Competotor_brand_name"] = ["Forest Essentials", "Kama Ayurveda", "Plum"]

brand_data["Website_Url"] = ""
brand_data["Social Media Profiles"] = {"Linkedin": "", "Instagram": "", "YouTube": ""}

# -------------page 3----------------------
brand_data["Core Tone Attributes"] = {
    "Premium": {"selected": True, "intensity": 80},
    "Empathetic": {"selected": True, "intensity": 55},
    "Inspirational": {"selected": True, "intensity": 55},
    "Professional / Formal": {"selected": True, "intensity": 50},
    "Playful": {"selected": True, "intensity": 20},
    "Authoritative": {"selected": True, "intensity": 45},
    "Bold": {"selected": False, "intensity": 0},
    "Platform Refinements": {
        "Instagram": "Slightly playful",
        "LinkedIn": "Professional",
        "YouTube": "Inspirational",
    },
}

brand_data["Primary Emotion"] = "Trust"
brand_data["Secondary Emotion"] = "Luxury"
brand_data["Avoided Emotion"] = "Aggressive"
brand_data["Sentence Length"] = "Medium (10-20 words)"
brand_data["Perspective"] = "Brand-as-human"
brand_data["Emoji Usage"] = "Minimal (0-1 per post)"

# -----------------page 4----------------------
brand_data["Persona Role"] = "Urban Conscious Millennial (Decision Maker)"
brand_data["Goals"] = "Clear skin, chemical-free beauty."
brand_data["Motivations"] = (
    "Wants safe ingredients and a premium skincare experience with credible validation."
)
brand_data["Fear And Pain Points"] = "Harmful ingredients, irritation, fake claims."
brand_data["Objections"] = (
    "Skeptical about Ayurveda claims unless backed by transparency and validation."
)
brand_data["content consumption behaviour"] = (
    "Instagram reels, beauty blogs, skincare reviews, short explainers."
)

brand_data["Age Range"] = "24–35"
brand_data["Gender"] = "All"
brand_data["Location"] = "Tier 1 cities (India)"
brand_data["Education Level"] = "Bachelor’s or Master’s"
brand_data["Employment Status"] = "Full-time / Students / Young professionals"
brand_data["Professional Background"] = "Mixed urban professionals"
brand_data["Household Size"] = "2–5"
brand_data["Langauge Prefrence"] = "English (primary) + light Hinglish"
brand_data["Income Level"] = "Upper-middle"
brand_data["Family Status or Life Stage"] = (
    "Early career / young families / self-care focused"
)
brand_data["Socio-economic Segment"] = "Upper middle"
brand_data["Digital Access"] = (
    "High—smartphone-first; active on Instagram; shops online."
)

# --------------------page5------------------
brand_data["What To Do"] = [
    "Use ingredient transparency",
    "Mention clinical validation carefully",
    "Maintain premium tone",
    "Use warm, reassuring storytelling",
]
brand_data["Positive Word Bank"] = [
    "Radiance",
    "Purity",
    "Herbal Science",
    "Gentle",
    "Trusted",
]
# brand_data["Upload Positive Bank"] = [
#     r"",
#     r"",
# ]
brand_data["Replacable Words"] = {
    "Cheap": "Accessible",
    "Fast results": "Visible results over time",
    "Guaranteed cure": "Supports skin clarity",
}
# brand_data["Upload Replacable Bank"] = [
#     r"",
#     r"",
# ]
brand_data["What Not To Do"] = [
    "Make medical claims",
    "Use discount-heavy language",
    "Use slang",
]
brand_data["Negative Word Bank"] = ["Cheap", "Fast results", "Guaranteed cure"]
# brand_data["Upload Negative Bank"] = [
#     r"",
#     r"",
# ]

# -----------------page6--------------
brand_data["Restricted Topics"] = [
    "Medical cure claims",
    "Prescription advice",
    "Legal/medical interpretations",
    "Personal data collection (Aadhaar/SSN, card numbers, passwords, OTPs)",
]

brand_data["Restricted Claims"] = [
    "100% acne cure",
    "Instant fairness",
    "Guaranteed results",
    "No side effects",
]

brand_data["Blocked Words/Phrases"] = [
    "cheap",
    "guaranteed",
    "cure",
    "instant fairness",
    "clinically proven",
]

# ---------------------page 7----------------
brand_data["Select Your End Goals"] = {
    "Reach / Impressions": False,
    "Engagement (likes, comments, saves)": True,
    "Shares / Virality": False,
    "Click-throughs (CTR)": True,
    "Follower Growth": False,
    "Conversations / DMs": False,
    "Brand Recall / Thought Leadership": True,
}

# ------------------page 8-----------------------------
brand_data["brand_mood"] = "Minimal luxury"
brand_data["Visual identity"] = "Soft light, warm skin tones, minimal luxury aesthetic"

brand_data["refrence Creatives"] = [
    r"cognixia/1-Cognixia-DEVops.pdf",
    r"cognixia/1-Cognixia-SecOps.pdf",
]

reference_creative_results = []
if brand_data["refrence Creatives"]:
    processor = GoogleVisionOCRProcessor()
    for file_path in brand_data["refrence Creatives"]:
        if os.path.isfile(file_path):
            print(f"🔍 Processing reference creative: {file_path}")
            try:
                # Extract text
                text_result = processor.extract_text_from_pdf(file_path)

                # Extract images for color analysis
                pdf_name = os.path.splitext(os.path.basename(file_path))[0]
                output_folder = f"extracted_content/{pdf_name}"
                image_paths = processor.extract_images_only(
                    file_path, output_folder, resolution=500
                )

                # Analyze colors from first image if available
                colors_data = {}
                if image_paths:
                    from PIL import Image

                    with Image.open(image_paths[0]) as img:
                        import io

                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format="PNG")
                        img_bytes = img_byte_arr.getvalue()

                    try:
                        color_analysis = processor.extract_full_image_analysis(
                            img_bytes,
                            processor._build_vision_client("org"),
                            filename=os.path.basename(file_path),
                        )
                        colors_data = {
                            "dominant_colors": color_analysis.get(
                                "dominant_colors", []
                            ),
                            "color_categories": color_analysis.get(
                                "color_categories", {}
                            ),
                        }
                    except Exception as color_err:
                        print(f"⚠️ Color analysis error: {color_err}")

                reference_creative_results.append(
                    {"file": file_path, "text": text_result, "colors": colors_data}
                )
                print(f"✅ Extracted text and colors from {file_path}")
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
                reference_creative_results.append({"file": file_path, "error": str(e)})
    print(f"Processed {len(reference_creative_results)} reference creatives")

    # Save detailed results to JSON
    with open(
        "extracted_content/reference_creatives_analysis.json", "w", encoding="utf-8"
    ) as f:
        json.dump(reference_creative_results, f, indent=2, ensure_ascii=False)
    print("✅ Saved detailed analysis to reference_creatives_analysis.json")

# brand_data["Mood Boards"] = [
#     r"",
#     r"",
# ]


# ------------------page 9 ----------------------
# brand_data["Upload Template"] = [
#     r"",
#     r"",
# ]

# brand_data["Upload Documentation"] = [
#     r"",
#     r"",
# ]

# ------------------page 10----------------------
# Select file
brand_data["Add Metadata"] = ["Template", "Graphics"]


with open("brand_data.json", "w", encoding="utf-8") as f:
    json.dump(brand_data, f, indent=2, ensure_ascii=False)

print("Saved to brand_data.json")



# # ----------------------------------for data ingest automatically dynamically-------------------------------------------

# import json
# import os

# # ─── Load brand data from JSON (single source of truth) ───────────────────────
# _JSON_PATH = os.path.join(os.path.dirname(__file__), "brand_data.json")

# with open(_JSON_PATH, "r", encoding="utf-8") as _f:
#     brand_data: dict = json.load(_f)

# # ─── Convenience accessors (all read dynamically from brand_data) ─────────────

# # Page 1
# Brand_Name              = brand_data.get("Brand Name", "")
# Brand_Description       = brand_data.get("Brand Description", "")
# Industry_Sub_Industry   = brand_data.get("Industry/Sub-Industry", "")
# Target_Geography        = brand_data.get("Target Geography", "")
# Audience_Type           = brand_data.get("Audience Type", "")
# Brand_Color_Palette     = brand_data.get("Brand Color Palette", {})
# Typography              = brand_data.get("Typography", {})

# # Page 2
# Brand_Mission               = brand_data.get("Brand Mission", "")
# Brand_Vision                = brand_data.get("Brand Vision", "")
# Brand_Promise               = brand_data.get("Brand Promise", "")
# Market_Positioning          = brand_data.get("Market Positioning", "")
# Key_Differentiators         = brand_data.get("Key Differentiators", "")
# Social_Media_Challenges     = brand_data.get("Social Media Challenges", "")
# Role_of_Digital_Platforms   = brand_data.get("Role of Digital Platforms", "")
# Business_Problem            = brand_data.get("Business Problem or Opportunity", "")
# Human_Problem               = brand_data.get("Human Problem", "")
# Insight                     = brand_data.get("Insight", "")
# Brand_Advantage             = brand_data.get("Brand Advantage", "")
# Strategy                    = brand_data.get("Strategy", "")
# Industry_Category           = brand_data.get("Industry Category", "")
# Market_Maturity             = brand_data.get("Market Maturity", "")
# Compliance_Sensitivity      = brand_data.get("Compliance Sensitivity", "")
# Competitor_Brand_Name       = brand_data.get("Competotor_brand_name", "")
# Website_Url                 = brand_data.get("Website_Url", "")
# Social_Media_Profiles       = brand_data.get("Social Media Profiles", {})

# # Page 3
# Core_Tone_Attributes = brand_data.get("Core Tone Attributes", {})
# Primary_Emotion      = brand_data.get("Primary Emotion", "")
# Secondary_Emotion    = brand_data.get("Secondary Emotion", "")
# Avoided_Emotion      = brand_data.get("Avoided Emotion", "")
# Sentence_Length      = brand_data.get("Sentence Length", "")
# Perspective          = brand_data.get("Perspective", "")
# Emoji_Usage          = brand_data.get("Emoji Usage", "")

# # Page 4 – Audience Persona
# Persona_Role                    = brand_data.get("Persona Role", "")
# Goals                           = brand_data.get("Goals", "")
# Motivations                     = brand_data.get("Motivations", "")
# Fear_And_Pain_Points            = brand_data.get("Fear And Pain Points", "")
# Objections                      = brand_data.get("Objections", "")
# Content_Consumption_Behaviour   = brand_data.get("content consumption behaviour", "")
# Age_Range                       = brand_data.get("Age Range", "")
# Gender                          = brand_data.get("Gender", "")
# Location                        = brand_data.get("Location", "")
# Education_Level                 = brand_data.get("Education Level", "")
# Employment_Status               = brand_data.get("Employment Status", "")
# Professional_Background         = brand_data.get("Professional Background", "")
# Household_Size                  = brand_data.get("Household Size", "")
# Language_Preference             = brand_data.get("Langauge Prefrence", "")
# Income_Level                    = brand_data.get("Income Level", "")
# Family_Status_or_Life_Stage     = brand_data.get("Family Status or Life Stage", "")
# Socio_Economic_Segment          = brand_data.get("Socio-economic Segment", "")
# Digital_Access                  = brand_data.get("Digital Access", "")

# # Page 5
# What_To_Do         = brand_data.get("What To Do", "")
# Positive_Word_Bank = brand_data.get("Positive Word Bank", "")
# Replacable_Words   = brand_data.get("Replacable Words", "")
# What_Not_To_Do     = brand_data.get("What Not To Do", "")
# Negative_Word_Bank = brand_data.get("Negative Word Bank", "")

# # Page 6
# Restricted_Topics   = brand_data.get("Restricted Topics", [])
# Restricted_Claims   = brand_data.get("Restricted Claims", [])
# Blocked_Words       = brand_data.get("Blocked Words/Phrases", [])

# # Page 7
# Select_End_Goals = brand_data.get("Select Your End Goals", {})

# # Page 8
# Brand_Mood      = brand_data.get("brand_mood", "")
# Visual_Identity = brand_data.get("Visual identity", "")

# # Page 10
# Add_Metadata = brand_data.get("Add Metadata", [])


# # ─── Verification (only runs when executed directly: python data.py) ──────────
# if __name__ == "__main__":
#     print("✅ brand_data.json loaded successfully!\n")
#     # Save read data to readed.json for verification
#     _READED_PATH = os.path.join(os.path.dirname(__file__), "readed.json")
#     with open(_READED_PATH, "w", encoding="utf-8") as _out:
#         json.dump(brand_data, _out, indent=2, ensure_ascii=False)
#     print(f"\n📄 Saved to readed.json for verification.")
