from __future__ import annotations

import os
import re
from typing import Literal

from openai import OpenAI
from langgraph.graph import END
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from nodes.common import GraphState

def check_files_in_vector_db(state: GraphState) -> Literal["in_db", "not_in_db"]:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    faiss_path = "metadata_faiss_index"
    file_metadata = state.get("file_metadata", [])

    if not file_metadata:
        print("📂 No file metadata, running full pipeline")
        return "not_in_db"

    if not os.path.exists(faiss_path):
        print("📂 Vector store not found, running full pipeline")
        return "not_in_db"

    try:
        embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
        vectorstore = FAISS.load_local(
            faiss_path, embedder, allow_dangerous_deserialization=True
        )

        file_names = [
            m.get("file_name", "") for m in file_metadata if m.get("file_name")
        ]

        if not file_names:
            print("📂 No file names to check, running full pipeline")
            return "not_in_db"

        existing_content = []
        for doc in vectorstore.docstore._dict.values():  # type: ignore
            existing_content.append(doc.page_content)

        for fname in file_names:
            found = any(
                re.search(rf"Filename:\s*{re.escape(fname)}", content, re.IGNORECASE)
                for content in existing_content
            )
            if not found:
                print(f"📂 File '{fname}' not in vector DB, running full pipeline")
                return "not_in_db"

        print(f"📂 All {len(file_names)} files exist in vector DB, skipping OCR")
        return "in_db"
    except Exception as e:
        print(f"📂 Error checking vector DB: {e}, running full pipeline")
        return "not_in_db"

def process_feedback(state: GraphState):
    user_feedback = (state.get("user_feedback") or "").strip()

    if not user_feedback or user_feedback.lower().startswith("y"):
        return END

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    routing_prompt = f"""
    You are a workflow decision engine.

    User feedback:
    "{user_feedback}"

    If the user wants to modify the EXISTING image (small change, addition, correction),
    respond with: EDIT

    Editing rules:
    - Modify ONLY the specific thing the user asked to change.
    - Keep the rest of the image exactly the same.
    - Do NOT change layout, spacing, composition, typography, image objects, structure, or styling unless explicitly requested.
    - Do NOT regenerate or redesign the image.
    - Preserve the original image and apply only the requested edit.

    Respond with ONLY one word: EDIT
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": routing_prompt}],
    )

    decision = response.choices[0].message.content.strip().upper()
    print("🧠 Routing Decision:", decision)

    if decision == "EDIT":
        return "edit_image"

    return END

def check_vector_store_exists(state: GraphState) -> Literal["exists", "not_exists"]:
    faiss_path = "metadata_faiss_index"
    if os.path.exists(faiss_path):
        index_files = ["index.faiss", "index.pkl"]
        if all(os.path.exists(os.path.join(faiss_path, f)) for f in index_files):
            print("📂 Vector store exists")
            return "exists"
    print("📂 Vector store does not exist")
    return "not_exists"
