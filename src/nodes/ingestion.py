from __future__ import annotations

import os
import re
from pathlib import Path
from datetime import datetime
from typing import List

from docx import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from nodes.common import GraphState, logger
from nodes.ocr import GoogleVisionOCRProcessor

def check_file_type(file_path: str) -> str:
    """Determine which FAISS index a file belongs to."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pptx":
        return "strategy"
    elif ext in {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".tif",
        ".tiff",
        ".bmp",
        ".docx",
    }:
        return "metadata"
    return "main"

def find_files_in_brand_folders(
    file_names: List[str], base_path: str = "."
) -> List[str]:
    """Find files by name in brand folder subdirectories.

    Searches for files in folders matching patterns:
    - {Brand} Brand Assets / Brand Assets / brand_assets
    - {Brand} Creatives / Creatives / creatives
    - {Brand} Strategy Decks / Strategy Decks / strategy_decks
    """
    found_paths = []
    file_names_set = set(file_names)

    folder_patterns = [
        "brand assets",
        "creatives",
        "strategy decks",
        "strategy_decks",
    ]

    for root, dirs, files in os.walk(base_path):
        root_lower = root.lower()

        for pattern in folder_patterns:
            if pattern in root_lower:
                for f in files:
                    if f in file_names_set:
                        full_path = os.path.join(root, f)
                        if full_path not in found_paths:
                            found_paths.append(full_path)
                            print(f"  📁 Found '{f}' in {root}")
                break

    for fn in file_names:
        if fn not in [os.path.basename(p) for p in found_paths]:
            if os.path.exists(fn):
                found_paths.append(os.path.abspath(fn))
                print(f"  📁 Found '{fn}' (direct path)")
            else:
                print(f"  ⚠️ File not found: {fn}")

    return found_paths

def resolve_brand_file_paths(state: GraphState) -> GraphState:
    """Resolve brand file paths from file names to full paths."""
    print("🔍 Resolving brand folder file paths...")

    existing_file_paths = state.get("file_paths", [])
    brand_assets = state.get("brand_assets_files", [])
    creatives = state.get("creatives_files", [])
    strategy_decks = state.get("strategy_decks_files", [])

    all_file_names = brand_assets + creatives + strategy_decks

    if existing_file_paths and not all_file_names:
        print("  Using existing file_paths directly (backward compatible)")
        return state

    if not all_file_names:
        print("  No file names provided!")
        return state

    resolved_paths = find_files_in_brand_folders(all_file_names)

    creatives_count = len(creatives)
    print(f"  ✅ Resolved {len(resolved_paths)} files from brand folders")
    print(f"  📊 Creative samples count: {creatives_count}")

    return {"file_paths": resolved_paths, "creatives_count": creatives_count}

def filter_files_to_ocr(state: GraphState) -> GraphState:
    print("🔍 Running Filter Files Node...")
    logger.log_workflow("filter_files")

    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    faiss_path = "metadata_faiss_index"
    file_paths = state.get("file_paths", [])

    print(
        f"  FILTER INPUT: {len(file_paths)} files: {[os.path.basename(f) for f in file_paths]}"
    )

    if not file_paths:
        return state

    if not os.path.exists(faiss_path):
        print(f"📂 Vector store not found, OCR all {len(file_paths)} files")
        return {"files_to_ocr": file_paths}

    try:
        embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
        vectorstore = FAISS.load_local(
            faiss_path, embedder, allow_dangerous_deserialization=True
        )

        existing_content = []
        for doc in vectorstore.docstore._dict.values():  # type: ignore
            existing_content.append(doc.page_content)

        print(f"  DEBUG: Total docs in vector DB: {len(existing_content)}")

        files_to_ocr = []
        skipped_files = []

        for file_path in file_paths:
            fname = os.path.basename(file_path)
            found = any(
                re.search(rf"Filename:\s*{re.escape(fname)}", content, re.IGNORECASE)
                for content in existing_content
            )
            if found:
                skipped_files.append(file_path)
                print(f"⏭️ Skipping '{fname}'")
            else:
                files_to_ocr.append(file_path)
                print(f"✅ OCR '{fname}'")

        print(
            f"  FILTER OUTPUT: {len(files_to_ocr)} files to OCR: {[os.path.basename(f) for f in files_to_ocr]}"
        )

        return {"files_to_ocr": files_to_ocr}
    except Exception as e:
        print(f"📂 Error filtering files: {e}, OCR all files")
        return {"files_to_ocr": file_paths}

def meta_node(state: GraphState) -> GraphState:
    print("⚪ Running Meta Node...")
    logger.log_workflow("meta_node")

    file_paths = state["file_paths"]
    unique_paths = list(dict.fromkeys([os.path.abspath(p) for p in file_paths]))
    file_metadata = []

    for file_path in unique_paths:
        metadata = {}

        if os.path.exists(file_path):
            file_stats = os.stat(file_path)

            metadata = {
                "file_name": os.path.basename(file_path),
                "file_path": os.path.abspath(file_path),
                "file_size_bytes": file_stats.st_size,
                "created_time": datetime.fromtimestamp(file_stats.st_ctime),
                "modified_time": datetime.fromtimestamp(file_stats.st_mtime),
                "is_file": os.path.isfile(file_path),
                "is_directory": os.path.isdir(file_path),
            }

            for key, value in metadata.items():
                print(f"{key}: {value}")
        else:
            print(f"File not found: {file_path}")

        file_metadata.append(metadata)

    # return {**state, "file_metadata": file_metadata}
    return {"file_metadata": file_metadata}

def ocr_node(state: GraphState) -> GraphState:
    print("🔵 Running OCR Node...")
    logger.log_workflow("ocr_node")
    logger.log_function_call("GoogleVisionOCRProcessor")

    file_paths = state.get("files_to_ocr", [])

    if not file_paths:
        print("📂 No files to OCR (all files already in index)")
        return {"documents": [], "images": []}

    processor = GoogleVisionOCRProcessor()

    all_documents = []
    all_images = []

    print(
        f"  OCR RECEIVED: {len(file_paths)} files: {[os.path.basename(f) for f in file_paths]}"
    )

    for file_path in file_paths:
        ext = Path(file_path).suffix.lower()
        file_name = Path(file_path).stem

        print(f"📄 Processing: {file_path}")

        if ext == ".pdf":
            logger.log_api_call("Google Vision OCR", f"PDF extraction: {file_name}")
            extracted_text = processor.extract_text_from_pdf(
                pdf_path=file_path, user_type="org"
            )
            all_documents.append(
                {
                    "file_name": file_name,
                    "file_path": file_path,
                    "text": extracted_text,
                    "images": [],
                }
            )

        elif ext in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}:
            logger.log_api_call("Google Vision OCR", f"Image extraction: {file_name}")
            extracted_text = processor.extract_text_from_image_file(
                image_path=file_path, user_type="org"
            )
            img_name = Path(file_path).stem
            output_folder = f"extracted_content/{img_name}"
            stored_img = processor.save_and_analyze_image_file(
                image_path=file_path, output_folder=output_folder, user_type="org"
            )
            all_documents.append(
                {
                    "file_name": file_name,
                    "file_path": file_path,
                    "text": extracted_text,
                    "images": [stored_img],
                }
            )
            all_images.append(stored_img)

        elif ext == ".pptx":
            logger.log_api_call(
                "Google Vision OCR", f"PPTX text extraction: {file_name}"
            )
            result = processor.extract_text_from_pptx(pptx_path=file_path)
            all_documents.append(
                {
                    "file_name": file_name,
                    "file_path": file_path,
                    "text": result["text"],
                    "images": [],
                }
            )

        elif ext == ".docx":
            doc_name = Path(file_path).stem
            output_dir = f"extracted_content/{doc_name}"
            os.makedirs(output_dir, exist_ok=True)

            doc = Document(file_path)
            extracted_text = "\n".join([para.text for para in doc.paragraphs])

            out_txt = os.path.join(output_dir, f"{doc_name}_ocr.txt")
            with open(out_txt, "w", encoding="utf-8") as f:
                f.write(extracted_text)
            print("✅ Text saved to:", out_txt)

            # Extract images from docx
            logger.log_api_call("Google Vision Image Properties", f"DOCX: {doc_name}")
            docx_images = processor.extract_images_from_docx(
                docx_path=file_path, output_dir=output_dir, user_type="org"
            )
            all_images.extend(docx_images)

            all_documents.append(
                {
                    "file_name": file_name,
                    "file_path": file_path,
                    "text": extracted_text,
                    "images": docx_images,
                }
            )

        else:
            raise ValueError(f"Unsupported file type: {ext} ({file_path})")

    return {"documents": all_documents, "images": all_images}

def split_by_type_node(state: GraphState) -> GraphState:
    print("🔀 Running Split by Type Node...")
    logger.log_workflow("split_by_type_node")

    documents = state.get("documents", [])
    file_paths = state.get("file_paths", [])

    metadata_docs = []
    strategy_docs = []

    for i, file_path in enumerate(file_paths):
        doc = documents[i] if i < len(documents) else {}
        file_type = check_file_type(file_path)

        if file_type == "strategy":
            strategy_docs.append(doc)
        else:
            metadata_docs.append(doc)

    print(f"  📄 Metadata docs: {len(metadata_docs)}")
    print(f"  📊 Strategy docs: {len(strategy_docs)}")

    return {
        "metadata_docs": metadata_docs,
        "strategy_docs": strategy_docs,
    }
