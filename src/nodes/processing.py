from __future__ import annotations

import os
import io
import re
import json
import base64
from pathlib import Path
from typing import List, Dict

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from nodes.common import GraphState, logger, image_desc_store
from nodes.ocr import GoogleVisionOCRProcessor
from nodes.ingestion import check_file_type

class TextProcessor:
    def preprocess_text(self, documents: List[Dict]) -> tuple[List[str], List[Dict]]:
        """Split and clean text documents into chunks with source tracking"""
        processed_docs = []
        chunks_sources = []

        for doc in documents:
            file_name = doc.get("file_name", "unknown")
            text = doc.get("text", "")

            if not text:
                continue

            # Clean lines and remove empty ones
            cleaned_text = "\n".join(
                [line.strip() for line in text.strip().split("\n") if line.strip()]
            )

            # Step 1: Split based on headers
            header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[("#", "Header")]
            )
            header_chunks = header_splitter.split_text(cleaned_text)

            # Step 2: Recursively split header chunks
            for chunk in header_chunks:
                recursive_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=400,
                    chunk_overlap=100,
                    separators=[
                        "\n\n",
                        "\n",
                        ".",
                        " ",
                    ],
                )
                text_chunks = recursive_splitter.split_text(chunk.page_content)
                processed_docs.extend(text_chunks)

                # Track source for each chunk
                for i in range(len(text_chunks)):
                    chunks_sources.append(
                        {
                            "file": file_name,
                            "chunk_index": len(processed_docs) - len(text_chunks) + i,
                        }
                    )

        return processed_docs, chunks_sources

def text_splitter_node(state: GraphState) -> GraphState:
    print("🟢 Running Text Splitter Node...")

    documents = state.get("documents", [])
    processor = TextProcessor()
    chunks, chunks_sources = processor.preprocess_text(documents)

    return {"chunks": chunks, "chunks_sources": chunks_sources}

class EmbeddingProcessor:
    def __init__(self, model: str = "text-embedding-ada-002"):
        self.embedder = OpenAIEmbeddings(model=model)

    def generate_embeddings(self, chunks: List[str]) -> List[List[float]]:
        # clean empty chunks (important)
        clean_chunks = [c.strip() for c in chunks if c and c.strip()]
        if not clean_chunks:
            return []
        # returns List[List[float]]
        return self.embedder.embed_documents(clean_chunks)

def _format_brand_data_for_embedding(brand_data: dict) -> str:
    """Convert structured brand form data into a single embeddable text block."""
    parts = []
    for key, value in brand_data.items():
        if value is None or value == "" or value == [] or value == {}:
            continue
        if isinstance(value, (dict, list)):
            value_str = json.dumps(value, ensure_ascii=False)
        else:
            value_str = str(value).strip()
        if value_str:
            parts.append(f"{key}: {value_str}")
    return "\n".join(parts)

def embeddings_node(state: GraphState) -> GraphState:
    print("🟣 Running Embeddings Node...")
    logger.log_workflow("embeddings_node")

    # DEBUG: Check incoming image_descriptions
    incoming_img_desc = state.get("image_descriptions", {})
    print(
        f"  🔍 DEBUG embeddings_node INPUT: image_descriptions count = {len(incoming_img_desc)}"
    )

    chunks = state.get("chunks", [])
    chunks_sources = state.get("chunks_sources", [])
    images = state.get("images", [])
    file_metadata = state.get("file_metadata", [])
    target_db = state.get("target_db", "main")

    combined_inputs = []
    combined_sources = []

    # Check if we are processing brand context (either Niroggi or Brand data)
    brand_data_path = None

    if target_db in ["brand", "main"]:
        brand_data_path = "Jiraaf_data.json"
    elif target_db == "faiss":
        brand_data_path = "Jiraaf_data.json"

    if brand_data_path and os.path.exists(brand_data_path):
        with open(brand_data_path, "r", encoding="utf-8") as f:
            brand_data_raw = json.load(f)  # Load the brand data

        # Format the brand data into an embeddable text block
        brand_text = _format_brand_data_for_embedding(brand_data_raw)

        # If the brand context is successfully formatted, append to the input
        if brand_text:
            combined_inputs.append(brand_text)
            combined_sources.append(
                {"file": brand_data_path, "chunk_index": 0, "db": "brand"}
            )
    else:
        print(f"⚠️ {brand_data_path} not found, skipping brand embedding.")

    # Add OCR chunks (text content from files)
    if chunks:
        combined_inputs.extend(chunks)
        combined_sources.extend(chunks_sources)

    # Process images (if any)
    if images:
        for img in images:
            image_summary = f"Image reference: {img}"
            combined_inputs.append(image_summary)
            combined_sources.append({"file": img, "chunk_index": -1})

    # Ensure we have content to embed
    if not combined_inputs:
        raise ValueError("No data found to embed.")

    # Generate embeddings
    logger.log_api_call("OpenAI Embeddings", f"Generating embeddings for {target_db}")
    processor = EmbeddingProcessor(model="text-embedding-ada-002")
    vectors = processor.generate_embeddings(combined_inputs)

    # Determine the correct FAISS index based on target_db
    save_paths = {
        "faiss": "brand_faiss_index",  # Use brand_faiss_index for Niroggi data
        "brand": "brand_data_faiss_index",  # Use a new FAISS index for Jiraaf_data.json
        "strategy": "strategy_faiss_index",
        "metadata": "metadata_faiss_index",
    }
    save_path = save_paths.get(
        target_db, "faiss_index"
    )  # Default to faiss_index if not found

    # Create or update the FAISS index
    # NOTE: VectorStoreProcessor is defined below — Python resolves this at call-time, not import-time.
    vs_proc = VectorStoreProcessor(model="text-embedding-ada-002")
    vs_proc.create_vector_store(
        chunks=combined_inputs, embeddings=vectors, save_path=save_path
    )

    # Return ONLY changed keys — fan-in with image_analyzer means both feed
    # create_all_vector_stores; returning {**state,...} would cause concurrent-write error.
    image_descriptions = state.get("image_descriptions", {})
    return {
        "chunks": combined_inputs,
        "chunks_sources": combined_sources,
        "embeddings": vectors,
        "image_descriptions": image_descriptions,
    }

class VectorStoreProcessor:
    def __init__(self, model: str = "text-embedding-ada-002"):
        self.embedder = OpenAIEmbeddings(model=model)

    def create_vector_store(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        save_path: str = "faiss_index",
    ):
        if not chunks:
            raise ValueError("Chunks are empty.")

        if not embeddings:
            raise ValueError("Embeddings are empty.")

        # Create FAISS index using precomputed embeddings
        vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip(chunks, embeddings)),
            embedding=self.embedder,
            normalize_L2=True,
        )

        # Save locally
        vectorstore.save_local(save_path)

        return vectorstore

def vector_store_node(state: GraphState) -> GraphState:
    print("🟡 Running Vector Store Node...")
    logger.log_workflow("vector_store_node")

    chunks = state.get("chunks", [])
    embeddings = state.get("embeddings", [])
    target_db = state.get("target_db", "main")

    save_paths = {
        "main": "faiss_index",
        "metadata": "metadata_faiss_index",
        "strategy": "strategy_faiss_index",
        "brand": "brand_faiss_index",
    }
    save_path = save_paths.get(target_db, "faiss_index")

    if not chunks and not embeddings:
        print(f"📂 No new chunks, loading existing vector store: {save_path}...")
        logger.log_function_call("FAISS.load_local")
        embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
        vectorstore = FAISS.load_local(
            save_path, embedder, allow_dangerous_deserialization=True
        )
    else:
        logger.log_function_call("FAISS.from_embeddings")
        logger.log_api_call(
            "OpenAI Embeddings", f"Creating FAISS vector store: {save_path}"
        )

        if not chunks:
            raise ValueError(
                "chunks not found. Run text_splitter_node before vector_store_node."
            )

        if not embeddings:
            raise ValueError(
                "embeddings not found. Run embeddings_node before vector_store_node."
            )

        processor = VectorStoreProcessor(model="text-embedding-ada-002")

        vectorstore = processor.create_vector_store(
            chunks=chunks, embeddings=embeddings, save_path=save_path
        )

    return {"vectorstore": vectorstore}

def image_analyzer_node(state: GraphState) -> GraphState:
    print("🟠 Running Image Analyzer Node...")
    logger.log_workflow("image_analyzer_node")

    documents = state.get("documents", [])
    file_paths = state.get("file_paths", [])
    processor = GoogleVisionOCRProcessor()

    all_images = []

    for doc in documents:
        doc_images = doc.get("images", [])
        all_images.extend(doc_images)

    # Also extract images from PDFs that haven't been analyzed
    for file_path in file_paths:
        ext = Path(file_path).suffix.lower()
        file_name = Path(file_path).stem

        if ext == ".pdf":
            output_folder = f"extracted_content/{file_name}"

            # Check if analysis already exists
            first_page_analysis = f"{output_folder}/page_1_analysis.json"
            if not os.path.exists(first_page_analysis):
                logger.log_api_call(
                    "Google Vision Image Properties", f"PDF: {file_name}"
                )
                image_paths = processor.extract_images_only(
                    pdf_path=file_path, output_folder=output_folder
                )
                all_images.extend(image_paths)

    return {"images": all_images}

def generate_image_descriptions_node(state: GraphState) -> GraphState:
    print("🎨 Running Image Descriptions Node...")
    logger.log_workflow("generate_image_descriptions_node")
    logger.log_api_call("OpenAI Vision", "Generating image descriptions")

    images = state.get("images", [])
    image_descriptions = {}

    if not images:
        print("  No images to describe")
        return {"image_descriptions": {}}

    cached_descriptions = image_desc_store.load()
    images_to_process = []
    for img_path in images:
        if img_path in cached_descriptions:
            print(f"  ✅ Using cached description for: {os.path.basename(img_path)}")
            image_descriptions[img_path] = cached_descriptions[img_path]
        else:
            images_to_process.append(img_path)

    if not images_to_process:
        print(f"  📝 All {len(image_descriptions)} images have cached descriptions")
        return {"image_descriptions": image_descriptions}

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    for img_path in images_to_process:
        if not os.path.exists(img_path):
            print(f"  ⚠️ Image not found: {img_path}")
            continue

        try:
            with open(img_path, "rb") as img_file:
                img_data = img_file.read()

            base64_image = base64.b64encode(img_data).decode("utf-8")

            prompt_text = """Analyze this creative/marketing material and provide a detailed description covering:

1. VISUAL CONTENT: What is physically shown (people, objects, setting, scene, background, lighting)
2. CONTENT TYPE & MEANING: What kind of content is this (e.g., lifestyle shot, product showcase, testimonial, infographic, brand story) and what message or meaning is it trying to convey
3. VISUAL ELEMENTS: Describe any graphs, charts, icons, graphics, or visual elements present and what meaning/purpose they serve
4. VISUAL STYLE: Overall style (photograph, illustration, minimalist, bold, elegant, casual, professional, etc.)
5. COMPOSITION: How elements are arranged (centered, Rule of thirds, symmetrical, asymmetrical)
6. MOOD & TONE: What emotion or feeling does this image evoke
7. COLORS & LIGHTING: Dominant colors, color mood, lighting style (natural, artificial, dramatic, soft)
8. TEXT: Any visible text and its placement/role in the design

Be specific and descriptive - this will be used to generate new marketing creatives."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
            )

            description = response.choices[0].message.content.strip()
            image_descriptions[img_path] = description

            print(f"  ✅ Generated description for: {os.path.basename(img_path)}")

            desc_file_path = os.path.splitext(img_path)[0] + "_description.txt"
            with open(desc_file_path, "w", encoding="utf-8") as f:
                f.write(description)
            print(f"  💾 Saved description to: {desc_file_path}")

        except Exception as e:
            print(f"  ⚠️ Error describing image {img_path}: {e}")
            image_descriptions[img_path] = f"Error generating description: {str(e)}"

    if image_descriptions:
        image_desc_store.save(image_descriptions)

    print(f"  📝 Generated {len(image_descriptions)} image descriptions")
    return {"image_descriptions": image_descriptions}

def create_all_vector_stores(state: GraphState) -> GraphState:
    print("📦 Creating all vector stores...")
    logger.log_workflow("create_all_vector_stores")

    chunks = state.get("chunks", [])
    embeddings = state.get("embeddings", [])
    images = state.get("images", [])
    file_metadata = state.get("file_metadata", [])
    metadata_docs = state.get("metadata_docs", [])
    strategy_docs = state.get("strategy_docs", [])
    image_descriptions = state.get("image_descriptions", {})

    embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
    processor = EmbeddingProcessor(model="text-embedding-ada-002")
    vector_processor = VectorStoreProcessor(model="text-embedding-ada-002")

    if chunks and embeddings:
        print("  📂 Saving to main FAISS index...")
        vector_processor.create_vector_store(
            chunks=chunks, embeddings=embeddings, save_path="faiss_index"
        )

    if file_metadata:
        print("  📂 Building metadata index...")
        metadata_inputs = []
        metadata_embeddings = []

        extracted_content_dir = "extracted_content"
        if os.path.exists(extracted_content_dir):
            for folder_name in os.listdir(extracted_content_dir):
                folder_path = os.path.join(extracted_content_dir, folder_name)
                if os.path.isdir(folder_path):
                    for file_name in os.listdir(folder_path):
                        if file_name.endswith("_analysis.json"):
                            analysis_path = os.path.join(folder_path, file_name)
                            try:
                                with open(analysis_path, "r", encoding="utf-8") as f:
                                    analysis = json.load(f)

                                summary_parts = []

                                if analysis.get("filename"):
                                    summary_parts.append(
                                        f"Filename: {analysis['filename']}"
                                    )

                                if analysis.get("text_preview"):
                                    summary_parts.append(
                                        f"Text Preview: {analysis['text_preview']}"
                                    )

                                if analysis.get("labels"):
                                    labels = ", ".join(
                                        [
                                            l.get("desc", l.get("description", ""))
                                            for l in analysis["labels"]
                                        ]
                                    )
                                    if labels:
                                        summary_parts.append(f"Labels: {labels}")

                                if analysis.get("dominant_colors"):
                                    colors = analysis["dominant_colors"]
                                    color_text = ", ".join(
                                        [
                                            f"{c.get('color_name', '')} ({c['hex']}, {c.get('percentage', 0)}%)"
                                            for c in colors
                                        ]
                                    )
                                    if color_text:
                                        summary_parts.append(
                                            f"Dominant colors: {color_text}"
                                        )

                                if analysis.get("color_categories"):
                                    for cat_name, colors in analysis[
                                        "color_categories"
                                    ].items():
                                        cat_colors = ", ".join(
                                            [
                                                c.get("color_name", "")
                                                for c in colors[:5]
                                            ]
                                        )
                                        if cat_colors:
                                            summary_parts.append(
                                                f"{cat_name}: {cat_colors}"
                                            )

                                if analysis.get("banners"):
                                    banner_texts = []
                                    for banner in analysis["banners"]:
                                        if banner.get("text"):
                                            banner_texts.append(banner["text"])
                                    if banner_texts:
                                        summary_parts.append(
                                            f"Banner text: {' | '.join(banner_texts)}"
                                        )

                                if analysis.get("sentences"):
                                    sentences = analysis["sentences"][:10]
                                    if sentences:
                                        sentence_parts = []
                                        for s in sentences:
                                            if isinstance(s, dict):
                                                text = s.get("text", "")
                                                text_color = s.get("text_color", {})
                                                bg_color = s.get("background_color", {})

                                                color_info = ""
                                                if (
                                                    isinstance(text_color, list)
                                                    and text_color
                                                ):
                                                    tc = text_color[0]
                                                    color_info = (
                                                        f" (text:{tc.get('hex', '')}"
                                                    )
                                                elif isinstance(text_color, dict):
                                                    color_info = f" (text:{text_color.get('hex', '')}"

                                                if isinstance(bg_color, dict):
                                                    color_info += f", bg:{bg_color.get('hex', '')}"
                                                if color_info:
                                                    color_info += ")"

                                                if text:
                                                    sentence_parts.append(
                                                        f"{text}{color_info}"
                                                    )
                                                elif text_color:
                                                    tc = (
                                                        text_color[0]
                                                        if isinstance(text_color, list)
                                                        else text_color
                                                    )
                                                    sentence_parts.append(
                                                        f"[text:{tc.get('hex', '')}]"
                                                    )
                                        if sentence_parts:
                                            summary_parts.append(
                                                f"Sentences with colors: {' | '.join(sentence_parts)}"
                                            )

                                if analysis.get("page_dimensions"):
                                    dims = analysis["page_dimensions"]
                                    summary_parts.append(
                                        f"Page dimensions: {dims.get('image_width_px', 'N/A')}x{dims.get('image_height_px', 'N/A')}px"
                                    )

                                image_summary = "\n".join(summary_parts).strip()

                                if image_summary:
                                    metadata_inputs.append(image_summary)
                            except Exception as e:
                                print(f"    ⚠️ Error reading {analysis_path}: {e}")

        if images:
            for img in images:
                analysis_path = os.path.splitext(img)[0] + "_analysis.json"
                img_name = Path(img).stem

                if os.path.exists(analysis_path):
                    with open(analysis_path, "r", encoding="utf-8") as f:
                        analysis = json.load(f)

                    summary_parts = []

                    if analysis.get("filename"):
                        summary_parts.append(f"Filename: {analysis['filename']}")

                    if analysis.get("text_preview"):
                        summary_parts.append(
                            f"Text Preview: {analysis['text_preview']}"
                        )

                    if analysis.get("labels"):
                        labels = ", ".join(
                            [
                                l.get("desc", l.get("description", ""))
                                for l in analysis["labels"]
                            ]
                        )
                        if labels:
                            summary_parts.append(f"Labels: {labels}")

                    if analysis.get("dominant_colors"):
                        colors = analysis["dominant_colors"]
                        color_text = ", ".join(
                            [
                                f"{c.get('color_name', '')} ({c['hex']}, {c.get('percentage', 0)}%)"
                                for c in colors
                            ]
                        )
                        if color_text:
                            summary_parts.append(f"Dominant colors: {color_text}")

                    if analysis.get("color_categories"):
                        for cat_name, colors in analysis["color_categories"].items():
                            cat_colors = ", ".join(
                                [c.get("color_name", "") for c in colors[:5]]
                            )
                            if cat_colors:
                                summary_parts.append(f"{cat_name}: {cat_colors}")

                    if analysis.get("banners"):
                        banner_texts = []
                        for banner in analysis["banners"]:
                            if banner.get("text"):
                                banner_texts.append(banner["text"])
                        if banner_texts:
                            summary_parts.append(
                                f"Banner text: {' | '.join(banner_texts)}"
                            )

                    if analysis.get("sentences"):
                        sentences = analysis["sentences"]
                        if sentences:
                            sentence_parts = []
                            for s in sentences:
                                if isinstance(s, dict):
                                    text = s.get("text", "")
                                    text_color = s.get("text_color", {})
                                    bg_color = s.get("background_color", {})

                                    color_info = ""
                                    if isinstance(text_color, list) and text_color:
                                        tc = text_color[0]
                                        color_info = f" (text:{tc.get('hex', '')}"
                                    elif isinstance(text_color, dict):
                                        color_info = (
                                            f" (text:{text_color.get('hex', '')}"
                                        )

                                    if isinstance(bg_color, dict):
                                        color_info += f", bg:{bg_color.get('hex', '')}"
                                    if color_info:
                                        color_info += ")"

                                    if text:
                                        sentence_parts.append(f"{text}{color_info}")
                                    elif text_color:
                                        tc = (
                                            text_color[0]
                                            if isinstance(text_color, list)
                                            else text_color
                                        )
                                        sentence_parts.append(
                                            f"[text:{tc.get('hex', '')}]"
                                        )
                            if sentence_parts:
                                summary_parts.append(
                                    f"Sentences with colors: {' | '.join(sentence_parts)}"
                                )

                    if analysis.get("page_dimensions"):
                        dims = analysis["page_dimensions"]
                        summary_parts.append(
                            f"Page dimensions: {dims.get('image_width_px', 'N/A')}x{dims.get('image_height_px', 'N/A')}px"
                        )

                    image_summary = "\n".join(summary_parts).strip()

                    if image_summary:
                        metadata_inputs.append(image_summary)

        if file_metadata:
            for meta in file_metadata:
                file_name = meta.get("file_name", "")
                if file_name:
                    metadata_str = f"Filename: {file_name}\n"
                    metadata_inputs.append(metadata_str)

        if image_descriptions:
            print(
                f"    🖼 Adding {len(image_descriptions)} image descriptions to metadata index..."
            )
            for img_path, description in image_descriptions.items():
                img_name = os.path.basename(img_path)
                desc_str = f"Image: {img_name}\nImage Description: {description}"
                metadata_inputs.append(desc_str)

        if metadata_inputs:
            metadata_embeddings = processor.generate_embeddings(metadata_inputs)
            print(f"    📊 Created {len(metadata_embeddings)} metadata embeddings")
            vector_processor.create_vector_store(
                chunks=metadata_inputs,
                embeddings=metadata_embeddings,
                save_path="metadata_faiss_index",
            )

    if strategy_docs:
        print("  📂 Building strategy index...")
        strategy_inputs = []
        for doc in strategy_docs:
            text = doc.get("text", "")
            if text:
                strategy_inputs.append(text)

        if strategy_inputs:
            strategy_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=100
            )
            strategy_chunks = strategy_splitter.split_text("\n\n".join(strategy_inputs))
            strategy_embeddings = processor.generate_embeddings(strategy_chunks)

            print(f"    📊 Created {len(strategy_embeddings)} strategy embeddings")
            vector_processor.create_vector_store(
                chunks=strategy_chunks,
                embeddings=strategy_embeddings,
                save_path="strategy_faiss_index",
            )

    # Build brand FAISS index from brand_data.json
    brand_data_path = "Jiraaf_data.json"
    if os.path.exists(brand_data_path):
        print("  📂 Building brand index...")
        try:
            with open(brand_data_path, "r", encoding="utf-8") as f:
                brand_data_raw = json.load(f)
            brand_text = _format_brand_data_for_embedding(brand_data_raw)
            if brand_text:
                brand_embeddings = processor.generate_embeddings([brand_text])
                print(f"    📊 Created {len(brand_embeddings)} brand embeddings")
                vector_processor.create_vector_store(
                    chunks=[brand_text],
                    embeddings=brand_embeddings,
                    save_path="brand_faiss_index",
                )
        except Exception as e:
            print(f"    ⚠️ Error building brand index: {e}")

    return {**state, "image_descriptions": image_descriptions}
