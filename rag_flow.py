import os
import json
import shutil
import time
from datetime import datetime
from dotenv import load_dotenv
import functools
from openai import OpenAI

load_dotenv()


class APILogger:
    def __init__(self, log_file="log.txt"):
        self.log_file = log_file
        self.api_calls = {}
        self.function_calls = {}
        self.workflow_steps = []
        self.start_time = None

    def log_api_call(self, api_name, details=""):
        if api_name not in self.api_calls:
            self.api_calls[api_name] = 0
        self.api_calls[api_name] += 1
        self._write_log(f"API CALL: {api_name} - {details}")

    def log_function_call(self, func_name):
        if func_name not in self.function_calls:
            self.function_calls[func_name] = 0
        self.function_calls[func_name] += 1
        self._write_log(f"FUNCTION CALL: {func_name}")

    def log_workflow(self, step):
        self.workflow_steps.append(step)
        self._write_log(f"WORKFLOW: {step}")

    def _write_log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")

    def start(self):
        self.start_time = time.time()
        self._write_log("=" * 50)
        self._write_log("STARTED NEW RUN")
        self._write_log("=" * 50)

    def finish(self):
        elapsed = time.time() - self.start_time if self.start_time else 0
        self._write_log("=" * 50)
        self._write_log("FINISHED RUN")
        self._write_log(f"Total time: {elapsed:.2f} seconds")
        self._write_log("API CALLS SUMMARY:")
        for api, count in self.api_calls.items():
            self._write_log(f"  {api}: {count}")
        self._write_log("FUNCTION CALLS SUMMARY:")
        for func, count in self.function_calls.items():
            self._write_log(f"  {func}: {count}")
        self._write_log("WORKFLOW STEPS:")
        for step in self.workflow_steps:
            self._write_log(f"  -> {step}")
        self._write_log("=" * 50)


logger = APILogger("log.txt")


def track_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.log_function_call(func.__name__)
        return func(*args, **kwargs)

    return wrapper


# Clear log file at start
with open("log.txt", "w") as f:
    f.write("")

logger.start()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import Literal
from typing import List, Dict


import pickle
from pathlib import Path

from langgraph.graph import END, StateGraph, START

import pdfplumber
from google.cloud import vision
from google.oauth2 import service_account

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS"
)

# ----------------------------------------------------------------------------------------------------------------------


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        file_paths: list of file paths to process
        mode: operation mode - "retrieve" or "chat"
        question: question
        generation: LLM generation
        documents: list of document dicts with metadata
        chunks: list of text_sources: source chunks
        chunks metadata for each chunk
        images: list of image paths
        embeddings: list of embeddings
        vectorstore: FAISS vectorstore
        retrieved_docs: retrieved documents
        file_metadata: metadata for each file
    """

    file_paths: List[str]
    mode: Literal["retrieve", "chat"]
    question: str
    question_metadata: str
    question_strategy: str
    question_brand: str
    generation: str
    documents: List[Dict]
    chunks: List[str]
    chunks_sources: List[Dict]
    images: List[str]
    embeddings: List[List[float]]
    vectorstore: object
    retrieved_docs: List[str]
    retrieved_docs_metadata: List[Dict]
    retrieved_docs_strategy: List[Dict]
    retrieved_docs_brand: List[Dict]
    file_metadata: List[Dict]
    target_db: str
    metadata_docs: List[Dict]
    strategy_docs: List[Dict]
    # --------generation--------
    prompt: str
    image_url: str
    user_feedback: str
    goal: str
    


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


# ----------------------------------------------------------------------------------------------------------------------
# Node0: Meta Node (extracts file metadata before OCR)
from pathlib import Path
from datetime import datetime


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
    return {**state, "file_paths": unique_paths, "file_metadata": file_metadata}


# ----------------------------------------------------------------------------------------------------------------------
# Node1: OCR Node
from pathlib import Path
from ocr_processor import GoogleVisionOCRProcessor
from docx import Document


def ocr_node(state: GraphState) -> GraphState:
    print("🔵 Running OCR Node...")
    logger.log_workflow("ocr_node")
    logger.log_function_call("GoogleVisionOCRProcessor")

    file_paths = state["file_paths"]
    processor = GoogleVisionOCRProcessor()

    all_documents = []
    all_images = []

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

    return {**state, "documents": all_documents, "images": all_images}


from pathlib import Path
from ocr_processor import GoogleVisionOCRProcessor


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
        **state,
        "metadata_docs": metadata_docs,
        "strategy_docs": strategy_docs,
    }


# ---------------------------------------------------------------------------------------------------------------------
# Node2: Text Splitter Node
from typing import List
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)


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


# Node function
def text_splitter_node(state: GraphState) -> GraphState:
    print("🟢 Running Text Splitter Node...")

    documents = state.get("documents", [])
    processor = TextProcessor()
    chunks, chunks_sources = processor.preprocess_text(documents)

    return {"chunks": chunks, "chunks_sources": chunks_sources}


# ---------------------------------------------------------------------------------------------------------------------
# Node3: Embedding Node
from typing import List
from langchain_openai import OpenAIEmbeddings


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


# def embeddings_node(state: GraphState) -> GraphState:
#     print("🟣 Running Embeddings Node...")

#     chunks = state.get("chunks", [])
#     if not chunks:
#         raise ValueError("chunks not found. Run text_splitter_node before embeddings_node.")

#     processor = EmbeddingProcessor(model="text-embedding-ada-002")
#     vectors = processor.generate_embeddings(chunks)

#     return {
#         **state,
#         "embeddings": vectors
#     }


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

    chunks = state.get("chunks", [])
    chunks_sources = state.get("chunks_sources", [])
    images = state.get("images", [])
    file_metadata = state.get("file_metadata", [])
    target_db = state.get("target_db", "main")

    combined_inputs = []
    combined_sources = []

    if target_db == "brand":
        brand_data_path = "brand_data.json"
        if os.path.exists(brand_data_path):
            with open(brand_data_path, "r", encoding="utf-8") as f:
                brand_data_raw = json.load(f)
            brand_text = _format_brand_data_for_embedding(brand_data_raw)
            if brand_text:
                combined_inputs.append(brand_text)
                combined_sources.append(
                    {"file": "brand_data.json", "chunk_index": 0, "db": "brand"}
                )
        else:
            print("⚠️ brand_data.json not found, skipping brand embedding.")
    elif target_db == "strategy":
        if chunks:
            combined_inputs.extend(chunks)
            combined_sources.extend(chunks_sources)
    elif target_db == "metadata":
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
                        summary_parts.append(analysis["text_preview"])

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
                                [c.get("color_name", "") for c in colors[:3]]
                            )
                            if cat_colors:
                                summary_parts.append(f"{cat_name}: {cat_colors}")

                    image_summary = "\n".join(summary_parts).strip()

                    if image_summary:
                        combined_inputs.append(image_summary)
                        combined_sources.append(
                            {"file": img_name, "chunk_index": -1, "db": "metadata"}
                        )
        if file_metadata:
            for meta in file_metadata:
                file_name = meta.get("file_name", "")
                if file_name:
                    metadata_str = f"Filename: {file_name}\n"
                    combined_inputs.append(metadata_str)
                    combined_sources.append(
                        {"file": file_name, "chunk_index": -2, "db": "metadata"}
                    )
    else:
        if chunks:
            combined_inputs.extend(chunks)
            combined_sources.extend(chunks_sources)

        if file_metadata:
            for meta in file_metadata:
                file_name = meta.get("file_name", "")
                file_ext = meta.get("file_extension", "")
                if file_name:
                    metadata_str = f"Filename: {file_name}{file_ext}\n"
                    combined_inputs.append(metadata_str)
                    combined_sources.append({"file": file_name, "chunk_index": -2})

        if images:
            for img in images:
                analysis_path = os.path.splitext(img)[0] + "_analysis.json"
                img_name = Path(img).stem

                if os.path.exists(analysis_path):
                    with open(analysis_path, "r", encoding="utf-8") as f:
                        analysis = json.load(f)

                    summary_parts = []

                    if analysis.get("text_preview"):
                        summary_parts.append(analysis["text_preview"])

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

                    image_summary = "\n".join(summary_parts).strip()

                    if image_summary:
                        combined_inputs.append(image_summary)
                        combined_sources.append({"file": img_name, "chunk_index": -1})
                else:
                    combined_inputs.append(f"Image reference: {img}")
                    combined_sources.append({"file": img_name, "chunk_index": -1})

    if not combined_inputs:
        raise ValueError("No data found to embed.")

    logger.log_api_call("OpenAI Embeddings", "Generating embeddings")
    processor = EmbeddingProcessor(model="text-embedding-ada-002")
    vectors = processor.generate_embeddings(combined_inputs)

    return {
        **state,
        "chunks": combined_inputs,
        "chunks_sources": combined_sources,
        "embeddings": vectors,
    }


# ---------------------------------------------------------------------------------------------------------------------
# Node4: Vector Store Node (knowledge Base)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import List


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

    return {**state, "vectorstore": vectorstore}


# ---------------------------------------------------------------------------------------------------------------------
# Node5: Retriever Node (To check what are the embeddings it contains)

from typing import List


class RetrieverProcessor:
    def __init__(self, k: int = 4):
        self.k = k

    def retrieve(self, question: str, vectorstore):
        if not question:
            raise ValueError("Question is empty.")

        if not vectorstore:
            raise ValueError("Vectorstore is not available.")

        docs_and_scores = vectorstore.similarity_search_with_score(question, k=self.k)

        return docs_and_scores


# node function
def retriever_node(state: GraphState) -> GraphState:
    print("🟢 Running Retriever Node...")
    logger.log_workflow("retriever_node")

    question = state.get("question", "")
    vectorstore = state.get("vectorstore", None)

    if not question:
        raise ValueError("question not found in state.")

    if not vectorstore:
        raise ValueError("vectorstore not found. Run vector_store_node first.")

    print(f"\n🔎 Question: {question}\n")

    processor = RetrieverProcessor(k=4)  # it was 4 before

    docs_and_scores = processor.retrieve(question=question, vectorstore=vectorstore)

    retrieved_docs = []

    print("📌 Retrieved Chunks With Similarity Scores:\n")

    for i, (doc, score) in enumerate(docs_and_scores, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Similarity Score: {score}")
        print(doc.page_content)  # print first 500 chars only
        retrieved_docs.append(doc.page_content)

    return {**state, "retrieved_docs": retrieved_docs}


def multi_retriever_node(state: GraphState) -> GraphState:
    print("🔍 Running Multi-Retriever Node...")
    logger.log_workflow("multi_retriever_node")

    question_main = state.get("question", "")
    question_metadata = state.get("question_metadata", "")
    question_strategy = state.get("question_strategy", "")
    question_brand = state.get("question_brand", "")

    embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
    processor = RetrieverProcessor(k=5)

    faiss_indexes = {
        "main": ("faiss_index", question_main),
        "metadata": ("metadata_faiss_index", question_metadata),
        "strategy": ("strategy_faiss_index", question_strategy),
        "brand": ("brand_faiss_index", question_brand),
    }

    all_results = {}
    metadata_results = []
    strategy_results = []
    brand_results = []

    for db_name, (db_path, question) in faiss_indexes.items():
        if not question:
            print(f"📂 No question for {db_name} DB, skipping...")
            all_results[db_name] = []
            continue

        if os.path.exists(db_path):
            try:
                print(f"\n🔎 {db_name.upper()} Question: {question}")
                vectorstore = FAISS.load_local(
                    db_path, embedder, allow_dangerous_deserialization=True
                )
                docs_and_scores = processor.retrieve(
                    question=question, vectorstore=vectorstore
                )

                results = []
                for doc, score in docs_and_scores:
                    results.append(
                        {
                            "content": doc.page_content,
                            "score": score,
                            "db": db_name,
                        }
                    )

                all_results[db_name] = results
                if db_name == "metadata":
                    metadata_results = results
                elif db_name == "strategy":
                    strategy_results = results
                elif db_name == "brand":
                    brand_results = results
                print(f"📂 Retrieved {len(results)} docs from {db_name} DB")
            except Exception as e:
                print(f"⚠️ Error loading {db_path}: {e}")
                all_results[db_name] = []
        else:
            print(f"📂 {db_path} does not exist, skipping...")
            all_results[db_name] = []

    return {
        **state,
        "retrieved_docs": all_results,
        "retrieved_docs_metadata": metadata_results,
        "retrieved_docs_strategy": strategy_results,
        "retrieved_docs_brand": brand_results,
    }


# ---------------------------------------------------------------------------------------------------------------------
# Node6: Chat Node (System + User Prompt using gpt-4o)

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


class ChatProcessor:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model=model, temperature=0)

        # 🔵 Define system + user roles explicitly
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Role:
                    Your function is to generate a structured, well-organized blog strictly based on the provided context document.

                    Instructions:
                    - Use ONLY the provided context.
                    - You may analyze, synthesize, and evaluate information from the context.
                    - Do NOT introduce external knowledge.
                    - Base your answer strictly on what is present in the context.
                    - If there is insufficient information, clearly state that.
                    """,
                ),
                ("user", "Context:\n{context}\n\nQuestion:\n{question}"),
            ]
        )

        self.output_parser = StrOutputParser()

        # 🔵 Pre-build chain once (better than rebuilding every call)
        self.chain = self.prompt | self.llm | self.output_parser

    def generate_answer(self, question: str, context: str) -> str:
        return self.chain.invoke({"question": question, "context": context})


# Node function
def chat_node(state: GraphState) -> GraphState:
    print("🔵 Running Chat Node...")
    logger.log_workflow("chat_node")
    logger.log_api_call("OpenAI Chat", "Generating responses")

    question = state.get("question", "")
    question_metadata = state.get("question_metadata", "")
    question_strategy = state.get("question_strategy", "")
    question_brand = state.get("question_brand", "")
    retrieved_docs = state.get("retrieved_docs", {})

    if not retrieved_docs:
        raise ValueError("retrieved_docs not found. Run multi_retriever_node first.")

    processor = ChatProcessor()
    answers = {}

    if isinstance(retrieved_docs, dict):
        db_questions = {
            "main": question,
            "metadata": question_metadata,
            "strategy": question_strategy,
            "brand": question_brand,
        }

        for db_name, db_question in db_questions.items():
            docs = retrieved_docs.get(db_name, [])

            if not db_question:
                answers[db_name] = "No question provided for this DB."
                continue

            if not docs:
                answers[db_name] = (
                    f"No documents found in {db_name} DB to answer the question."
                )
                continue

            context_parts = [f"=== {db_name.upper()} RESULTS ===\n"]
            for doc in docs:
                content = doc.get("content", "")
                if content:
                    context_parts.append(content)

            context = "\n\n".join(context_parts)
            answer = processor.generate_answer(question=db_question, context=context)
            answers[db_name] = answer

            print(f"\n📢 {db_name.upper()} Answer:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
    else:
        if question:
            context = "\n\n".join(retrieved_docs)
            answers["main"] = processor.generate_answer(
                question=question, context=context
            )
            print(f"\n📢 MAIN Answer:")
            print(answers["main"])

    return {**state, "generation": str(answers)}


def generate_prompt(state: GraphState) -> GraphState:
    print("🎨 Running Generate Prompt Node...")

    goal = state.get("goal", "")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def flatten_results(results):
        if isinstance(results, list):
            return "\n".join([r.get("content", "") if isinstance(r, dict) else str(r) for r in results])
        return str(results)

    # topic = state.get("generation", "")
    retrieved_main = flatten_results(state.get("retrieved_docs", {}).get("main", []))
    retrieved_metadata = flatten_results(state.get("retrieved_docs_metadata", []))
    retrieved_strategy = flatten_results(state.get("retrieved_docs_strategy", []))
    retrieved_brand = flatten_results(state.get("retrieved_docs_brand", []))
    context = f"""
    MAIN CONTENT:
    {retrieved_main}

    METADATA DESIGN:
    {retrieved_metadata}

    STRATEGY:
    {retrieved_strategy}

    BRAND:
    {retrieved_brand}
    """
    feedback = state.get("user_feedback", "")

    system_msg = """
    You are a senior enterprise visual director and DALL-E 3 production prompt engineer.

    You generate ultra-detailed, layout-controlled, brand-consistent image prompts
    for enterprise LinkedIn marketing templates.

    CRITICAL RULES:

    1. You MUST translate brand + metadata + strategy into precise visual composition.
    2. Specify spatial hierarchy (top / mid / bottom / left / right).
    3. Define background style and depth.
    4. Define text block placement.
    5. Define CTA placement.
    6. Define color ratios (percentage usage).
    7. Define typography style description (modern sans serif, bold geometric, etc.).
    8. Define visual tone (minimal, premium, tech-forward, corporate).
    9. No generic marketing fluff.
    10. Output ONLY one final DALL-E optimized prompt paragraph.
    11. Do NOT output explanations.
    """

    user_content = f"""
        GOAL:
        {goal}

        Using the following structured enterprise brand intelligence,
        generate a production-grade DALL-E 3 prompt for a 1024x1024 LinkedIn carousel slide.

        CONTEXT:
        {context}

        CRITICAL REQUIREMENTS:
        - The visual must strictly reflect brand palette hierarchy.
        - Background must follow dominant color pattern from metadata DB.
        - Text colors must respect contrast rules from metadata DB.
        - Layout structure must follow observed layout style patterns.
        - CTA placement must follow strategy DB.
        - Tone must match brand emotional intensity.
        - Typography must follow brand guardrails.
        - Composition must be spatially defined (top/middle/bottom zones).
        - Output must be a single ultra-detailed rendering prompt paragraph.

        Only output the final DALL-E ready prompt.
        """

    if feedback:
        user_content += f"\n\nModify previous concept according to this feedback: {feedback}"

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ],
    )

    state["prompt"] = response.choices[0].message.content.strip()
    return state


import requests
import os
from datetime import datetime

import base64
import os
from datetime import datetime

def generate_image(state: GraphState) -> GraphState:
    print("🖼 Running Generate Image Node...")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = state["prompt"]

    response = client.images.generate(
        model="gpt-image-1-mini",
        prompt=prompt,
        size="1024x1024",
    )

    # 🔥 Get base64 image instead of URL
    image_base64 = response.data[0].b64_json

    if not image_base64:
        raise ValueError("Image generation failed. No image data returned.")

    image_bytes = base64.b64decode(image_base64)

    os.makedirs("generated_images", exist_ok=True)
    filename = f"generated_images/image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    with open(filename, "wb") as f:
        f.write(image_bytes)

    print(f"✅ Image saved at: {filename}")

    state["saved_image_path"] = filename

    return state



from langgraph.types import interrupt

def image_feedback(state: GraphState) -> GraphState:
    print("\n🖼 Generated Image:", state.get("saved_image_path", "N/A"))
    print("Are you satisfied with this image?")
    print("Type 'y' for yes OR give feedback to modify it.\n")

    user_input = input("Your response: ")

    state["user_feedback"] = user_input
    return state


def process_feedback(state: GraphState):
    user_feedback = state.get("user_feedback", "")

    if len(user_feedback.strip()) == 0 or user_feedback.lower().strip().startswith("y"):
        return END

    return "generate_prompt"


def check_vector_store_exists(state: GraphState) -> Literal["exists", "not_exists"]:
    faiss_path = "faiss_index"
    if os.path.exists(faiss_path):
        index_files = ["index.faiss", "index.pkl"]
        if all(os.path.exists(os.path.join(faiss_path, f)) for f in index_files):
            print("📂 Vector store exists")
            return "exists"
    print("📂 Vector store does not exist")
    return "not_exists"


def check_files_in_vector_db(state: GraphState) -> Literal["in_db", "not_in_db"]:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings

    faiss_path = "faiss_index"
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

        file_names = [m.get("file_name", "") for m in file_metadata]

        existing_content = []
        for doc in vectorstore.docstore._dict.values():  # type: ignore
            existing_content.append(doc.page_content)

        for fname in file_names:
            found = any(fname in content for content in existing_content)
            if not found:
                print(f"📂 File '{fname}' not in vector DB, running full pipeline")
                return "not_in_db"

        print(
            f"📂 All files exist in vector DB, skipping to retrieval in filename {fname}"
        )
        return "in_db"
    except Exception as e:
        print(f"📂 Error checking vector DB: {e}, running full pipeline")
        return "not_in_db"


# WorkFlow Evalution------------------------------------------------------
def create_all_vector_stores(state: GraphState) -> GraphState:
    print("📦 Creating all vector stores...")
    logger.log_workflow("create_all_vector_stores")

    chunks = state.get("chunks", [])
    embeddings = state.get("embeddings", [])
    images = state.get("images", [])
    file_metadata = state.get("file_metadata", [])
    metadata_docs = state.get("metadata_docs", [])
    strategy_docs = state.get("strategy_docs", [])

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
                        sentences = analysis["sentences"][:10]
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
    brand_data_path = "brand_data.json"
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

    return state


workflow = StateGraph(GraphState)
workflow.add_node("meta", meta_node)
workflow.add_node("ocr", ocr_node)
workflow.add_node("split_by_type", split_by_type_node)
workflow.add_node("image_analyzer", image_analyzer_node)
workflow.add_node("text_split", text_splitter_node)
workflow.add_node("embeddings", embeddings_node)
workflow.add_node("create_all_vector_stores", create_all_vector_stores)
workflow.add_node("Retriever", retriever_node)
workflow.add_node("multi_retriever", multi_retriever_node)
workflow.add_node("chat_node", chat_node)
workflow.add_node("generate_prompt", generate_prompt)
workflow.add_node("generate_image", generate_image)
workflow.add_node("image_feedback", image_feedback)


workflow.add_edge(START, "meta")

workflow.add_conditional_edges(
    "meta",
    check_files_in_vector_db,
    {"in_db": "create_all_vector_stores", "not_in_db": "ocr"},
)

workflow.add_edge("ocr", "split_by_type")

workflow.add_edge("split_by_type", "image_analyzer")
workflow.add_edge("split_by_type", "text_split")

workflow.add_edge("image_analyzer", "embeddings")
workflow.add_edge("text_split", "embeddings")

workflow.add_edge("embeddings", "create_all_vector_stores")
workflow.add_edge("create_all_vector_stores", "multi_retriever")

workflow.add_edge("multi_retriever", "chat_node")
# workflow.add_edge("chat_node", END)
workflow.add_edge("chat_node", "generate_prompt")
workflow.add_edge("generate_prompt", "generate_image")
workflow.add_edge("generate_image", "image_feedback")
workflow.add_conditional_edges(
    "image_feedback",
    process_feedback,
    {
        END: END,
        "generate_prompt": "generate_prompt"
    },
)




# -------------------------------------------------
app = workflow.compile()

# result = app.invoke({
# Usage Examples:
# --------------------------------------------------------------------

# Example 1: Just get chunks (retrieve mode - no LLM call)

result = app.invoke(
    {
        "file_paths":[
            r"C:\Users\rkart\Downloads\Brandlovestudioai\cognixia\1-Cognixia-DEVops 1.pdf",
            r"C:\Users\rkart\Downloads\Brandlovestudioai\cognixia\1-Cognixia-DEVops.pdf",
            r"C:\Users\rkart\Downloads\Brandlovestudioai\cognixia\1-Cognixia-Malware-as-a-service.pdf",
            r"C:\Users\rkart\Downloads\Brandlovestudioai\cognixia\1-Cognixia-SecOps 1.pdf",
            r"C:\Users\rkart\Downloads\Brandlovestudioai\cognixia\2-Benefits-Of-ERGs (1).pdf",
            r"C:\Users\rkart\Downloads\Brandlovestudioai\cognixia\2-Cognixia-Apr-25.jpg",
            r"C:\Users\rkart\Downloads\Brandlovestudioai\cognixia\2-Gen-Ai-Disclosure-act-.jpg",
            r"C:\Users\rkart\Downloads\Brandlovestudioai\cognixia\2-Gen-AI-X-Network-Ops.jpg",
            r"C:\Users\rkart\Downloads\Brandlovestudioai\cognixia\Cognixia _ Digital Strategy and Ideas - Red & Blue Digital.pptx",
            r"C:\Users\rkart\Downloads\Brandlovestudioai\cognixia\US-Cognixia-Logo 1.png",
            r"C:\Users\rkart\Downloads\Brandlovestudioai\cognixia\Website Colors 1.pdf",
            r"C:\Users\rkart\Downloads\Brandlovestudioai\cognixia\Website Font Style 2.pdf",
            r"C:\Users\rkart\Downloads\Brandlovestudioai\cognixia\2-women-in-tech.png"
        ],
        "goal": """
        Generate structured content for a 1-page enterprise blog template about
        "Agentic AI Workflows in Digital Transformation".
        """,
        "question": """
        Extract the core transformation topic and key educational angles.
        Return:
        1. Primary blog theme
        2. 3–5 supporting subtopics
        3. Enterprise pain points
        4. Educational tone direction
        """,
        "question_metadata":  """
        From image metadata extract:
        1. Top 5 dominant background colors with hex
        2. Top 3 text colors used
        3. Accent color suggestions
        4. Layout style pattern (minimal, gradient, split layout, circular mask, etc.)
        5. Typography style hints
        Return structured.
        """,
        "question_strategy": """
        Extract brand communication strategy elements:
        1. Positioning statement
        2. Emotional anchor word
        3. Tone of voice traits
        4. Key messaging structure used in visuals
        5. CTA structure style
        """,
        "question_brand" : """
        Extract structured brand guardrails:
        1. Official color palette with hierarchy (primary, secondary, accent)
        2. Typography family and usage
        3. Tone intensity attributes
        4. Primary and secondary emotions
        5. Persona target
        6. Do/Don’t emotional restrictions
        Return strictly structured.
        """,
        "generation": "",
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
        # ---generate-----
        "prompt": "",
        "image_url": "",
        "user_feedback": "",
    }
)

print("\n🎯 Final Generated Image Path:")
print(result.get("saved_image_path"))

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


# should work for all formats


# -------------------------------------------------------
# Extract Text Colors from Image
# -------------------------------------------------------
def extract_text_colors_from_image(
    image_path: str, output_json: str = None, visualize: bool = False
) -> dict:
    """
    Extract text and their dominant colors from an image.

    Args:
        image_path: Path to the image file
        output_json: Optional path to save JSON output
        visualize: Whether to draw bounding boxes on image

    Returns:
        Dictionary with text and color information
    """
    processor = GoogleVisionOCRProcessor()

    return processor.extract_text_colors(
        image_path=image_path, output_json_path=output_json, visualize=visualize
    )


# Example usage:
# if __name__ == "__main__":
#     result = extract_text_colors_from_image(
#         image_path="sample.png",
#         output_json="text_colors.json",
#         visualize=True
#     )
#     print(json.dumps(result, indent=2))
