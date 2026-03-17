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
# from typing import List, Dict, NotRequired
from typing import List, Dict
from typing_extensions import NotRequired


import pickle
from pathlib import Path

from langgraph.graph import END, StateGraph, START


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

    file_paths: NotRequired[List[str]]
    brand_assets_files: NotRequired[List[str]]
    creatives_files: NotRequired[List[str]]
    strategy_decks_files: NotRequired[List[str]]
    mode: Literal["retrieve", "chat"]
    question: str
    question_metadata: str
    question_strategy: str
    question_brand: str
    generation: NotRequired[str]   # written by chat_node
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
    # --- image generation loop state ---
    prompt: str
    user_feedback: str
    saved_image_path: str
    goal: str
    image_model: str
    brand_data: NotRequired[Dict]
    # --- intermediate prompt outputs (parallel nodes write these) ---
    prompt_main: NotRequired[str]
    prompt_metadata: NotRequired[str]
    prompt_strategy: NotRequired[str]
    prompt_brand: NotRequired[str]


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

    print(f"  ✅ Resolved {len(resolved_paths)} files from brand folders")

    return {**state, "file_paths": resolved_paths}


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
    return {**state, "file_metadata": file_metadata}

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
# (duplicate preliminary version removed — see _format_brand_data_for_embedding below)


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


# def embeddings_node(state: GraphState) -> GraphState:
#     print("🟣 Running Embeddings Node...")
#     logger.log_workflow("embeddings_node")

#     chunks = state.get("chunks", [])
#     chunks_sources = state.get("chunks_sources", [])
#     images = state.get("images", [])
#     file_metadata = state.get("file_metadata", [])
#     target_db = state.get("target_db", "main")

#     combined_inputs = []
#     combined_sources = []

#     # Check if we are processing brand context (either Niroggi or Brand data)
#     brand_data_path = None

#     if target_db in ["brand", "main"]:
#         brand_data_path = "brand.json"
#     elif target_db == "niroggi":
#         brand_data_path = "Niroggi_data.json"

#     if brand_data_path and os.path.exists(brand_data_path):
#         with open(brand_data_path, "r", encoding="utf-8") as f:
#             brand_data_raw = json.load(f)  # Load the brand data

#         # Format the brand data into an embeddable text block
#         brand_text = _format_brand_data_for_embedding(brand_data_raw)

#         # If the brand context is successfully formatted, append to the input
#         if brand_text:
#             combined_inputs.append(brand_text)
#             combined_sources.append(
#                 {"file": brand_data_path, "chunk_index": 0, "db": "brand"}
#             )
#     else:
#         print(f"⚠️ {brand_data_path} not found, skipping brand embedding.")

#     # Add OCR chunks (text content from files)
#     if chunks:
#         combined_inputs.extend(chunks)
#         combined_sources.extend(chunks_sources)

#     # Process images (if any)
#     if images:
#         for img in images:
#             image_summary = f"Image reference: {img}"
#             combined_inputs.append(image_summary)
#             combined_sources.append({"file": img, "chunk_index": -1})

#     # Ensure we have content to embed
#     if not combined_inputs:
#         raise ValueError("No data found to embed.")

#     # Generate embeddings
#     logger.log_api_call("OpenAI Embeddings", f"Generating embeddings for {target_db}")
#     processor = EmbeddingProcessor(model="text-embedding-ada-002")
#     vectors = processor.generate_embeddings(combined_inputs)

#     # Determine the correct FAISS index based on target_db
#     save_paths = {
#         "niroggi": "brand_faiss_index",  # Use brand_faiss_index for Niroggi data
#         "brand": "brand_data_faiss_index",  # Use a new FAISS index for brand.json
#         "strategy": "strategy_faiss_index",
#         "metadata": "metadata_faiss_index",
#     }
#     save_path = save_paths.get(
#         target_db, "faiss_index"
#     )  # Default to faiss_index if not found

#     # Create or update the FAISS index
#     vectorstore = VectorStoreProcessor(model="text-embedding-ada-002")
#     vectorstore.create_vector_store(
#         chunks=combined_inputs,
#         embeddings=vectors,
#         save_path=save_path,
#     )
#     return {
#         **state,
#         "chunks": combined_inputs,
#         "chunks_sources": combined_sources,
#         "embeddings": vectors,
#     }


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

    # Check if we are processing brand context (either Niroggi or Brand data)
    brand_data_path = None

    if target_db in ["brand", "main"]:
        brand_data_path = "brand.json"
    elif target_db == "faiss":
        brand_data_path = "Niroggi_data.json"

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
        "brand": "brand_data_faiss_index",  # Use a new FAISS index for brand.json
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
    return {
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


def store_brand_data_in_variables(brand_data: dict):
    # Extract brand details from the retrieved brand data
    brand_tone = brand_data.get("brand_tone", {})
    persona = brand_data.get("persona", {})
    dos_and_donts = brand_data.get("dos_and_donts", {})
    brand_mission = brand_data.get("brand_mission", "")
    brand_vision = brand_data.get("brand_vision", "")
    word_bank = brand_data.get("word_bank", {})

    # Store values as variables for placeholders
    brand_tone_str = "\n".join([f"{key}: {value}" for key, value in brand_tone.items()])
    persona_str = f"Name: {persona.get('name', 'N/A')}\nAge: {persona.get('age', 'N/A')}\nGoals: {persona.get('goals', 'N/A')}\nPain Points: {persona.get('pain_points', 'N/A')}"
    dos_str = "\n".join([f"Do: {item}" for item in dos_and_donts.get("dos", [])])
    donts_str = "\n".join([f"Don't: {item}" for item in dos_and_donts.get("donts", [])])

    word_bank_str = f"Positive Word Bank: {', '.join(word_bank.get('positive_word_bank', []))}\nNegative Word Bank: {word_bank.get('negative_word_bank', '')}\nreplaceable_words: {word_bank.get('replaceable_words', '')}"

    # Return the variables so they can be used in the prompt
    return {
        "brand_tone": brand_tone_str,
        "persona": persona_str,
        "dos_and_donts": f"{dos_str}\n{donts_str}",
        "brand_mission": brand_mission,
        "brand_vision": brand_vision,
        "word_bank": word_bank_str,
    }

def load_brand_data_node(state: GraphState) -> GraphState:
    print("🏷 Loading Brand Data...")

    if os.path.exists("brand.json"):
        with open("brand.json", "r", encoding="utf-8") as f:
            brand_data = json.load(f)
        return {"brand_data": brand_data}   # return ONLY changed key

    return {}   # nothing to update



def build_strategy_prompt(template: str, json_path: str, goal: str = ""):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    values = {
        "Brand_Name": data.get("Brand_Name", [""])[0],
        "Brand_Mission": data.get("Brand_Mission", [""])[0],
        "Brand_Vision": data.get("Brand_Vision", [""])[0],
        "Brand_Promise": data.get("Brand_Promise", [""])[0],
        "Market_Positioning": data.get("Market_Positioning", [""])[0],
        "Key_Differentiators": data.get("Key_Differentiators", [""])[0],
        "Audience_Type": data.get("Audience_Type", [""])[0],
        "Primary_Emotion": data.get("Primary_Emotion", ""),
        "Avoided_Emotion": data.get("Avoided_Emotion", ""),
        "Persona_Role": data.get("Persona_Role", ""),
        "Persona_Goals": data.get("Persona_Goals", ""),
        "Fear_And_Pain_Points": data.get("Fear_And_Pain_Points", ""),
        "What_To_Do": " ".join(data.get("What_To_Do", [])),
        "What_Not_To_Do": " ".join(data.get("What_Not_To_Do", [])),
        "goal": goal
    }

    return template.format(**values)


# node function
# def retriever_node(state: GraphState) -> GraphState:
#     print("🟢 Running Retriever Node...")
#     logger.log_workflow("retriever_node")

#     question = state.get("question", "")
#     vectorstore = state.get("vectorstore", None)

#     if not question:
#         raise ValueError("question not found in state.")

#     if not vectorstore:
#         raise ValueError("vectorstore not found. Run vector_store_node first.")

#     print(f"\n🔎 Question: {question}\n")

#     processor = RetrieverProcessor(k=4)  # it was 4 before

#     docs_and_scores = processor.retrieve(question=question, vectorstore=vectorstore)

#     retrieved_docs = []

#     print("📌 Retrieved Chunks With Similarity Scores:\n")

#     for i, (doc, score) in enumerate(docs_and_scores, 1):
#         print(f"\n--- Chunk {i} ---")
#         print(f"Similarity Score: {score}")
#         print(doc.page_content)  # print first 500 chars only
#         retrieved_docs.append(doc.page_content)

#     return {**state, "retrieved_docs": retrieved_docs}


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

    processor = RetrieverProcessor(k=4)  # It was 4 before

    docs_and_scores = processor.retrieve(question=question, vectorstore=vectorstore)

    retrieved_docs = []

    print("📌 Retrieved Chunks With Similarity Scores:\n")

    # Log the content of each retrieved document to verify brand context
    for i, (doc, score) in enumerate(docs_and_scores, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Similarity Score: {score}")
        print(f"Full Content: {doc.page_content}")  # Print full content for debugging
        retrieved_docs.append(doc.page_content)

    return {**state, "retrieved_docs": retrieved_docs}


def multi_retriever_node(state: GraphState) -> GraphState:
    print("🔍 Running Multi-Retriever Node...")
    logger.log_workflow("multi_retriever_node")

    question_main = state.get("question", "")
    question_metadata = state.get("question_metadata", "")
    # question_strategy = state.get("question_strategy", "")
    question_strategy = state.get("strategy_question_filled") or state.get("question_strategy", "")
    question_strategy_template = state.get("question_strategy", "")

    question_strategy = build_strategy_prompt(
        template=question_strategy_template,
        json_path="Niroggi_data.json",
        goal=state.get("goal", "")
    )
    question_brand = state.get("question_brand", "")

    # Do NOT mutate state directly — write to a local var and save to file only.
    with open("strategy_filled_ip.txt", "w", encoding="utf-8") as f:
        f.write(question_strategy)

    print("✅ Saved filled strategy prompt -> strategy_filled_ip.txt")

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
                retrieved_strategy_text = []
                for doc, score in docs_and_scores:
                    results.append(
                        {
                            "content": doc.page_content,
                            "score": score,
                            "db": db_name,
                        }
                    )

                    if db_name == "strategy":
                        retrieved_strategy_text.append(doc.page_content)

                if db_name == "strategy" and retrieved_strategy_text:
                    with open("strategy_retrived.txt", "w", encoding="utf-8") as f:
                        f.write("\n\n--- CHUNK ---\n\n".join(retrieved_strategy_text))

                    print("✅ Saved retrieved strategy chunks -> strategy_retrived.txt")

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
        "strategy_question_filled": question_strategy,
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
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0)

        # 🔵 Define system + user roles explicitly
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Role:
                    Your function is to generate a concise structured synthesis strictly based on the provided context.

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

        # ✅ Blog generation (poster-friendly blog copy)
    # blog_question = (
    #     "Write a short blog for a mobile poster (200-350 words) using ONLY the context. "
    #     "Output in this exact structure:\n"
    #     "1) Title (max 8 words)\n"
    #     "2) 3 short sections with headings (2-3 lines each)\n"
    #     "3) 3 bullet key takeaways\n"
    #     "4) CTA (max 8 words)\n"
    #     "Keep the language crisp and marketing-friendly."
    # )

    # # Use ALL DB answers as blog context (safe + already based on retrieved docs)
    # blog_context = "\n\n".join([f"{k.upper()}:\n{v}" for k, v in answers.items()])
    # blog_text = processor.generate_answer(question=blog_question, context=blog_context)

    return {"generation": str(answers)}


# -----------------------------
# NODE: generate_prompt
# -----------------------------
# =======================================================================================================================
from openai import OpenAI

"""
Four separated generate_prompt functions, each focused on one retrieval source.

State keys written:
  - generate_prompt_main     → state["prompt_main"]
  - generate_prompt_metadata → state["prompt_metadata"]
  - generate_prompt_strategy → state["prompt_strategy"]
  - generate_prompt_brand    → state["prompt_brand"]

Brand placeholders are resolved from state["brand_data"] (a dict).
Brand context is injected into BOTH system_msg and user_content so the model
has full brand knowledge at every layer of the prompt.
"""

import os
from openai import OpenAI


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _flatten(results) -> str:
    """Flatten a list of retrieved doc dicts into a plain string."""
    if isinstance(results, list):
        return "\n".join(
            r.get("content", "") if isinstance(r, dict) else str(r)
            for r in results
        )
    return str(results)


def _b(brand: dict, key: str) -> str:
    """Safe brand value lookup — returns 'MISSING' when absent."""
    return (brand.get(key) or "MISSING").strip()




def _call_openai(system_msg: str, user_content: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_content},
        ],
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 1. generate_prompt_main
# ---------------------------------------------------------------------------

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
        Brand_Name=_b(brand,"Brand_Name"),
        Brand_Mission=_b(brand,"Brand_Mission"),
        Brand_Vision=_b(brand,"Brand_Vision"),
        Brand_Promise=_b(brand,"Brand_Promise"),
        Market_Positioning=_b(brand,"Market_Positioning"),
        Key_Differentiators=_b(brand,"Key_Differentiators"),
        Audience_Type=_b(brand,"Audience_Type"),
        Persona_Role=_b(brand,"Persona_Role"),
        Persona_Goals=_b(brand,"Persona_Goals"),
        Fear_And_Pain_Points=_b(brand,"Fear_And_Pain_Points"),
        Primary_Emotion=_b(brand,"Primary_Emotion"),
        Avoided_Emotion=_b(brand,"Avoided_Emotion"),
        What_To_Do=_b(brand,"What_To_Do"),
        What_Not_To_Do=_b(brand,"What_Not_To_Do"),
        goal=goal
    )

    user_content = USER_GOAL.format(
    retrieved_main=retrieved_main,
    Brand_Name=_b(brand,"Brand_Name"),
    Brand_Mission=_b(brand,"Brand_Mission"),
    Brand_Vision=_b(brand,"Brand_Vision"),
    Brand_Promise=_b(brand,"Brand_Promise"),
    Market_Positioning=_b(brand,"Market_Positioning"),
    Key_Differentiators=_b(brand,"Key_Differentiators"),
    Audience_Type=_b(brand,"Audience_Type"),
    Persona_Role=_b(brand,"Persona_Role"),
    Persona_Goals=_b(brand,"Persona_Goals"),
    Fear_And_Pain_Points=_b(brand,"Fear_And_Pain_Points"),
    Primary_Emotion=_b(brand,"Primary_Emotion"),
    Avoided_Emotion=_b(brand,"Avoided_Emotion"),
    What_To_Do=_b(brand,"What_To_Do"),
    What_Not_To_Do=_b(brand,"What_Not_To_Do"),
    goal=goal
)

    prompt = _call_openai(system_msg, user_content)

    # Return ONLY the key this node writes — avoids concurrent-write conflict in fan-out
    return {"prompt_main": prompt}
 
# ---------------------------------------------------------------------------
# 2. generate_prompt_metadata
# ---------------------------------------------------------------------------

def generate_prompt_metadata(state) -> dict:
    """
    Generates a DALL-E prompt grounded in METADATA / DESIGN documents only.
    Result stored in state["prompt_metadata"].
    """
    print("🎨 Running Generate Prompt (METADATA) Node...")

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
    7. Focus only on visual design rules and layout execution.
    8. Extract layout structure, spacing, typography hierarchy, and color usage rules.
    9. Provide guidance usable by an image generation system.
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
    
    ## RETRIEVED METADATA CONTEXT
    {retrieved_metadata}
    
    ## TASK
    Using only the retrieved main content and the brand context above, extract and synthesize the poster messaging direction.

    """
    brand = state.get("brand_data", {})
    goal = state.get("goal", "")

    retrieved_metadata = _flatten(state.get("retrieved_docs_metadata", []))

    system_msg = SYSTEM_GOAL.format(
    Brand_Name=_b(brand,"Brand_Name"),
    Brand_Mission=_b(brand,"Brand_Mission"),
    Brand_Vision=_b(brand,"Brand_Vision"),
    Brand_Promise=_b(brand,"Brand_Promise"),
    Market_Positioning=_b(brand,"Market_Positioning"),
    Key_Differentiators=_b(brand,"Key_Differentiators"),
    Audience_Type=_b(brand,"Audience_Type"),
    Persona_Role=_b(brand,"Persona_Role"),
    Persona_Goals=_b(brand,"Persona_Goals"),
    Fear_And_Pain_Points=_b(brand,"Fear_And_Pain_Points"),
    Primary_Emotion=_b(brand,"Primary_Emotion"),
    Avoided_Emotion=_b(brand,"Avoided_Emotion"),
    What_To_Do=_b(brand,"What_To_Do"),
    What_Not_To_Do=_b(brand,"What_Not_To_Do"),
    goal=goal
)

    user_content = USER_GOAL.format(
    retrieved_metadata=retrieved_metadata,
    Brand_Name=_b(brand,"Brand_Name"),
    Brand_Mission=_b(brand,"Brand_Mission"),
    Brand_Vision=_b(brand,"Brand_Vision"),
    Brand_Promise=_b(brand,"Brand_Promise"),
    Market_Positioning=_b(brand,"Market_Positioning"),
    Key_Differentiators=_b(brand,"Key_Differentiators"),
    Audience_Type=_b(brand,"Audience_Type"),
    Persona_Role=_b(brand,"Persona_Role"),
    Persona_Goals=_b(brand,"Persona_Goals"),
    Fear_And_Pain_Points=_b(brand,"Fear_And_Pain_Points"),
    Primary_Emotion=_b(brand,"Primary_Emotion"),
    Avoided_Emotion=_b(brand,"Avoided_Emotion"),
    What_To_Do=_b(brand,"What_To_Do"),
    What_Not_To_Do=_b(brand,"What_Not_To_Do"),
    goal=goal
)

    prompt = _call_openai(system_msg, user_content)

    # Return ONLY the key this node writes — avoids concurrent-write conflict in fan-out
    return {"prompt_metadata": prompt}


# ---------------------------------------------------------------------------
# 3. generate_prompt_strategy
# ---------------------------------------------------------------------------

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
    Brand_Name=_b(brand,"Brand_Name"),
    Brand_Mission=_b(brand,"Brand_Mission"),
    Brand_Vision=_b(brand,"Brand_Vision"),
    Brand_Promise=_b(brand,"Brand_Promise"),
    Market_Positioning=_b(brand,"Market_Positioning"),
    Key_Differentiators=_b(brand,"Key_Differentiators"),
    Audience_Type=_b(brand,"Audience_Type"),
    Persona_Role=_b(brand,"Persona_Role"),
    Persona_Goals=_b(brand,"Persona_Goals"),
    Fear_And_Pain_Points=_b(brand,"Fear_And_Pain_Points"),
    Primary_Emotion=_b(brand,"Primary_Emotion"),
    Avoided_Emotion=_b(brand,"Avoided_Emotion"),
    What_To_Do=_b(brand,"What_To_Do"),
    What_Not_To_Do=_b(brand,"What_Not_To_Do"),
    goal=goal
)

    user_content = USER_GOAL.format(
    retrieved_strategy=retrieved_strategy,
    Brand_Name=_b(brand,"Brand_Name"),
    Brand_Mission=_b(brand,"Brand_Mission"),
    Brand_Vision=_b(brand,"Brand_Vision"),
    Brand_Promise=_b(brand,"Brand_Promise"),
    Market_Positioning=_b(brand,"Market_Positioning"),
    Key_Differentiators=_b(brand,"Key_Differentiators"),
    Audience_Type=_b(brand,"Audience_Type"),
    Persona_Role=_b(brand,"Persona_Role"),
    Persona_Goals=_b(brand,"Persona_Goals"),
    Fear_And_Pain_Points=_b(brand,"Fear_And_Pain_Points"),
    Primary_Emotion=_b(brand,"Primary_Emotion"),
    Avoided_Emotion=_b(brand,"Avoided_Emotion"),
    What_To_Do=_b(brand,"What_To_Do"),
    What_Not_To_Do=_b(brand,"What_Not_To_Do"),
    goal=goal
)

    prompt = _call_openai(system_msg, user_content)

    # Return ONLY the key this node writes — avoids concurrent-write conflict in fan-out
    return {"prompt_strategy": prompt}


# ---------------------------------------------------------------------------
# 4. generate_prompt_brand
# ---------------------------------------------------------------------------

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
    11. What The Brand Must Always Communicate
    12. What The Brand Must Never Communicate
    13. What To Do
    14. What Not To Do
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
    Brand_Name=_b(brand,"Brand_Name"),
    Brand_Mission=_b(brand,"Brand_Mission"),
    Brand_Vision=_b(brand,"Brand_Vision"),
    Brand_Promise=_b(brand,"Brand_Promise"),
    Market_Positioning=_b(brand,"Market_Positioning"),
    Key_Differentiators=_b(brand,"Key_Differentiators"),
    Audience_Type=_b(brand,"Audience_Type"),
    Persona_Role=_b(brand,"Persona_Role"),
    Persona_Goals=_b(brand,"Persona_Goals"),
    Fear_And_Pain_Points=_b(brand,"Fear_And_Pain_Points"),
    Primary_Emotion=_b(brand,"Primary_Emotion"),
    Avoided_Emotion=_b(brand,"Avoided_Emotion"),
    What_To_Do=_b(brand,"What_To_Do"),
    What_Not_To_Do=_b(brand,"What_Not_To_Do"),
    goal=goal
    )

    user_content = USER_GOAL.format(
    retrieved_brand=retrieved_brand,
    Brand_Name=_b(brand,"Brand_Name"),
    Brand_Mission=_b(brand,"Brand_Mission"),
    Brand_Vision=_b(brand,"Brand_Vision"),
    Brand_Promise=_b(brand,"Brand_Promise"),
    Market_Positioning=_b(brand,"Market_Positioning"),
    Key_Differentiators=_b(brand,"Key_Differentiators"),
    Audience_Type=_b(brand,"Audience_Type"),
    Persona_Role=_b(brand,"Persona_Role"),
    Persona_Goals=_b(brand,"Persona_Goals"),
    Fear_And_Pain_Points=_b(brand,"Fear_And_Pain_Points"),
    Primary_Emotion=_b(brand,"Primary_Emotion"),
    Avoided_Emotion=_b(brand,"Avoided_Emotion"),
    What_To_Do=_b(brand,"What_To_Do"),
    What_Not_To_Do=_b(brand,"What_Not_To_Do"),
    goal=goal
    )

    prompt = _call_openai(system_msg, user_content)

    # Return ONLY the key this node writes — avoids concurrent-write conflict in fan-out
    return {"prompt_brand": prompt}


# def merge_prompts(state) -> dict:
#     """Merges the 4 individual prompts into state["prompt"] for generate_image."""
#     print("🔀 Running Merge Prompts Node...")
#     merged = "\n\n".join(filter(None, [
#         state.get("prompt_main", ""),
#         state.get("prompt_metadata", ""),
#         state.get("prompt_strategy", ""),
#         state.get("prompt_brand", ""),
#     ]))
#     return {"prompt": merged}

def merge_prompts(state) -> dict:
    print("🔀 Running Merge Prompts Node...")

    merged = f"""
[MAIN MESSAGE]
{state.get("prompt_main", "")}

[VISUAL RULES]
{state.get("prompt_metadata", "")}

[COMMUNICATION STRATEGY]
{state.get("prompt_strategy", "")}

[BRAND GUARDRAILS]
{state.get("prompt_brand", "")}

[FINAL EXECUTION RULES]
- Maintain brand tone and messaging.
- Keep layout clean and readable.
- Avoid cartoon, anime, vector illustration styles.
- Prefer photorealistic healthcare poster style.
- Maintain safe margins for all text.
""".strip()

    return {"prompt": merged}


# def generate_prompt(state: GraphState) -> GraphState:
#     print("🎨 Running Generate Prompt Node...")

#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#     goal = state.get("goal", "")

#     def flatten_results(results):
#         if isinstance(results, list):
#             return "\n".join(
#                 [
#                     r.get("content", "") if isinstance(r, dict) else str(r)
#                     for r in results
#                 ]
#             )

#         return str(results)

#     retrieved_main = flatten_results(state.get("retrieved_docs", {}).get("main", []))
#     retrieved_metadata = flatten_results(state.get("retrieved_docs_metadata", []))
#     retrieved_strategy = flatten_results(state.get("retrieved_docs_strategy", []))
#     retrieved_brand = flatten_results(state.get("retrieved_docs_brand", []))

#     context = f"""
#     MAIN CONTENT:
#     {retrieved_main}

#     METADATA DESIGN:
#     {retrieved_metadata}

#     STRATEGY:
#     {retrieved_strategy}

#     BRAND:
#     {retrieved_brand}
#     """
#     feedback = state.get("user_feedback", "")
#     # goal = (state.get("goal", "") or "").strip()

#     system_msg = """
    
#     You are a senior enterprise visual director and DALL-E 3 production prompt engineer.
#     You generate ultra-detailed, layout-controlled, brand-consistent image prompts
#     for enterprise LinkedIn marketing templates.

#     CRITICAL RULES:
#     1. You MUST translate brand + metadata + strategy into precise visual composition.
#     2. Specify spatial hierarchy (top / mid / bottom / left / right).
#     3. Define background style and depth.
#     4. Define text block placement.
#     5. Define CTA placement.
#     6. Define color ratios (percentage usage).
#     7. Define typography style description (modern sans serif, bold geometric, etc.).
#     8. Define visual tone (minimal, premium, tech-forward, corporate).
#     9. No generic marketing fluff.
#     10. Output ONLY one final DALL-E optimized prompt paragraph.
#     11. Do NOT output explanations.
#     """

#     user_content = f"""

#         PRIMARY USER GOAL (HIGHEST PRIORITY - MUST FOLLOW EXACTLY):
#         {goal}
#         Using the following structured enterprise brand intelligence,
#         generate a production-grade DALL-E 3 prompt for a 1024x1024 poster carousel slide.

#         CONTEXT:
#         {context}
#         CRITICAL REQUIREMENTS:
#         - The visual must strictly reflect brand palette hierarchy
#         - Background must follow dominant color pattern from metadata DB.
#         - Text colors must respect contrast rules from metadata DB.
#         - Layout structure must follow observed layout style patterns.
#         - CTA placement must follow strategy DB.
#         - Tone must match brand emotional intensity.
#         - Typography must follow brand guardrails.
#         - Composition must be spatially defined (top/middle/bottom zones).
#         - Output must be a single ultra-detailed rendering prompt paragraph.
#         Only output the final DALL-E ready prompt.
#         """
#     if feedback:
#         # user_content += f"\n\nModify previous concept according to this feedback: {feedback}"
#         user_content += f"\n\nApply this feedback exactly: {feedback}"
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         temperature=0.2,
#         messages=[
#             {"role": "system", "content": system_msg},
#             {"role": "user", "content": user_content},
#         ],
#     )

#     state["prompt"] = response.choices[0].message.content.strip()
#     return state


# -----------------------------------------------------------------------
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


import re


def get_model_output_path(
    model_name: str, prefix: str = "image", ext: str = "png"
) -> str:
    safe_model_name = re.sub(r"[^a-zA-Z0-9._-]", "_", model_name.strip())
    model_dir = os.path.join("samples_1", safe_model_name)
    os.makedirs(model_dir, exist_ok=True)

    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
    return os.path.join(model_dir, filename)


# -----------------------------
# NODE: generate_image
# -----------------------------
import base64
import os
import requests
from datetime import datetime
from openai import OpenAI


# def generate_image(state: GraphState) -> GraphState:
#     print("🖼 Running Generate Image Node...")

#     client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#     # prompt = (state.get("prompt") or "").strip()
#     prompt = state.get("prompt", "")
#     if not prompt:
#         raise ValueError(
#             "Missing 'prompt' in state. generate_prompt must run before generate_image."
#         )
#     model_name = state.get("image_model") or "gpt-image-1-mini"
#     response = client.images.generate(
#         model=model_name,
#         prompt=prompt,
#         size="1024x1024",
#         n=1,
#         quality="high",
#     )
#     data0 = response.data[0]
#     filename = get_model_output_path(model_name=model_name, prefix="image", ext="png")
#     # ✅ Case 1: base64 returned
#     b64 = getattr(data0, "b64_json", None)
#     if b64:
#         image_bytes = base64.b64decode(b64)
#         with open(filename, "wb") as f:
#             f.write(image_bytes)

#         print(f"✅ Image saved at: {filename}")
#         state["saved_image_path"] = filename
#         state["image_model"] = model_name
#         return state
#     # ✅ Case 2: URL returned (common)
#     url = getattr(data0, "url", None)
#     if url:
#         r = requests.get(url, timeout=60)
#         r.raise_for_status()
#         with open(filename, "wb") as f:
#             f.write(r.content)
#         print(f"✅ Image downloaded & saved at: {filename}")
#         state["saved_image_path"] = filename
#         return state
#     # ✅ Neither returned => log and fail
#     raise ValueError(
#         f"Image generation failed. No b64_json or url returned. Raw: {data0}"
#     )


def generate_image(state: GraphState) -> GraphState:
    print("🖼 Running Generate Image Node...")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    base_prompt = (state.get("prompt") or "").strip()
    if not base_prompt:
        raise ValueError("Missing 'prompt' in state. generate_prompt must run before generate_image.")

    realism_guardrail = """
IMPORTANT VISUAL RULES:
- Create a photorealistic image.
- Do not make it cartoon, anime, vector art, flat illustration, 3D cartoon, or digital painting.
- Use realistic human facial proportions, natural skin texture, realistic lighting, and real photographic depth.
- Keep the result clean and premium, like a real healthcare campaign poster.
""".strip()

    layout_guardrail = """
LAYOUT RULES:
- Design this as a square poster with generous safe margins.
- Keep all text fully visible inside the canvas.
- Leave enough padding from every edge, especially the top edge.
- Do not let the headline touch, cross, or get clipped by the top border.
- Keep the headline in the upper-left area with breathing room.
- Reserve the left side primarily for text and the right side for the subject.
- Keep body text shorter and well spaced.
- Ensure CTA button is fully visible and not too close to the bottom edge.
- Maintain clean visual hierarchy: headline, supporting text, CTA, logo.
""".strip()

    prompt = f"{base_prompt}\n\n{realism_guardrail}\n\n{layout_guardrail}"

    model_name = state.get("image_model") or "gpt-image-1-mini"

    response = client.images.generate(
        model=model_name,
        prompt=prompt,
        size="1024x1024",
        n=1
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

    raise ValueError(f"Image generation failed. No b64_json or url returned. Raw: {data0}")


# -----------------------------
# NODE: image_feedback
# -----------------------------

from langgraph.types import interrupt


def image_feedback(state: GraphState) -> GraphState:
    print("\n🖼 Generated Image:", state.get("saved_image_path", "N/A"))
    print("Are you satisfied with this image?")
    print("Type 'y' for yes OR give feedback to modify it.\n")

    user_input = input("Your response: ")
    state["user_feedback"] = user_input
    return state


# -----------------------------

# NODE: edit_image

# -----------------------------

import base64
import requests


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
    brand_data_path = "Niroggi_data.json"
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


def parse_brand_text_to_dict(text: str) -> dict:
    """Parse text format brand data back to dictionary."""
    result = {}
    for line in text.strip().split("\n"):
        if ":" not in line:
            continue
        key, _, value = line.partition(": ")
        key = key.strip()
        value = value.strip()
        if not key or not value:
            continue
        if value.startswith("{") or value.startswith("["):
            try:
                result[key] = json.loads(value)
            except json.JSONDecodeError:
                result[key] = value
        else:
            result[key] = value
    return result


def retrieve_brand_data() -> dict:
    # Load FAISS index for the brand data
    faiss_index_path = "faiss_index"
    embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.load_local(
        faiss_index_path, embedder, allow_dangerous_deserialization=True
    )

    # Query the FAISS index to retrieve all brand details
    processor = RetrieverProcessor(k=1)
    question = "Retrieve all brand details"
    docs_and_scores = processor.retrieve(question=question, vectorstore=vectorstore)

    # Extract content from the first result
    if docs_and_scores:
        brand_data = docs_and_scores[0][0].page_content
        print(f"DEBUG: Retrieved brand data: {brand_data[:500]}...")
        try:
            return parse_brand_text_to_dict(brand_data)
        except Exception as e:
            print(f"ERROR: Failed to parse brand data: {e}")
            print(f"Raw data: {brand_data}")
            return {}
    else:
        print("No brand data found in the FAISS index.")
        return {}


def generate_prompt_from_faiss(state: GraphState) -> GraphState:
    print("📄 Running Generate Prompt From FAISS Node...")

    # Step 1: Retrieve brand data from FAISS
    brand_data = retrieve_brand_data()

    if not brand_data:
        print("No brand data available to generate prompt.")
        return state

    # Step 2: Store brand data in variables
    brand_variables = store_brand_data_in_variables(brand_data)

    # Step 3: Generate the prompt using brand data variables
    prompt = generate_prompt_with_placeholders(brand_variables)

    # Step 4: Write the generated prompt to a .txt file
    write_prompt_to_txt(prompt, "generated_prompt.txt")

    # Add generated prompt to state for later use
    # state["generated_prompt"] = prompt
    # return state
    return {**state, "generated_prompt": prompt}


workflow = StateGraph(GraphState)

workflow.add_node("resolve_brand_paths", resolve_brand_file_paths)
workflow.add_node("meta", meta_node)
workflow.add_node("ocr", ocr_node)
workflow.add_node("split_by_type", split_by_type_node)
workflow.add_node("image_analyzer", image_analyzer_node)
workflow.add_node("text_split", text_splitter_node)
workflow.add_node("embeddings", embeddings_node)
workflow.add_node("create_all_vector_stores", create_all_vector_stores)
workflow.add_node("generate_prompt_from_faiss", generate_prompt_from_faiss)
workflow.add_node("load_brand_data", load_brand_data_node)  # FIX: was defined but never added
workflow.add_node("multi_retriever", multi_retriever_node)
workflow.add_node("chat_node", chat_node)
workflow.add_node("generate_prompt_main",     generate_prompt_main)
workflow.add_node("generate_prompt_metadata", generate_prompt_metadata)
workflow.add_node("generate_prompt_strategy", generate_prompt_strategy)
workflow.add_node("generate_prompt_brand",    generate_prompt_brand)
workflow.add_node("merge_prompts",            merge_prompts)
workflow.add_node("generate_image", generate_image)
workflow.add_node("image_feedback", image_feedback)
workflow.add_node("edit_image", edit_image)


workflow.add_edge(START, "resolve_brand_paths")
workflow.add_edge("resolve_brand_paths", "meta")
# workflow.add_conditional_edges(
#     "meta",
#     check_files_in_vector_db,
#     {"in_db": "create_all_vector_stores", "not_in_db": "ocr"},
# )
workflow.add_conditional_edges(
    "meta",
    check_files_in_vector_db,
    {"in_db": "load_brand_data", "not_in_db": "ocr"},
)
workflow.add_edge("ocr", "split_by_type")
# Fan-out: text_split feeds embeddings; image_analyzer runs in parallel.
# Both then synchronise into create_all_vector_stores.
workflow.add_edge("split_by_type", "text_split")
workflow.add_edge("split_by_type", "image_analyzer")
workflow.add_edge("text_split", "embeddings")
# image_analyzer result flows directly into create_all_vector_stores (images already collected).
workflow.add_edge("image_analyzer", "create_all_vector_stores")
workflow.add_edge("embeddings", "create_all_vector_stores")
workflow.add_edge("create_all_vector_stores", "generate_prompt_from_faiss")
workflow.add_edge("generate_prompt_from_faiss", "load_brand_data")
workflow.add_edge("load_brand_data", "multi_retriever")
workflow.add_edge("multi_retriever", "chat_node")
# Run all 4 prompt-generation nodes in parallel after chat_node:
workflow.add_edge("chat_node", "generate_prompt_main")
workflow.add_edge("chat_node", "generate_prompt_metadata")
workflow.add_edge("chat_node", "generate_prompt_strategy")
workflow.add_edge("chat_node", "generate_prompt_brand")

# Fan-in: merge all 4 prompts, then generate the image:
workflow.add_edge(
    ["generate_prompt_main", "generate_prompt_metadata",
     "generate_prompt_strategy", "generate_prompt_brand"],
    "merge_prompts"
)
workflow.add_edge("merge_prompts", "generate_image")
workflow.add_edge("generate_image", "image_feedback")

workflow.add_conditional_edges(
    "image_feedback",
    process_feedback,
    {
        END: END,
        "edit_image": "edit_image",
    },
)

workflow.add_edge("edit_image", "image_feedback")
# -------------------------------------------------
app = workflow.compile()
result = app.invoke(
   {
        # Option 1: Use brand folder file names (auto-resolved)
        "brand_assets_files": [
            "LOGO_Niroggi_Log+Tagline_PRUPLE.png",
        ],
        "creatives_files": [
            "2-SM-Post-11.png",
            "01.png",
            "2-SM-Post-11.png",
            "02.png",
            "3-SM-Post-11.png",
            "03.png",
            "4-SM-Post-11.png",
            "04.png",
            "5-SM-Post-11.png",
            "05.png",
            "06.png",
            "Ad-1-300x250.jpg",
            "AD-1-300x1050.jpg",
            "AD-1-320x50.jpg",
            "4-SM-Post-11.png",
        ],
        "strategy_decks_files": [
            "Niroggi - Brand Strategy Routes.pptx",
        ],
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
        "goal": """
        Create an Instagram carousel encouraging mindful eating habits for children.
        """,

        #___________________________________________________________________
        # "goal": """
        # Create a carousel slide explaining the importance of balancing physical activity with mental wellness.
        # Use this as slide 2 in a wellness education series.
        # Format it as a social media carousel slide for Instagram
        # """,
 
        # "goal": """
        
        # """,
        "question": """        
        Return:
        1) Primary theme (1 sentence)
        2) 4–6 supporting sections (each: headline + 1–2 lines)
        3) 5 enterprise pain points caused by neglecting security
        4) 5 practical recommendations (actionable, enterprise-ready)
        5) A Niroggi-style CTA line
        
        and answer this 
        What is the tone of the brand for Niroggi?
        """,
        "question_metadata": """
        From the provided Niroggi creatives (images/PDFs), extract ONLY what is visibly present.
 
        Return:
        1) Top 5 dominant background colors (hex where possible) + brief usage note
        2) Top 3 text colors used (hex where possible)
        3) Accent color suggestions consistent with the creatives (hex where possible)
        4) Key visual labels/themes detected (e.g., icons, shapes, objects, motifs)
        5) Layout patterns observed (e.g., minimal, split layout, gradient, cards, timeline, circular mask)
        6) Typography hints (headline vs body style, weight, spacing, casing) — if unclear, say MISSING
        Rules: Do not guess. If not present, return MISSING.
        """,
            
        "question_strategy": """
        You are a senior brand strategist for {Brand_Name}. Your task is to retrieve and synthesize a comprehensive communication strategy from the brand knowledge base using ONLY the brand context provided below.
        ---

        ## BRAND CONTEXT

        {Brand_Name} is a brand operating with the mission of {Brand_Mission} and a long-term vision of {Brand_Vision}. The brand promises its customers {Brand_Promise} and is positioned in the market as {Market_Positioning}, standing apart through its key differentiators: {Key_Differentiators}. The brand primarily serves a {Audience_Type} audience, with the persona playing the role of {Persona_Role}, driven by goals such as {Persona_Goals} and facing pain points including {Fear_And_Pain_Points}. The current strategic goal guiding this retrieval is {goal}. In all communications, the brand must lead with the emotion of {Primary_Emotion} and must strictly avoid evoking {Avoided_Emotion}. The brand always follows these behavioral principles — {What_To_Do} — and must never engage in the following — {What_Not_To_Do}.

        ---

        ## RETRIEVAL RULES

        1. Retrieve ALL strategy elements strictly aligned with the brand context above.
        2. Every output field MUST reflect the primary_emotion: {Primary_Emotion} in tone and framing.
        3. NEVER include content that triggers the avoided_emotion: {Avoided_Emotion}.
        4. All retrieved content MUST serve the goal: {goal} and align with brand_mission: {Brand_Mission}.
        5. Apply what_to_do: {What_To_Do} as behavioral guardrails — follow these always.
        6. Apply what_not_to_do: {What_Not_To_Do} as hard restrictions — never violate these.
        7. If information for any field is unavailable in the knowledge base or context, return exactly: "MISSING"

        ---

        ## OUTPUT FIELDS TO RETRIEVE

        Using the brand context above as your foundation, retrieve and return ONLY the following 10 fields:

        1. **Key Challenges**
        What are the major obstacles in the industry/market? What does {Brand_Name} face?

        2. **Environmental Influences**
        What external factors are impacting {Brand_Name} or its target audience?

        3. **Target Audience / Target Consumers**
        Who are we speaking to? What are their characteristics, needs, and motivations?

        4. **Solutions Personalizer**
        How does {Brand_Name} personalize its approach or offering to meet individual needs?

        5. **Solutions Humanizer**
        What human elements (empathy, connection) are integrated into {Brand_Name}'s solutions?

        6. **Brand Offers**
        What does {Brand_Name} offer to its customers? What is its core value proposition?

        7. **Brand Strategy**
        What is the overarching brand strategy, including goals, vision, and mission?

        8. **Brand Position**
        How is {Brand_Name} positioned in the market relative to competitors?

        9. **Target Options**
        What are the target segments or growth opportunities identified for {Brand_Name}?

        10. **Challenges**
            What are the internal and external challenges identified for {Brand_Name}?

        ---

        ## OUTPUT FORMAT

        Return ONLY the 10 fields above. No preamble. No explanation. No extra commentary.
        If a field cannot be derived from the provided context or knowledge base, return "MISSING" for that field.

        """,

        "question_brand": """
        Build {Brand_Name} brand guardrails STRICTLY from the provided {Brand_Name} assets. Do not use external knowledge. If any item is not explicitly supported by the assets, write MISSING (do not hallucinate).
 
        Return:
        1) Brand essence + audience (primary/secondary)
        2) Tone keywords + avoid list
        3) Visual style direction (photo vs illustration, minimal vs maximal, modern vs classic)
        4) Color palette with usage rules (primary/secondary/accent + contrast guidance)
        5) Typography hierarchy rules (headline/body/caption) — include font names only if present; otherwise MISSING
        6) Layout system rules (alignment, grid, whitespace, corner radius, icon style)
        7) Mandatory brand assets rules (logo variants, safe area, watermark usage)
        8) Offer/content rules (hero message, CTA, any pricing/offer patterns if present)
        9) Do / Don’t guardrails (emotional + visual + copy)
        10) Recommended template types (3–6) with one rule each (e.g., cover slide, stats slide, checklist slide, quote slide)
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
        "saved_image_path": "",   # FIX: required by GraphState
        "brand_data": {},          # FIX: populated by load_brand_data_node
        "image_model": "gpt-image-1-mini",
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
