from __future__ import annotations

import os
import json
import time
import functools
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Literal, Annotated
from typing_extensions import TypedDict
from operator import add

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS"
)

class ImageDescriptionStore:
    def __init__(self, store_path: str = "image_descriptions.json"):
        self.store_path = store_path

    def load(self) -> dict:
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"  ⚠️ Error loading image descriptions: {e}")
        return {}

    def save(self, descriptions: dict) -> None:
        existing = self.load()
        existing.update(descriptions)
        try:
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
            print(
                f"  💾 Saved {len(descriptions)} image descriptions to {self.store_path}"
            )
        except IOError as e:
            print(f"  ⚠️ Error saving image descriptions: {e}")

    def get_descriptions_for_images(self, image_paths: list) -> dict:
        all_descriptions = self.load()
        return {
            path: all_descriptions.get(path)
            for path in image_paths
            if path in all_descriptions
        }

image_desc_store = ImageDescriptionStore()

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

def dict_merge(x: dict, y: dict) -> dict:
    """Merge two dictionaries (y overwrites x)."""
    if x is None:
        return y
    if y is None:
        return x
    result = dict(x)
    result.update(y)
    return result

def list_merge(x: list, y: list) -> list:
    """Merge two lists (extend)."""
    if x is None:
        return y if y else []
    if y is None:
        return x
    return list(x) + list(y)

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

    # file_paths: NotRequired[List[str]]
    file_paths: Annotated[List[str], add]
    # brand_assets_files: NotRequired[List[str]]
    # creatives_files: NotRequired[List[str]]
    # strategy_decks_files: NotRequired[List[str]]
    brand_assets_files: Annotated[List[str], add]
    creatives_files: Annotated[List[str], add]
    strategy_decks_files: Annotated[List[str], add]
    mode: Literal["retrieve", "chat"]
    question: Annotated[str, lambda x, y: y]
    question_metadata: Annotated[str, lambda x, y: y]
    question_strategy: Annotated[str, lambda x, y: y]
    question_brand: Annotated[str, lambda x, y: y]
    # generation: NotRequired[str]  # written by chat_node
    generation: Annotated[str, lambda x, y: y] 
    db_answers: Annotated[Dict[str, str], dict_merge]
    # blog_text: NotRequired[str]
    blog_text: Annotated[str, lambda x, y: y]
    # blog_summary: NotRequired[str]
    blog_summary: Annotated[str, lambda x, y: y]
    documents: Annotated[List[Dict], lambda x, y: y]
    # chunks: List[str]
    chunks: Annotated[List[str], add]
    # chunks_sources: List[Dict]
    chunks_sources: Annotated[List[Dict], add]
    # images: List[str]
    images: Annotated[List[str], add]
    # embeddings: List[List[float]]
    embeddings: Annotated[List[List[float]], add]
    vectorstore: Annotated[object, lambda x, y: y]
    # retrieved_docs: Dict  # FIX: Use dict_merge for dict combination
    retrieved_docs: Annotated[Dict, dict_merge]
    # retrieved_docs: Annotated[List, add]
    retrieved_docs_metadata: Annotated[List[Dict], list_merge]
    retrieved_docs_strategy: Annotated[List[Dict], list_merge]
    retrieved_docs_brand: Annotated[List[Dict], list_merge]
    file_metadata: Annotated[List[Dict], list_merge]
    # target_db: str
    target_db: Annotated[str, lambda x, y: y]
    # metadata_docs: List[Dict]
    metadata_docs: Annotated[List[Dict], list_merge]
    # strategy_docs: List[Dict]
    strategy_docs: Annotated[List[Dict], list_merge]
    files_to_ocr: Annotated[List[str], add]
    # --- image generation loop state ---
    prompt: Annotated[str, lambda x, y: y]
    user_feedback: Annotated[str, lambda x, y: y]
    saved_image_path: Annotated[str, lambda x, y: y]
    goal: Annotated[str, lambda x, y: y]
    image_model: Annotated[str, lambda x, y: y]
    brand_data: Annotated[Dict, dict_merge]
    # --- intermediate prompt outputs (parallel nodes write these) ---
    prompt_main: Annotated[str, lambda x, y: y]
    prompt_metadata: Annotated[str, lambda x, y: y]
    prompt_strategy: Annotated[str, lambda x, y: y]
    prompt_brand: Annotated[str, lambda x, y: y]
    # --- image descriptions from GPT-4o vision ---
    image_descriptions: Annotated[Dict[str, str], dict_merge]
