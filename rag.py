import os
from dotenv import load_dotenv
load_dotenv()

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

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

#----------------------------------------------------------------------------------------------------------------------

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]
    pdf_path: str
    chunks: List[str]
    embeddings: List[List[float]]   

#----------------------------------------------------------------------------------------------------------------------
#Node1: OCR Node
from ocr_processor import GoogleVisionOCRProcessor
ocr_processor = GoogleVisionOCRProcessor()

def ocr_node(state: GraphState) -> GraphState:
    print("🔵 Running OCR Node...")

    processor = GoogleVisionOCRProcessor()

    extracted_text = processor.extract_text_from_pdf(
        pdf_path=state["pdf_path"],
        user_type="org"
    )

    return {
        **state,
        "documents": [extracted_text]
    }

#---------------------------------------------------------------------------------------------------------------------
# Node2: Text Splitter Node
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

class TextProcessor:
    def preprocess_text(self, documents: List[str]) -> List[str]:
        """Split and clean text documents into chunks"""
        processed_docs = []

        for doc in documents:
            # Clean lines and remove empty ones
            cleaned_text = "\n".join([line.strip() for line in doc.strip().split("\n") if line.strip()])

            # Step 1: Split based on headers
            header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header")])
            header_chunks = header_splitter.split_text(cleaned_text)

            # Step 2: Recursively split header chunks
            for chunk in header_chunks:
                recursive_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=3000,
                    chunk_overlap=500
                )
                # chunk.page_content is used because MarkdownHeaderTextSplitter returns Document objects
                processed_docs.extend(recursive_splitter.split_text(chunk.page_content))

        return processed_docs


# Node function
def text_splitter_node(state: dict) -> dict:
    print("🟢 Running Text Splitter Node...")

    documents = state.get("documents", [])
    processor = TextProcessor()
    chunks = processor.preprocess_text(documents)

    return {
        **state,
        "chunks": chunks
    }

#---------------------------------------------------------------------------------------------------------------------
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
def embeddings_node(state: GraphState) -> GraphState:
    print("🟣 Running Embeddings Node...")

    chunks = state.get("chunks", [])
    if not chunks:
        raise ValueError("chunks not found. Run text_splitter_node before embeddings_node.")

    processor = EmbeddingProcessor(model="text-embedding-ada-002")
    vectors = processor.generate_embeddings(chunks)

    return {
        **state,
        "embeddings": vectors
    }
#---------------------------------------------------------------------------------------------------------------------
#Node4: Vector Store Node











#WorkFlow Evalution
workflow = StateGraph(GraphState)
workflow.add_node("ocr", ocr_node)
workflow.add_node("text_split", text_splitter_node)
workflow.add_node("embeddings", embeddings_node)


workflow.add_edge(START, "ocr")
workflow.add_edge("ocr", "text_split")
workflow.add_edge("text_split", "embeddings")
workflow.add_edge("embeddings", END)



#-------------------------------------------------
app = workflow.compile()

result = app.invoke({
    "question": "What is this document about?",
    "generation": "",
    "documents": [],
    "pdf_path": r"D:\Brandlovestudio\Sailatha_DS (2).pdf"
})

print(app.get_graph().draw_mermaid())



