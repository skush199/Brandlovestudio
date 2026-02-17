import os
import json
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
    images: List[str] 
    embeddings: List[List[float]]
    vectorstore: object
    retrieved_docs: List[str]
  

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
    
    
    
from pathlib import Path
from ocr_processor import GoogleVisionOCRProcessor


def image_analyzer_node(state: GraphState) -> GraphState:
    print("🟠 Running Image Analyzer Node...")

    pdf_path = state["pdf_path"]
    pdf_name = Path(pdf_path).stem
    output_folder = f"extracted_images/{pdf_name}"

    processor = GoogleVisionOCRProcessor()

    image_paths = processor.extract_images_only(
        pdf_path=pdf_path,
        output_folder=output_folder
    )

    return {
    "images": image_paths
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
                    chunk_size=400,#3000,600,400,300
                    chunk_overlap=100,#500,100,80,60
                    separators=["\n\n", "\n", ".", " "] # this is new if working only need to keep it
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

def embeddings_node(state: GraphState) -> GraphState:
    print("🟣 Running Embeddings Node...")

    chunks = state.get("chunks", [])
    images = state.get("images", [])

    combined_inputs = []

    # Add text chunks
    if chunks:
        combined_inputs.extend(chunks)

    # Add image summaries from analysis files
    if images:
        for img in images:
            analysis_path = img.replace(".png", "_analysis.json")

            if os.path.exists(analysis_path):
                with open(analysis_path, "r", encoding="utf-8") as f:
                    analysis = json.load(f)

                summary_parts = []

                # OCR preview text
                if analysis.get("text_preview"):
                    summary_parts.append(analysis["text_preview"])

                # Labels
                if analysis.get("labels"):
                    labels = ", ".join(
                        [l["description"] for l in analysis["labels"] if "description" in l]
                    )
                    if labels:
                        summary_parts.append(f"Labels: {labels}")

                # Best guess labels
                if analysis.get("best_guess_labels"):
                    guess = ", ".join(analysis["best_guess_labels"])
                    if guess:
                        summary_parts.append(f"Best guess: {guess}")

                # Dominant colors
                if analysis.get("dominant_colors"):
                    colors = analysis["dominant_colors"][:3]
                    color_text = ", ".join(
                        [f"{c.get('color_name', '')} ({c['r']},{c['g']},{c['b']})"
                         for c in colors]
                    )
                    if color_text:
                        summary_parts.append(f"Dominant colors: {color_text}")

                image_summary = "\n".join(summary_parts).strip()

                if image_summary:
                    combined_inputs.append(image_summary)
            else:
                # Only fallback if JSON truly missing
                combined_inputs.append(f"Image reference: {img}")

    if not combined_inputs:
        raise ValueError("No data found to embed.")

    processor = EmbeddingProcessor(model="text-embedding-ada-002")
    vectors = processor.generate_embeddings(combined_inputs)

    return {
        **state,
        "chunks": combined_inputs,
        "embeddings": vectors
    }


#---------------------------------------------------------------------------------------------------------------------
#Node4: Vector Store Node (knowledge Base)
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
        save_path: str = "faiss_index"
    ):
        if not chunks:
            raise ValueError("Chunks are empty.")

        if not embeddings:
            raise ValueError("Embeddings are empty.")

        # Create FAISS index using precomputed embeddings
        vectorstore = FAISS.from_embeddings(
            text_embeddings=list(zip(chunks, embeddings)),
            embedding=self.embedder,
            normalize_L2=True

        )

        # Save locally
        vectorstore.save_local(save_path)

        return vectorstore
    
def vector_store_node(state: GraphState) -> GraphState:
    print("🟡 Running Vector Store Node...")

    chunks = state.get("chunks", [])
    embeddings = state.get("embeddings", [])

    if not chunks:
        raise ValueError("chunks not found. Run text_splitter_node before vector_store_node.")

    if not embeddings:
        raise ValueError("embeddings not found. Run embeddings_node before vector_store_node.")

    processor = VectorStoreProcessor(model="text-embedding-ada-002")

    vectorstore = processor.create_vector_store(
        chunks=chunks,
        embeddings=embeddings,
        save_path="faiss_index"
    )

    return {
        **state,
        "vectorstore": vectorstore
    }
    
#---------------------------------------------------------------------------------------------------------------------
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

        docs_and_scores = vectorstore.similarity_search_with_score(
            question,
            k=self.k
        )

        return docs_and_scores

#node function
def retriever_node(state: GraphState) -> GraphState:
    print("🟢 Running Retriever Node...")

    question = state.get("question", "")
    vectorstore = state.get("vectorstore", None)

    if not question:
        raise ValueError("question not found in state.")

    if not vectorstore:
        raise ValueError("vectorstore not found. Run vector_store_node first.")

    print(f"\n🔎 Question: {question}\n")

    processor = RetrieverProcessor(k=4)# it was 4 before

    docs_and_scores = processor.retrieve(
        question=question,
        vectorstore=vectorstore
    )

    retrieved_docs = []

    print("📌 Retrieved Chunks With Similarity Scores:\n")

    for i, (doc, score) in enumerate(docs_and_scores, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Similarity Score: {score}")
        print(doc.page_content)  # print first 500 chars only
        retrieved_docs.append(doc.page_content)

    return {
        **state,
        "retrieved_docs": retrieved_docs
    }
#---------------------------------------------------------------------------------------------------------------------
# Node6: Chat Node (System + User Prompt using gpt-4o)

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


class ChatProcessor:
    def __init__(self, model: str = "gpt-4o"):
        self.llm = ChatOpenAI(
            model=model,
            temperature=0
        )

        # 🔵 Define system + user roles explicitly
        self.prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an intelligent document analysis assistant.

        Instructions:
        - Use ONLY the provided context.
        - You may analyze, synthesize, and evaluate information from the context.
        - Do NOT introduce external knowledge.
        - Base your answer strictly on what is present in the context.
        - If there is insufficient information, clearly state that.

        Your goal is to provide reasoned answers grounded in the document.
        """
    ),
    (
        "user",
        "Context:\n{context}\n\nQuestion:\n{question}"
    )
])


        self.output_parser = StrOutputParser()

        # 🔵 Pre-build chain once (better than rebuilding every call)
        self.chain = self.prompt | self.llm | self.output_parser

    def generate_answer(self, question: str, context: str) -> str:
        return self.chain.invoke({
            "question": question,
            "context": context
        })



# Node function
def chat_node(state: GraphState) -> GraphState:
    print("🔵 Running Chat Node...")

    question = state.get("question", "")
    retrieved_docs = state.get("retrieved_docs", [])

    if not question:
        raise ValueError("question not found in state.")

    if not retrieved_docs:
        raise ValueError("retrieved_docs not found. Run retriever_node first.")

    context = "\n\n".join(retrieved_docs)

    processor = ChatProcessor()

    answer = processor.generate_answer(
        question=question,
        context=context
    )

    print("\n📢 Final Answer:\n")
    print(answer)

    return {
        **state,
        "generation": answer
    }






#WorkFlow Evalution------------------------------------------------------
workflow = StateGraph(GraphState)
workflow.add_node("ocr", ocr_node)
workflow.add_node("image_analyzer", image_analyzer_node)
workflow.add_node("text_split", text_splitter_node)
workflow.add_node("embeddings", embeddings_node)
workflow.add_node("vector_store",vector_store_node)
workflow.add_node("Retriever",retriever_node)
workflow.add_node("chat_node",chat_node)


workflow.add_edge(START, "ocr")
# Parallel split
workflow.add_edge("ocr", "text_split")
workflow.add_edge("ocr", "image_analyzer")


# Merge into embeddings
workflow.add_edge("text_split", "embeddings")
workflow.add_edge("image_analyzer", "embeddings")

workflow.add_edge("embeddings","vector_store")
workflow.add_edge("vector_store","Retriever")
workflow.add_edge("Retriever","chat_node")
workflow.add_edge("chat_node", END)



#-------------------------------------------------
app = workflow.compile()

# result = app.invoke({
#     "question": "What is this document about?",
#     "generation": "",
#     "documents": [],
#     "pdf_path": r"vishruth_ai.pdf"
# })

result = app.invoke({
    "question": "what is the document about",
    "generation": "",
    "documents": [],
    "pdf_path": r"1-Cognixia-SecOps.pdf",
    "chunks": [],
    "images": [],   
    "embeddings": [],
    "vectorstore": None,
    "retrieved_docs": []
})




print(app.get_graph().draw_mermaid())



