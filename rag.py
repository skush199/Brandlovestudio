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
from typing import List


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

#----------------------------------------------------------------------------------------------------------------------
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












#WorkFlow Evalution
workflow = StateGraph(GraphState)
workflow.add_node("ocr", ocr_node)

workflow.add_edge(START, "ocr")
workflow.add_edge("ocr", END)

app = workflow.compile()

result = app.invoke({
    "question": "What is this document about?",
    "generation": "",
    "documents": [],
    "pdf_path": r"D:\Brandlovestudio\Sailatha_DS (2).pdf"
})

print(app.get_graph().draw_mermaid())



