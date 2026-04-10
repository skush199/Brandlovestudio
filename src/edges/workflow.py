from langgraph.graph import END, START, StateGraph

from nodes.common import GraphState
from nodes.ingestion import (
    resolve_brand_file_paths,
    filter_files_to_ocr,
    meta_node,
    ocr_node,
    split_by_type_node,
)
from nodes.processing import (
    image_analyzer_node,
    generate_image_descriptions_node,
    text_splitter_node,
    embeddings_node,
    create_all_vector_stores,
)
from nodes.retrieval import (
    load_brand_data_node,
    multi_retriever_node,
    chat_node,
    generate_prompt_from_faiss,
)
from nodes.prompts import (
    generate_prompt_main,
    generate_prompt_metadata,
    generate_prompt_strategy,
    generate_prompt_brand,
    merge_prompts,
)
from nodes.generation import (
    generate_blog_node,
    generate_image,
    image_feedback,
    edit_image,
)
from edges.conditions import check_files_in_vector_db, process_feedback


def build_workflow():
    workflow = StateGraph(GraphState)

    workflow.add_node("resolve_brand_paths", resolve_brand_file_paths)
    workflow.add_node("meta", meta_node)
    workflow.add_node("filter_files", filter_files_to_ocr)
    workflow.add_node("ocr", ocr_node)
    workflow.add_node("split_by_type", split_by_type_node)
    workflow.add_node("image_analyzer", image_analyzer_node)
    workflow.add_node("generate_image_descriptions", generate_image_descriptions_node)
    workflow.add_node("text_split", text_splitter_node)
    workflow.add_node("embeddings", embeddings_node)
    workflow.add_node("create_all_vector_stores", create_all_vector_stores)
    workflow.add_node("generate_prompt_from_faiss", generate_prompt_from_faiss)
    workflow.add_node("load_brand_data", load_brand_data_node)
    workflow.add_node("multi_retriever", multi_retriever_node)
    workflow.add_node("chat_node", chat_node)
    workflow.add_node("generate_prompt_main", generate_prompt_main)
    workflow.add_node("generate_prompt_metadata", generate_prompt_metadata)
    workflow.add_node("generate_prompt_strategy", generate_prompt_strategy)
    workflow.add_node("generate_prompt_brand", generate_prompt_brand)
    workflow.add_node("merge_prompts", merge_prompts)
    workflow.add_node("generate_blog", generate_blog_node)
    workflow.add_node("generate_image", generate_image)
    workflow.add_node("image_feedback", image_feedback)
    workflow.add_node("edit_image", edit_image)

    workflow.add_edge(START, "resolve_brand_paths")
    workflow.add_edge("resolve_brand_paths", "meta")
    workflow.add_conditional_edges(
        "meta",
        check_files_in_vector_db,
        {"in_db": "create_all_vector_stores", "not_in_db": "filter_files"},
    )
    workflow.add_edge("filter_files", "ocr")
    workflow.add_edge("ocr", "split_by_type")
    workflow.add_edge("split_by_type", "text_split")
    workflow.add_edge("split_by_type", "image_analyzer")
    workflow.add_edge("image_analyzer", "generate_image_descriptions")
    workflow.add_edge("text_split", "embeddings")
    workflow.add_edge("generate_image_descriptions", "embeddings")
    workflow.add_edge("embeddings", "create_all_vector_stores")
    workflow.add_edge("create_all_vector_stores", "generate_prompt_from_faiss")
    workflow.add_edge("generate_prompt_from_faiss", "load_brand_data")
    workflow.add_edge("load_brand_data", "multi_retriever")
    workflow.add_edge("multi_retriever", "chat_node")
    workflow.add_edge("chat_node", "generate_prompt_main")
    workflow.add_edge("chat_node", "generate_prompt_metadata")
    workflow.add_edge("chat_node", "generate_prompt_strategy")
    workflow.add_edge("chat_node", "generate_prompt_brand")
    workflow.add_edge(
        [
            "generate_prompt_main",
            "generate_prompt_metadata",
            "generate_prompt_strategy",
            "generate_prompt_brand",
        ],
        "merge_prompts",
    )
    workflow.add_edge("merge_prompts", "generate_blog")
    workflow.add_edge("generate_blog", "generate_image")
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

    return workflow.compile()


app = build_workflow()
