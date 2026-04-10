from __future__ import annotations

import os
import json

from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from nodes.common import GraphState, logger
from nodes.prompts import generate_prompt_with_placeholders, write_prompt_to_txt

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
 
    brand_data = {}
 
    # Load from Jiraaf_data.json (has all brand fields)
    if os.path.exists("Jiraaf_data.json"):
        with open("Jiraaf_data.json", "r", encoding="utf-8") as f:
            brand_data = json.load(f)
        print("  ✅ Loaded from Jiraaf_data.json")
 
    if brand_data:
        return {**state, "brand_data": brand_data}
 
    return state

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
        "goal": goal,
    }

    return template.format(**values)

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

    return {"retrieved_docs": retrieved_docs}

def multi_retriever_node(state: GraphState) -> GraphState:
    print("🔍 Running Multi-Retriever Node...")
    logger.log_workflow("multi_retriever_node")

    question_main = state.get("question", "")
    question_metadata = state.get("question_metadata", "")
    question_brand = state.get("question_brand", "")

    brand_name_raw = (
        state.get("brand_name")
        or state.get("brand_data", {}).get("Brand_Name")
        or ""
    )
    if isinstance(brand_name_raw, list):
        brand_name = brand_name_raw[0] if brand_name_raw else ""
    else:
        brand_name = str(brand_name_raw).strip()

    question_strategy_template = state.get("question_strategy", "").strip()

    # Dynamic but focused semantic retrieval query
    strategy_focus = (
        "brand identity tone of voice personality traits target audience "
        "communication style platform strategy content approach messaging themes "
        "brand guardrails compliance restrictions marketing objectives"
    )

    question_strategy = (
        f"{brand_name} {question_strategy_template} {strategy_focus}".strip()
        if question_strategy_template
        else f"{brand_name} {strategy_focus}".strip()
    )

    # Full filled prompt for downstream LLM step
    question_strategy_filled = build_strategy_prompt(
        template=question_strategy_template or "{Brand_Name} {goal}",
        json_path="Jiraaf_data.json",
        goal=state.get("goal", ""),
    )

    with open("strategy_filled_ip.txt", "w", encoding="utf-8") as f:
        f.write("=== RETRIEVAL QUERY ===\n")
        f.write(question_strategy + "\n\n")
        f.write("=== FILLED PROMPT ===\n")
        f.write(question_strategy_filled)

    print("✅ Saved filled strategy prompt -> strategy_filled_ip.txt")

    embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
    processor = RetrieverProcessor(k=10)

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

                if db_name == "strategy":
                    strategy_queries = [
                        question_strategy,
                        f"{brand_name} brand identity tone voice personality traits",
                        f"{brand_name} target audience behavior motivations pain points",
                        f"{brand_name} positioning differentiation value proposition",
                        f"{brand_name} social media platform content strategy",
                        f"{brand_name} brand guardrails compliance dos donts restrictions",
                        f"{brand_name} marketing content objectives messaging themes",
                    ]

                    docs_and_scores = []
                    for q in strategy_queries:
                        docs_and_scores.extend(
                            processor.retrieve(question=q, vectorstore=vectorstore)
                        )

                    # Deduplicate retrieved chunks
                    seen = set()
                    deduped_docs_and_scores = []
                    for doc, score in docs_and_scores:
                        content = doc.page_content.strip()
                        if content not in seen:
                            seen.add(content)
                            deduped_docs_and_scores.append((doc, score))
                    docs_and_scores = deduped_docs_and_scores
                else:
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

                # Brand-specific filtering for strategy DB
                if db_name == "strategy" and brand_name:
                    filtered = [
                        r for r in results
                        if brand_name.lower() in r["content"].lower()
                    ]
                    results = filtered if filtered else results
                    print(f"  🏷 Brand filter '{brand_name}': {len(results)} chunks kept")

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
        "strategy_question_filled": question_strategy_filled,
        "retrieved_docs": all_results,
        "retrieved_docs_metadata": metadata_results,
        "retrieved_docs_strategy": strategy_results,
        "retrieved_docs_brand": brand_results,
    }

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
                # ("user", "Context:\n{context}\n\nQuestion:\n{question}"),
                (
                    "user",
                    "Goal:\n{goal}\n\nContext:\n{context}\n\nQuestion:\n{question}",
                ),
            ]
        )

        self.output_parser = StrOutputParser()

        # 🔵 Pre-build chain once (better than rebuilding every call)
        self.chain = self.prompt | self.llm | self.output_parser

    # def generate_answer(self, question: str, context: str) -> str:
    #     return self.chain.invoke({"question": question, "context": context})
    def generate_answer(self, question: str, context: str, goal: str) -> str:
        return self.chain.invoke(
            {
                "question": question,
                "context": context,
                "goal": goal,
            }
        )

def chat_node(state: GraphState) -> GraphState:
    print("🔵 Running Chat Node...")
    logger.log_workflow("chat_node")
    logger.log_api_call("OpenAI Chat", "Generating responses")

    goal = state.get("goal", "")
    question = state.get("question", "")
    question_metadata = state.get("question_metadata", "")
    # AFTER (correct — uses the filled version built in multi_retriever_node)
    question_strategy = state.get("strategy_question_filled") or state.get("question_strategy", "")
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
            # answer = processor.generate_answer(question=db_question, context=context)
            answer = processor.generate_answer(
                question=db_question,
                context=context,
                goal=goal,
            )
            answers[db_name] = answer

            print(f"\n📢 {db_name.upper()} Answer:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
    else:
        if question:
            context = "\n\n".join(retrieved_docs)
            # answers["main"] = processor.generate_answer(
            #     question=question, context=context
            # )
            answers["main"] = processor.generate_answer(
                question=question,
                context=context,
                goal=goal,
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

    # Save db_answers to file for persistence
    try:
        with open("db_answers.json", "w", encoding="utf-8") as f:
            json.dump(answers, f, ensure_ascii=False, indent=2)
        print(f"  💾 Saved db_answers to db_answers.json")
    except Exception as e:
        print(f"  ⚠️ Error saving db_answers: {e}")

    return {
        "generation": json.dumps(answers, ensure_ascii=False, indent=2),
        "db_answers": answers,
    }

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
    return {"generated_prompt": prompt}
