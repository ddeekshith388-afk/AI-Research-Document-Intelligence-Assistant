from typing import TypedDict, List, Any

from langgraph.graph import StateGraph, END

from app.rag import (
    retrieve_documents,
    generate_answer
)


# =========================================================
# GRAPH STATE
# =========================================================

class GraphState(TypedDict, total=False):
    question: str
    documents: List[Any]
    answer: str
    grounded: bool


# =========================================================
# RETRIEVE NODE
# =========================================================

def retrieve_node(state: GraphState):

    question = state["question"]

    from app.vectorstore import load_vectorstore

    vectorstore = load_vectorstore()

    documents = retrieve_documents(
        vectorstore,
        question
    )

    if documents is None:
        documents = []

    elif not isinstance(documents, list):
        documents = list(documents)

    return {
        "documents": documents
    }


# =========================================================
# GENERATE NODE
# =========================================================

def generate_node(state: GraphState):

    question = state["question"]

    documents = state.get(
        "documents",
        []
    )

    answer, _ = generate_answer(
        question,
        documents
    )

    # Make sure answer is a string
    if answer is None:
        answer = ""

    elif not isinstance(answer, str):
        answer = str(answer)

    answer = answer.strip()

    return {
        "answer": answer
    }


# =========================================================
# GROUNDING CHECK NODE
# =========================================================

def grounding_check_node(state: GraphState):

    answer = state.get(
        "answer",
        ""
    )

    documents = state.get(
        "documents",
        []
    )

    # -----------------------------------------------------
    # Make sure answer is a string
    # -----------------------------------------------------

    if answer is None:
        answer = ""

    elif not isinstance(answer, str):
        answer = str(answer)

    answer = answer.strip()

    # -----------------------------------------------------
    # Build context from documents
    # -----------------------------------------------------

    context_parts = []

    for doc in documents:

        if hasattr(doc, "page_content"):

            content = doc.page_content

            if content:
                context_parts.append(
                    str(content)
                )

        elif isinstance(doc, dict):

            content = doc.get(
                "page_content",
                ""
            )

            if content:
                context_parts.append(
                    str(content)
                )

        else:

            if doc:
                context_parts.append(
                    str(doc)
                )

    context = "\n".join(
        context_parts
    ).strip()

    # -----------------------------------------------------
    # Convert to lowercase
    # -----------------------------------------------------

    context_lower = context.lower()

    answer_lower = answer.lower()

    # -----------------------------------------------------
    # Grounding check
    # -----------------------------------------------------

    grounded = True

    # No answer
    if not answer:
        grounded = False

    # No retrieved context
    if not context:
        grounded = False

    # -----------------------------------------------------
    # Check if Gemini says information was not found
    # -----------------------------------------------------

    not_found_phrases = [
        "i could not find this information",
        "i couldn't find this information",
        "could not find this information",
        "couldn't find this information",
        "information is not available",
        "not enough information",
        "i don't have enough information",
        "i do not have enough information",
        "not found in the documents",
        "not mentioned in the documents",
        "cannot find this information"
    ]

    for phrase in not_found_phrases:

        if phrase in answer_lower:

            grounded = False
            break

    return {
        "answer": answer,
        "grounded": grounded
    }


# =========================================================
# BUILD GRAPH
# =========================================================

def build_graph():

    workflow = StateGraph(
        GraphState
    )

    # -----------------------------------------------------
    # Add nodes
    # -----------------------------------------------------

    workflow.add_node(
        "retrieve",
        retrieve_node
    )

    workflow.add_node(
        "generate",
        generate_node
    )

    workflow.add_node(
        "grounding_check",
        grounding_check_node
    )

    # -----------------------------------------------------
    # Entry point
    # -----------------------------------------------------

    workflow.set_entry_point(
        "retrieve"
    )

    # -----------------------------------------------------
    # Edges
    # -----------------------------------------------------

    workflow.add_edge(
        "retrieve",
        "generate"
    )

    workflow.add_edge(
        "generate",
        "grounding_check"
    )

    workflow.add_edge(
        "grounding_check",
        END
    )

    # -----------------------------------------------------
    # Compile
    # -----------------------------------------------------

    return workflow.compile()