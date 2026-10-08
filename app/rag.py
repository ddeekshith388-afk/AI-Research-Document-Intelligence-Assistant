from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from app.config import GEMINI_MODEL


# =========================================================
# GEMINI MODEL
# =========================================================

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0
)


# =========================================================
# PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_template(
    """
You are an AI Research and Document Intelligence Assistant.

Answer the user's question ONLY using the provided document context.

If the answer is not present in the context, clearly say:

"I could not find this information in the uploaded document."

Do not invent facts.

Context:
{context}

Question:
{question}

Answer:
"""
)


# =========================================================
# RETRIEVE DOCUMENTS
# =========================================================

def retrieve_documents(vectorstore, question):

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    documents = retriever.invoke(question)

    return documents


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(question, documents):

    # -----------------------------------------------------
    # Create context from retrieved documents
    # -----------------------------------------------------

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # -----------------------------------------------------
    # Create prompt messages
    # -----------------------------------------------------

    messages = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    # -----------------------------------------------------
    # Call Gemini
    # -----------------------------------------------------

    response = llm.invoke(messages)

    # -----------------------------------------------------
    # Extract Gemini response content
    # -----------------------------------------------------

    content = response.content

    # Normal string response
    if isinstance(content, str):

        answer = content

    # Gemini may return a list of content blocks
    elif isinstance(content, list):

        text_parts = []

        for item in content:

            # Example:
            # {
            #   "type": "text",
            #   "text": "Mr. Ronald",
            #   "extras": {...}
            # }

            if isinstance(item, dict):

                if item.get("type") == "text":

                    text_parts.append(
                        item.get("text", "")
                    )

            elif isinstance(item, str):

                text_parts.append(item)

        answer = "".join(text_parts)

    # Dictionary response
    elif isinstance(content, dict):

        answer = content.get(
            "text",
            ""
        )

    # Any other unexpected format
    else:

        answer = str(content)

    # -----------------------------------------------------
    # Clean answer
    # -----------------------------------------------------

    answer = answer.strip()

    # -----------------------------------------------------
    # Return answer + documents
    # -----------------------------------------------------

    return answer, documents