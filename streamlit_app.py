import os
import tempfile

import streamlit as st

from app.loaders import load_document
from app.vectorstore import create_vectorstore
from app.graph import build_graph


st.set_page_config(
    page_title="AI Research & Document Assistant",
    page_icon="📚",
    layout="wide"
)


st.title(
    "📚 AI Research & Document Intelligence Assistant"
)

st.write(
    "Upload a document and ask questions about its content."
)


# -------------------------------
# Document Upload
# -------------------------------

st.sidebar.header("📄 Upload Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload PDF, DOCX or TXT",
    type=["pdf", "docx", "txt"]
)


if uploaded_file is not None:

    file_extension = os.path.splitext(
        uploaded_file.name
    )[1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_extension
    ) as temp_file:

        temp_file.write(
            uploaded_file.getvalue()
        )

        temp_path = temp_file.name

    try:

        with st.spinner(
            "Processing document..."
        ):

            documents = load_document(
                temp_path
            )

            create_vectorstore(
                documents
            )

        st.sidebar.success(
            f"Document processed successfully!"
        )

        st.sidebar.write(
            f"Chunks created: {len(documents)}"
        )

    except Exception as e:

        st.sidebar.error(
            f"Error: {str(e)}"
        )


# -------------------------------
# Question
# -------------------------------

question = st.text_input(
    "Ask a question about your document:"
)


if st.button("🔍 Ask"):

    if not os.path.exists("chroma_db"):

        st.warning(
            "Please upload and process a document first."
        )

    elif not question:

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Searching document and generating answer..."
        ):

            graph = build_graph()

            result = graph.invoke(
                {
                    "question": question,
                    "documents": [],
                    "answer": "",
                    "grounded": False
                }
            )

        st.subheader("💡 Answer")

        st.write(
            result["answer"]
        )

        st.divider()

        if result["grounded"]:

            st.success(
                "✅ Grounding check passed"
            )

        else:

            st.warning(
                "⚠️ The answer could not be fully grounded."
            )

        st.subheader(
            "📑 Retrieved Sources"
        )

        for i, document in enumerate(
            result["documents"],
            start=1
        ):

            with st.expander(
                f"Source {i}"
            ):

                st.write(
                    document.page_content
                )

                st.caption(
                    str(document.metadata)
                )