from dotenv import load_dotenv

load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from pypdf.errors import PdfReadError
import streamlit as st
from streamlit.typing import UploadedFile

from rag import (
    NO_EVIDENCE_ANSWER,
    build_prompt,
    build_sources,
    build_verification_prompt,
    build_vector_store,
    get_embeddings,
    get_file_hash,
    invalid_page_citations,
    missing_page_citations,
    needs_answer_verification,
    parse_pdf,
    retrieve_documents,
    split_pdf_documents,
)
from storage import (
    clear_conversation,
    create_chat,
    get_chat,
    get_chat_by_hash,
    initialize_storage,
    list_chats,
    load_messages,
    load_vector_store,
    save_message,
    vector_index_exists,
)

st.set_page_config(page_title="PDF Insight")
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        background-color: #141414;
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #f4f4f5;
    }

    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #a1a1aa !important;
    }

    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        min-height: 48px;
        height: auto;
        padding: 10px 12px;
        border: 1px solid #303030;
        border-radius: 11px;
        background: #212121;
        color: #f4f4f5;
        text-align: left;
        justify-content: flex-start !important;
        white-space: normal;
        overflow-wrap: anywhere;
    }

    section[data-testid="stSidebar"] .stButton > button:hover,
    section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] > button {
        border-color: #10a37f;
        background: #303030;
        color: #10a37f;
    }

    section[data-testid="stSidebar"] .stButton > button p {
        width: 100%;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SUGGESTED_QUESTIONS = [
    "Give me a 5-bullet summary of this document.",
    "What are the most important deadlines or dates?",
    "What risks or limitations are mentioned?",
    "Explain the key conclusions in simple language.",
]


def render_citation_warning(answer: str, sources: list[dict]) -> None:
    invalid_pages = invalid_page_citations(answer, sources)
    if invalid_pages:
        pages = ", ".join(str(page) for page in invalid_pages)
        st.warning(f"Could not verify page citation(s): {pages}.")
    elif missing_page_citations(answer):
        st.warning("No page citations could be verified for this answer.")


def queue_document_processing() -> None:
    st.session_state.is_processing = (
        st.session_state.get("file_uploader") is not None
    )


initialize_storage()


@st.cache_resource
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        thinking_level="medium",
        retries=2,
    )


st.session_state.setdefault("vector_db", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("document_uploaded", False)
st.session_state.setdefault("chunk_count", 0)
st.session_state.setdefault("page_count", 0)
st.session_state.setdefault("document_name", "")
st.session_state.setdefault("document_hash", None)
st.session_state.setdefault("pending_question", None)
st.session_state.setdefault("active_source", None)
st.session_state.setdefault("active_chat_id", None)
st.session_state.setdefault("history_load_error", None)
st.session_state.setdefault("is_processing", False)
st.session_state.setdefault("low_text_warning", False)
if (
    st.session_state.document_uploaded
    and st.session_state.vector_db is not None
    and st.session_state.active_chat_id is None
):
    st.session_state.active_chat_id = create_chat(
        st.session_state.document_name or "Document",
        st.session_state.vector_db,
        st.session_state.chunk_count,
        st.session_state.page_count,
        st.session_state.document_hash,
        st.session_state.low_text_warning,
    )


def process_uploaded_document(uploaded_file: UploadedFile, status) -> bool:
    document_hash = get_file_hash(uploaded_file)
    existing_chat = get_chat_by_hash(document_hash)
    reuse_vector_file = None

    if existing_chat and vector_index_exists(existing_chat):
        status.write("Reusing the saved document index...")
        vector_db = load_vector_store(existing_chat, get_embeddings())
        chunk_count = existing_chat["chunk_count"]
        page_count = existing_chat["page_count"]
        low_text_warning = bool(existing_chat["low_text_warning"])
        reuse_vector_file = existing_chat["vector_file"]
    else:
        if existing_chat:
            status.write("The saved index is missing; rebuilding it...")

        status.write("Reading PDF...")
        documents, page_count = parse_pdf(uploaded_file)

        if not documents:
            return False

        extracted_char_count = sum(
            len(document.page_content.strip()) for document in documents
        )
        low_text_warning = extracted_char_count < 500

        status.write("Splitting text into chunks...")
        chunks = split_pdf_documents(documents)
        if not chunks:
            return False

        chunk_count = len(chunks)
        status.write(f"Creating embeddings for {chunk_count} chunks...")
        with st.spinner("Creating document embeddings..."):
            vector_db = build_vector_store(chunks)

    st.session_state.vector_db = vector_db
    st.session_state.document_uploaded = True
    st.session_state.chunk_count = chunk_count
    st.session_state.page_count = page_count
    st.session_state.low_text_warning = low_text_warning
    st.session_state.document_name = uploaded_file.name
    st.session_state.document_hash = document_hash
    st.session_state.active_chat_id = create_chat(
        uploaded_file.name,
        vector_db,
        chunk_count,
        page_count,
        document_hash,
        low_text_warning,
        reuse_vector_file,
    )
    st.session_state.history_load_error = None
    return True


def render_sources(sources, message_index: int) -> None:
    page_sources = {}
    for source in sources:
        page_number = source.get("page", 0) + 1
        page_sources.setdefault(page_number, source)

    st.caption("Sources")
    with st.container(horizontal=True, wrap=True, gap="small"):
        for page_number in page_sources:
            if st.button(
                f"Page {page_number}",
                key=f"source_{message_index}_{page_number}",
            ):
                st.session_state.active_source = (message_index, page_number)

    active_source = st.session_state.active_source
    if active_source and active_source[0] == message_index:
        source = page_sources.get(active_source[1])
        if source:
            content = source.get("content", "").strip()
            preview = content[:300].rstrip()
            if len(content) > 300:
                preview += "..."
            st.caption(f"Page {active_source[1]} excerpt")
            st.write(preview)


def render_message(message, message_index: int) -> None:
    if message["role"] == "user":
        with st.container(horizontal=True, horizontal_alignment="right"):
            with st.container(width="content", border=True):
                st.caption("You")
                st.markdown(message["content"])
        return

    st.caption("Assistant")
    st.markdown(message["content"])
    sources = message.get("sources", [])
    render_citation_warning(message["content"], sources)
    if sources:
        render_sources(sources, message_index)

with st.sidebar:
    st.header("PdfInsight")
    if st.button("New chat", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    saved_chats = list_chats()
    st.subheader("Previous Chats")
    st.caption("Previous chats are saved locally on this computer.")
    if not saved_chats:
        st.caption("No previous chats yet.")
    for chat in saved_chats:
        chat_id = chat["id"]
        chat_title = chat["title"] or chat["document_name"]
        is_active = chat_id == st.session_state.active_chat_id
        if st.button(
            f"Chat {chat_id[:7]}: {chat_title}",
            key=f"history_chat_{chat_id}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            saved_chat = get_chat(chat_id)
            if saved_chat is None:
                st.error("This saved chat could not be found.")
            else:
                st.session_state.messages = load_messages(chat_id)
                st.session_state.active_chat_id = chat_id
                st.session_state.document_name = saved_chat["document_name"]
                st.session_state.chunk_count = saved_chat["chunk_count"]
                st.session_state.page_count = saved_chat["page_count"]
                st.session_state.document_hash = saved_chat["document_hash"]
                st.session_state.low_text_warning = bool(
                    saved_chat["low_text_warning"]
                )
                st.session_state.active_source = None
                st.session_state.pending_question = None
                try:
                    st.session_state.vector_db = load_vector_store(
                        saved_chat,
                        get_embeddings(),
                    )
                    st.session_state.document_uploaded = True
                    st.session_state.history_load_error = None
                except FileNotFoundError:
                    st.session_state.vector_db = None
                    st.session_state.document_uploaded = False
                    st.session_state.history_load_error = (
                        "This chat's saved document index is missing. You can still "
                        "view its messages, but cannot continue it."
                    )

    if st.session_state.document_uploaded:
        st.caption("Document ready")
        st.caption(st.session_state.document_name)
        if st.session_state.page_count:
            st.metric("Pages", st.session_state.page_count)
        st.metric("Chunks indexed", st.session_state.chunk_count)
        st.caption("Answers are grounded in the uploaded document.")
        if st.button(
            "Clear conversation",
            key=f"clear_conversation_{st.session_state.active_chat_id}",
            use_container_width=True,
        ):
            clear_conversation(
                st.session_state.active_chat_id,
                st.session_state.document_name,
            )
            st.session_state.messages = []
            st.session_state.active_source = None
            st.session_state.pending_question = None
            st.rerun()
    elif not st.session_state.history_load_error:
        st.caption("Upload a PDF to begin.")

if st.session_state.history_load_error:
    st.warning(st.session_state.history_load_error)

title_column, document_badge_column = st.columns([3, 1])
with title_column:
    st.title("PdfInsight")
if st.session_state.document_uploaded:
    with document_badge_column:
        with st.container(border=True):
            st.caption(st.session_state.document_name)
            if st.session_state.page_count:
                st.caption(f"{st.session_state.page_count} pages")

st.caption(
    "Ask questions across your PDFs with page-level citations."
)
if st.session_state.low_text_warning:
    st.warning(
        "Very little selectable text was extracted. This PDF may be scanned or "
        "contain incomplete text."
    )

if not st.session_state.document_uploaded:
    with st.container(border=True):
        st.subheader("Start with a document")
        st.markdown(
            "**Best for:** reports, research papers, contracts, manuals, and study notes."
        )
        st.markdown(
            "**Example questions**\n"
            "- What are the main findings?\n"
            "- Summarize section 3.\n"
            "- What deadlines are mentioned?\n"
            "- Compare the two options discussed."
        )

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        key="file_uploader",
        disabled=st.session_state.is_processing,
        on_change=queue_document_processing,
    )
    if st.session_state.is_processing and uploaded_file is None:
        st.session_state.is_processing = False
        st.error("Choose a PDF to start processing.")
    elif st.session_state.is_processing and uploaded_file is not None:
        try:
            with st.status("Processing your document...", expanded=True) as status:
                if process_uploaded_document(uploaded_file, status):
                    status.update(
                        label="Document ready! Ask me anything.", state="complete"
                    )
                    st.rerun()
                else:
                    status.update(
                        label="No extractable text was found in this PDF.",
                        state="error",
                    )
                    st.error(
                        "This PDF has no selectable text. Try a text-based PDF."
                    )
        except PdfReadError:
            st.error("This file could not be read as a PDF. Please choose another file.")
        finally:
            st.session_state.is_processing = False

if (
    st.session_state.document_uploaded
    and st.session_state.vector_db is not None
    and not st.session_state.messages
):
    st.caption("Try asking")
    question_columns = st.columns(2)
    for index, question in enumerate(SUGGESTED_QUESTIONS):
        if question_columns[index % 2].button(
            question,
            key=f"suggested_question_{index}",
        ):
            st.session_state.pending_question = question
            st.rerun()

if st.session_state.messages:
    for message_index, message in enumerate(st.session_state.messages):
        render_message(message, message_index)

if st.session_state.document_uploaded and st.session_state.vector_db is not None:
    query = st.chat_input("Ask a question about the document:")
    if query is None:
        query = st.session_state.pop("pending_question", None)

    if query:
        user_message = {"role": "user", "content": query}
        st.session_state.messages.append(user_message)
        save_message(st.session_state.active_chat_id, user_message)
        with st.container(horizontal=True, horizontal_alignment="right"):
            with st.container(width="content", border=True):
                st.caption("You")
                st.markdown(query)

        documents = retrieve_documents(st.session_state.vector_db, query)
        if not documents:
            assistant_message = {
                "role": "assistant",
                "content": NO_EVIDENCE_ANSWER,
                "sources": [],
            }
            st.session_state.messages.append(assistant_message)
            save_message(st.session_state.active_chat_id, assistant_message)
            st.rerun()

        sources = build_sources(documents)
        prompt = build_prompt(query, documents)

        st.caption("Assistant")
        answer_slot = st.empty()
        streamed_answer = answer_slot.write_stream(
            chunk.text for chunk in get_llm().stream(prompt)
        )
        if not isinstance(streamed_answer, str):
            raise TypeError("The streamed answer was not plain text.")
        answer = streamed_answer

        if needs_answer_verification(answer, sources):
            verification_prompt = build_verification_prompt(
                query,
                answer,
                documents,
            )
            with st.spinner("Checking the answer against the document..."):
                answer = "".join(
                    chunk.text
                    for chunk in get_llm().stream(verification_prompt)
                ).strip()
            if not answer:
                raise RuntimeError("The answer verification request returned no text.")
            answer_slot.markdown(answer)

        render_citation_warning(answer, sources)
        assistant_index = len(st.session_state.messages)
        render_sources(sources, assistant_index)

        assistant_message = {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
        st.session_state.messages.append(assistant_message)
        save_message(st.session_state.active_chat_id, assistant_message)