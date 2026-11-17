import html
import logging
import os
import uuid

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
    get_usage_today,
    initialize_storage,
    list_chats,
    load_messages,
    load_vector_store,
    record_usage,
    save_message,
    vector_index_exists,
)

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_MODEL_LABEL = "Gemini 3.5 Flash-Lite"
THINKING_LEVEL = "medium"
# Daily limits (reset at 00:00 UTC) keep the public demo within the free tier.
DAILY_QUESTION_LIMIT = 20  # per visitor
GLOBAL_DAILY_QUESTION_LIMIT = 500  # across all visitors

st.set_page_config(page_title="Lumenfold")

# Semantic colours per mode (base colours live in .streamlit/config.toml).
# Teal = act / AI / verified · amber = review · red = risk or unsupported.
PALETTES = {
    "dark": {
        "accent": "#C9A227",
        "accent_hover": "#E3BC3A",
        "elevated": "#16181D",
        "muted": "#A6ABB4",
        "info": "#7FA6C9",
        # Gold is too close to the amber warning, so the ring starts neutral.
        "ring": "#A6ABB4",
        "warning": "#D9A441",
        "error": "#E06A5A",
        "evidence": "rgba(201, 162, 39, 0.16)",
        "evidence_edge": "#C9A227",
    },
    "light": {
        "accent": "#0F766E",
        "accent_hover": "#115E59",
        "elevated": "#F1F5F9",
        "muted": "#64748B",
        "info": "#0F766E",
        "ring": "#0F766E",
        "warning": "#B45309",
        "error": "#B91C1C",
        "evidence": "#FEF3C7",
        "evidence_edge": "#D97706",
    },
}
COLORS = PALETTES["light" if st.context.theme.type == "light" else "dark"]

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,500&display=swap');

    /* Sidebar chat list: left-aligned, wrapping rows. Colours come from the theme. */
    section[data-testid="stSidebar"] .stButton > button {{
        width: 100%;
        min-height: 48px;
        height: auto;
        padding: 10px 12px;
        text-align: left;
        justify-content: flex-start !important;
        white-space: normal;
        overflow-wrap: anywhere;
        box-shadow: none;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        background: {COLORS["elevated"]};
        border-color: {COLORS["accent_hover"]};
        color: {COLORS["accent_hover"]};
    }}
    section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {{
        background: {COLORS["elevated"]};
        border-color: {COLORS["accent"]};
        color: {COLORS["accent"]};
    }}
    section[data-testid="stSidebar"] .stButton > button p {{
        width: 100%;
        text-align: left;
    }}

    /* Evidence excerpt: quiet, serif, highlighted like marked-up PDF text */
    .evidence {{
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 15.5px;
        line-height: 1.6;
        padding: 12px 16px;
        border-left: 3px solid {COLORS["evidence_edge"]};
        border-radius: 0 12px 12px 0;
        background: {COLORS["evidence"]};
    }}

    /* Claude-style status row under the chat box: model · usage ring */
    .usage-row {{
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 14px;
        padding: 2px 6px 0;
        font-size: 13px;
        line-height: 1;
        user-select: none;
    }}
    .usage-row .model {{ color: inherit; font-weight: 500; }}
    .usage-row .muted {{ color: {COLORS["muted"]}; }}
    .usage-row .ring {{
        display: inline-flex;
        align-items: center;
        gap: 7px;
        cursor: default;
    }}
    /* Progress ring: a conic gradient with the centre masked out */
    .usage-row .dial {{
        width: 16px;
        height: 16px;
        border-radius: 50%;
        -webkit-mask: radial-gradient(circle, transparent 5px, #000 5.5px);
        mask: radial-gradient(circle, transparent 5px, #000 5.5px);
    }}
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


def api_status_code(error: BaseException) -> int | None:
    """Find the HTTP status code of a Gemini API error, if there is one."""
    current: BaseException | None = error
    while current is not None:
        code = getattr(current, "code", None)
        if isinstance(code, int):
            return code
        current = current.__cause__
    return None


def friendly_error_message(error: Exception) -> str:
    """Turn a Gemini or runtime failure into a short message users can act on."""
    code = api_status_code(error)
    details = str(error).lower()
    if code == 429 or "resource_exhausted" in details or "quota" in details:
        return (
            "The Gemini API usage limit has been reached. "
            "Please wait a minute and try again, or try again later today."
        )
    if code in (401, 403) or "api key" in details or "api_key" in details:
        return (
            "The Gemini API key is missing or invalid. "
            "Check the GEMINI_API_KEY setting."
        )
    if (code is not None and code >= 500) or "timeout" in details:
        return "Gemini is not responding right now. Please try again in a moment."
    return "Something went wrong while answering. Please try again."


def record_exchange(user_message: dict, assistant_message: dict) -> None:
    """Save a question and its answer together, only once both exist."""
    for message in (user_message, assistant_message):
        st.session_state.messages.append(message)
        save_message(
            st.session_state.active_chat_id,
            message,
            owner_id=st.session_state.chat_owner_id,
        )


def format_tokens(count: int) -> str:
    return f"{count / 1000:.1f}k" if count >= 1000 else str(count)


def usage_limit_reached(usage: dict) -> bool:
    return (
        usage["questions"] >= DAILY_QUESTION_LIMIT
        or usage["all_questions"] >= GLOBAL_DAILY_QUESTION_LIMIT
    )


def render_usage_row(usage: dict) -> None:
    """Small status row like Claude's: model name and usage ring."""
    used = usage["questions"]
    fraction = min(used / DAILY_QUESTION_LIMIT, 1.0)
    if usage_limit_reached(usage):
        color = COLORS["error"]
    elif fraction >= 0.8:
        color = COLORS["warning"]
    else:
        color = COLORS["ring"]

    total_tokens = usage["input_tokens"] + usage["output_tokens"]
    tooltip = html.escape(
        f"{used} of {DAILY_QUESTION_LIMIT} questions used today\n"
        f"{format_tokens(total_tokens)} tokens "
        f"({format_tokens(usage['input_tokens'])} in · "
        f"{format_tokens(usage['output_tokens'])} out)\n"
        "Resets daily at 00:00 UTC"
    )
    st.html(
        f"""
        <div class="usage-row">
          <span class="model">{GEMINI_MODEL_LABEL}</span>
          <span class="ring" title="{tooltip}">
            <span class="muted">{used}/{DAILY_QUESTION_LIMIT}</span>
            <span class="dial" style="background: conic-gradient({color} {fraction * 360:.0f}deg, rgba(148, 163, 184, 0.35) 0deg);"></span>
          </span>
        </div>
        """
    )


def render_theme_toggle() -> None:
    """Button that flips between light and dark mode.

    Streamlit has no Python API for changing the viewer's theme, so the
    button runs a tiny script that picks the mode in Streamlit's own menu
    (⋮ → Light / Dark). Streamlit then remembers the choice in the browser.
    A rerun afterwards lets the custom colours above match the new mode.
    """
    is_light = st.context.theme.type == "light"
    target = "Dark" if is_light else "Light"
    if st.button(
        f"{target} mode",
        icon=":material/dark_mode:" if is_light else ":material/light_mode:",
        key="theme_toggle",
        use_container_width=True,
    ):
        st.html(
            f"""
            <script>
            (async () => {{
              const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
              const menuButton = '[data-testid="stMainMenuButton"]';
              async function choose(testId) {{
                const selector = `[data-testid="${{testId}}"]`;
                if (!document.querySelector(selector)) {{
                  document.querySelector(menuButton)?.click();  // open the menu
                }}
                for (let i = 0; i < 40; i++) {{
                  const item = document.querySelector(selector);
                  if (item) {{ item.click(); return true; }}
                  await sleep(25);
                }}
                return false;
              }}
              // Decide from what is on screen, so a stale label can't flip it wrong.
              const bg = getComputedStyle(document.querySelector(".stApp")).backgroundColor;
              const [r, g, b] = bg.match(/\\d+/g).map(Number);
              const isLight = (r + g + b) / 3 > 128;
              await choose(isLight ? "stMainMenuItem-theme-Dark" : "stMainMenuItem-theme-Light");
              // Let Streamlit register the new theme before rerunning, so
              // st.context.theme (and the custom colours) see the new mode.
              await sleep(600);
              await choose("stMainMenuItem-rerun");
            }})();
            </script>
            """,
            unsafe_allow_javascript=True,
        )


def stream_llm_text(prompt: str, token_usage: dict):
    """Stream Gemini's answer text and add up the tokens it reports."""
    for chunk in get_llm(GEMINI_API_KEY).stream(prompt):
        metadata = chunk.usage_metadata
        if metadata:
            token_usage["input"] += metadata.get("input_tokens", 0)
            token_usage["output"] += metadata.get("output_tokens", 0)
        yield chunk.text


def queue_document_processing() -> None:
    st.session_state.is_processing = (
        st.session_state.get("file_uploader") is not None
    )


def get_gemini_api_key() -> str | None:
    """Read the Gemini key from Streamlit secrets (cloud) or .env (local)."""
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        try:
            value = st.secrets.get(name)
        except Exception:  # no secrets.toml when running locally
            value = None
        value = (value or os.getenv(name) or "").strip().strip('"').strip("'")
        if value:
            return value
    return None


initialize_storage()

GEMINI_API_KEY = get_gemini_api_key()
if not GEMINI_API_KEY:
    st.error(
        "No Gemini API key found. Add `GEMINI_API_KEY` to `.env` locally, "
        "or to **Settings → Secrets** on Streamlit Cloud."
    )
    st.stop()


@st.cache_resource
def get_llm(api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        thinking_config={"thinking_level": THINKING_LEVEL},
        max_retries=2,
        timeout=60,
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
st.session_state.setdefault("chat_owner_id", uuid.uuid4().hex)
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
        owner_id=st.session_state.chat_owner_id,
    )


def process_uploaded_document(uploaded_file: UploadedFile, status) -> bool:
    document_hash = get_file_hash(uploaded_file)
    owner_id = st.session_state.chat_owner_id
    existing_chat = get_chat_by_hash(document_hash, owner_id=owner_id)
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
        owner_id=owner_id,
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
            st.html(f'<div class="evidence">{html.escape(preview)}</div>')


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
    st.header("Lumenfold")
    render_theme_toggle()
    if st.button("New chat", use_container_width=True):
        chat_owner_id = st.session_state.chat_owner_id
        st.session_state.clear()
        st.session_state.chat_owner_id = chat_owner_id
        st.rerun()

    saved_chats = list_chats(owner_id=st.session_state.chat_owner_id)
    st.subheader("Previous Chats")
    st.caption("Your chats are private to this browser session.")
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
            saved_chat = get_chat(
                chat_id,
                owner_id=st.session_state.chat_owner_id,
            )
            if saved_chat is None:
                st.error("This saved chat could not be found.")
            else:
                st.session_state.messages = load_messages(
                    chat_id,
                    owner_id=st.session_state.chat_owner_id,
                )
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
                owner_id=st.session_state.chat_owner_id,
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
    st.title("Lumenfold")
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
        except Exception as error:
            logger.exception("Processing %s failed", uploaded_file.name)
            st.error(
                "Could not process this PDF. Please try again, or try a "
                "different PDF if the problem continues."
            )
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
    usage = get_usage_today(st.session_state.chat_owner_id)
    limit_reached = usage_limit_reached(usage)
    # st._bottom is the area pinned under the chat, where st.chat_input lives.
    bottom = st._bottom if hasattr(st, "_bottom") else st.container()
    with bottom:
        query = st.chat_input(
            "Daily demo limit reached. Resets at 00:00 UTC."
            if limit_reached
            else "Ask a question about the document:",
            disabled=limit_reached,
        )
        render_usage_row(usage)
    if query is None:
        query = st.session_state.pop("pending_question", None)
    if query and limit_reached:
        st.info(
            "You've used today's free questions for this demo. Run it locally "
            "with your own Gemini key for unlimited use (see the README)."
        )
        query = None

    if query:
        user_message = {"role": "user", "content": query}
        with st.container(horizontal=True, horizontal_alignment="right"):
            with st.container(width="content", border=True):
                st.caption("You")
                st.markdown(query)

        st.caption("Assistant")
        answer_slot = st.empty()
        try:
            documents = retrieve_documents(st.session_state.vector_db, query)
            if not documents:
                record_exchange(
                    user_message,
                    {"role": "assistant", "content": NO_EVIDENCE_ANSWER, "sources": []},
                )
                st.rerun()

            sources = build_sources(documents)
            prompt = build_prompt(query, documents)
            token_usage = {"input": 0, "output": 0}

            streamed_answer = answer_slot.write_stream(
                stream_llm_text(prompt, token_usage)
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
                        stream_llm_text(verification_prompt, token_usage)
                    ).strip()
                if not answer:
                    raise RuntimeError(
                        "The answer verification request returned no text."
                    )
                answer_slot.markdown(answer)
        except Exception as error:
            logger.exception("Answering a question failed")
            answer_slot.error(
                f"{friendly_error_message(error)} Your question was not saved."
            )
        else:
            render_citation_warning(answer, sources)
            assistant_index = len(st.session_state.messages) + 1
            render_sources(sources, assistant_index)
            record_exchange(
                user_message,
                {"role": "assistant", "content": answer, "sources": sources},
            )
            record_usage(
                st.session_state.chat_owner_id,
                token_usage["input"],
                token_usage["output"],
            )
            st.rerun()  # refresh the usage ring with this answer counted