import hashlib
import re
from functools import lru_cache
from io import BytesIO
from typing import Protocol

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

NO_EVIDENCE_ANSWER = (
    "I could not find enough information in this document to answer that question."
)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVAL_K = 6
MAX_CONTEXT_CHARS = 6000

# Runs locally on CPU: no API key, no quota, no per-request cost.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# BGE models retrieve better when questions (not passages) carry this instruction.
EMBEDDING_QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

SYSTEM_PROMPT = f"""You are a precise PDF question-answering assistant.

Answer ONLY using the provided context.

Rules:
1. Use only facts explicitly stated in the context.
2. Cite the page number for every factual claim, like [Page 4], using only page numbers shown in source labels.
3. If the context does not contain enough information, reply exactly:
   "{NO_EVIDENCE_ANSWER}"
4. Do not use outside knowledge or infer numbers, dates, names, or conclusions.
5. If sources conflict, say that the document contains conflicting information.
6. Briefly quote or closely paraphrase the supporting evidence when answering.
7. Be concise: use 3-6 sentences unless detail is necessary."""

PAGE_CITATION_PATTERN = re.compile(r"\[Page\s+(\d+)\]", flags=re.IGNORECASE)


class UploadedPdf(Protocol):
    name: str

    def getvalue(self) -> bytes: ...


def get_file_hash(uploaded_file: UploadedPdf) -> str:
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def parse_pdf(uploaded_file: UploadedPdf) -> tuple[list[Document], int]:
    reader = PdfReader(BytesIO(uploaded_file.getvalue()))
    documents = []
    for page_number, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            documents.append(
                Document(
                    page_content=page_text,
                    metadata={"source": uploaded_file.name, "page": page_number},
                )
            )
    return documents, len(reader.pages)


def split_pdf_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once per process and reuse it."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
        query_encode_kwargs={
            "normalize_embeddings": True,
            "prompt": EMBEDDING_QUERY_PROMPT,
        },
    )


def build_vector_store(chunks: list[Document]) -> InMemoryVectorStore:
    return InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
    )


def retrieve_documents(
    vector_db: InMemoryVectorStore,
    query: str,
    k: int = RETRIEVAL_K,
) -> list[Document]:
    documents = vector_db.similarity_search(query, k=k)
    return [document for document in documents if document.page_content.strip()]


def build_sources(documents: list[Document]) -> list[dict]:
    return [
        {
            "page": document.metadata.get("page", 0),
            "content": document.page_content,
        }
        for document in documents
    ]


def build_prompt(query: str, documents: list[Document]) -> str:
    context = build_context(documents)
    return f"""{SYSTEM_PROMPT}

Context:
{context}

Question:
{query}

Answer:"""


def build_context(documents: list[Document]) -> str:
    context_parts = []
    characters_used = 0
    for index, document in enumerate(documents, start=1):
        page_number = document.metadata.get("page", 0) + 1
        content = document.page_content.strip()
        if not content:
            continue

        separator_length = 2 if context_parts else 0
        source_label = f"[Source {index} | Page {page_number}]\n"
        remaining = MAX_CONTEXT_CHARS - characters_used - separator_length
        remaining -= len(source_label)
        if remaining <= 0:
            break

        truncated = len(content) > remaining
        if truncated:
            marker = "\n[truncated]"
            if remaining > len(marker):
                content = content[: remaining - len(marker)].rstrip() + marker
            else:
                content = content[:remaining]

        context_parts.append(f"{source_label}{content}")
        characters_used += separator_length + len(source_label) + len(content)
        if truncated:
            break

    return "\n\n".join(context_parts)


def build_verification_prompt(
    query: str,
    answer: str,
    documents: list[Document],
) -> str:
    context = build_context(documents)
    return f"""You are a precise answer verifier.

Check every factual claim in the draft answer against the context. Remove claims that are not explicitly supported, correct citations using only page numbers in source labels, and preserve supported claims. If the context supports no answer, reply exactly:
"{NO_EVIDENCE_ANSWER}"
If sources conflict, state that the document contains conflicting information. Do not use outside knowledge.

Context:
{context}

Question:
{query}

Draft answer:
{answer}

Verified answer:"""


def invalid_page_citations(answer: str, sources: list[dict]) -> list[int]:
    available_pages = {source.get("page", 0) + 1 for source in sources}
    cited_pages = {
        int(page_number)
        for page_number in PAGE_CITATION_PATTERN.findall(answer)
    }
    return sorted(cited_pages - available_pages)


def missing_page_citations(answer: str) -> bool:
    if answer.strip() == NO_EVIDENCE_ANSWER:
        return False
    return not PAGE_CITATION_PATTERN.search(answer)


def needs_answer_verification(answer: str, sources: list[dict]) -> bool:
    return bool(invalid_page_citations(answer, sources)) or missing_page_citations(
        answer
    )
