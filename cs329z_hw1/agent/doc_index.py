"""Problem 9 (search_docs): Design retrieval for an agent (5 points)"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from cs329z_hw1 import data
from cs329z_hw1.pipelines.bm25 import BM25Index
from cs329z_hw1.tokenizer import tokenize
from cs329z_hw1.tokens import CHARS_PER_TOKEN, count_tokens
from cs329z_hw1.types import DocWindow, SearchResult

LINES_PER_CHUNK = 3  # non-blank lines (paragraphs or headings) per chunk
SNIPPET_MAX_TOKENS = 150  # from assignment
WINDOW_MAX_TOKENS = 1500  # from assignment


@dataclass
class Chunk:
    """A run of LINES_PER_CHUNK non-blank lines of one document. Line numbers are 1-based
    positions via `text.splitlines()`. Since a chunk is 3 non-empty lines, indexing via
    start_line or end_line may contain some empty lines. A chunk never spans two sections, so
    the last chunk of a section may be shorter.
    """

    doc_id: str
    start_line: int
    end_line: int
    text: str  # what's indexed: the title, section heading, then the chunk's non-blank lines


@dataclass
class DocIndex:
    bm25: BM25Index
    chunks: list[Chunk]
    lines: dict[str, list[str]]  # doc_id -> doc.splitlines() (includes blank lines)
    titles: dict[str, str]  # doc_id -> title line


def build_doc_index(docs: list[dict]) -> DocIndex:
    """
    Indexes docs in chunks of LINES_PER_CHUNK non-blank lines.
    """
    chunks: list[Chunk] = []
    lines: dict[str, list[str]] = {}
    titles: dict[str, str] = {}

    for doc in docs:
        doc_id, title = doc["doc_id"], doc["title"]
        doc_lines = doc["text"].splitlines()
        lines[doc_id] = doc_lines
        titles[doc_id] = title

        # Group the 1-based line numbers worth indexing (not blank, not the "# Title" line).
        # A heading always starts a new group, so no chunk spans two sections.
        #
        # TODO: Add intro/title SECTION, not just the title line
        chunks_line_nums: list[list[int]] = []
        for number, line in enumerate(doc_lines, start=1):
            if not line.strip() or line.strip() == f"# {title}":
                continue
            if (
                line.startswith("#")
                or not chunks_line_nums
                or len(chunks_line_nums[-1]) == LINES_PER_CHUNK
            ):
                chunks_line_nums.append([])
            chunks_line_nums[-1].append(number)

        # Indexed text: title, then the current section heading (unless the chunk starts with it)
        heading = None
        for chunk_line_nums in chunks_line_nums:
            context = [title]
            first_line = doc_lines[chunk_line_nums[0] - 1]
            if first_line.startswith("#"):
                heading = first_line
            elif heading is not None:
                context.append(heading)
            text = "\n".join(
                context
                + [doc_lines[chunk_line_num - 1] for chunk_line_num in chunk_line_nums]
            )
            chunks.append(Chunk(doc_id, chunk_line_nums[0], chunk_line_nums[-1], text))

    return DocIndex(BM25Index([chunk.text for chunk in chunks]), chunks, lines, titles)


def search_docs(index: DocIndex, query: str, k: int = 5) -> list[SearchResult]:
    """
    At most k ranked results, each with doc_id, title, and a snippet of at most SNIPPET_MAX_TOKENS tokens
    taken from the document. No matches returns an empty list.
    """
    results: list[SearchResult] = []
    for chunk_id, _ in index.bm25.search(query, k):
        chunk = index.chunks[chunk_id]
        results.append(
            {
                "doc_id": chunk.doc_id,
                "title": index.titles[chunk.doc_id],
                "snippet": _snippet(chunk.text, set(tokenize(query))),
            }
        )
    return results


def _snippet(chunk_text: str, query_terms: set[str]) -> str:
    """
    Start snippet at the line with the most found query items, instead of just the first line of the chunk.

    This helps pass the `test_a_word_found_in_one_document_finds_that_document` test.

    Bounded by SNIPPET_MAX_TOKENS tokens.
    """
    chunk_lines = chunk_text.splitlines()
    hits = [len(query_terms & set(tokenize(line))) for line in chunk_lines]
    best = hits.index(max(hits))
    return "\n".join(chunk_lines[best:])[: SNIPPET_MAX_TOKENS * CHARS_PER_TOKEN]


def read_doc(index: DocIndex, doc_id: str, start_line: int = 1) -> DocWindow:
    if doc_id not in index.lines:
        raise ValueError(f"Unknown doc_id {doc_id!r}.")
    lines = index.lines[doc_id]
    if not 1 <= start_line <= len(lines):
        raise ValueError(
            f"start_line {start_line} is outside of the bounds of [1, {len(lines)}] for {doc_id!r}."
        )

    text = lines[start_line - 1]
    end_line = start_line
    while end_line < len(lines):
        longer = text + "\n" + lines[end_line]
        if count_tokens(longer) > WINDOW_MAX_TOKENS:
            break
        text, end_line = longer, end_line + 1

    return {
        "doc_id": doc_id,
        "start_line": start_line,
        "end_line": end_line,
        "total_lines": len(lines),
        "text": text,
    }


@functools.cache
def doc_index() -> DocIndex:
    """The index the tools use: built the first time a tool needs it, never when a session
    is created, and shared by every session in the process."""
    return build_doc_index(data.load_docs())
