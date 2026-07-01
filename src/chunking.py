
import re
import pdfplumber
from pathlib import Path
from typing import List, Dict
import re
from typing import List, Dict


import tiktoken
_ENC = tiktoken.get_encoding("cl100k_base")

def clean_page_text(text: str) -> str:
    if not text:
        return ""

    # Fix words split across a line break by a hyphen: "exam-\nple" -> "example"
    text = re.sub(r"-\n(?=[a-z])", "", text)

    # Normalize remaining newlines to spaces within a paragraph,
    # but keep paragraph breaks (double newline) intact
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Strip lines that are just numbers (likely page numbers)
    lines = text.split("\n")
    lines = [ln for ln in lines if not re.fullmatch(r"\s*\d+\s*", ln)]
    text = "\n".join(lines)

    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def extract_book(pdf_path: str, book_title: str) -> List[Dict]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            raw_text = page.extract_text() or ""
            cleaned = clean_page_text(raw_text)
            if len(cleaned) < 20:
                # Skip near-empty pages (cover, blank, etc.)
                continue
            pages.append({
                "book": book_title,
                "page_num": i + 1,
                "text": cleaned,
            })
    return pages


def extract_corpus(book_specs: List[Dict], out_path: str = None) -> List[Dict]:
  
    import json

    all_pages = []
    for spec in book_specs:
        print(f"Extracting: {spec['title']} ({spec['path']})")
        pages = extract_book(spec["path"], spec["title"])
        print(f"  -> {len(pages)} non-empty pages extracted")
        all_pages.extend(pages)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in all_pages:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(all_pages)} page records to {out_path}")

    return all_pages



def count_tokens(text: str) -> int:
    return len(_ENC.encode(text))




def split_into_paragraphs(text: str) -> List[str]:

    paras = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paras) <= 1:
        # Fallback: split on sentence boundaries if no paragraph breaks found
        paras = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        paras = [p.strip() for p in paras if p.strip()]
    return paras


def chunk_pages(
    pages: List[Dict],
    target_tokens: int = 600,
    overlap_tokens: int = 120,
) -> List[Dict]:
  
    chunks = []
    chunk_counter = 0

    # Group pages by book, preserving order
    books = {}
    for p in pages:
        books.setdefault(p["book"], []).append(p)

    for book, book_pages in books.items():
        book_pages = sorted(book_pages, key=lambda x: x["page_num"])

        # Flatten this book into a list of (paragraph_text, page_num) tuples
        flat_paras = []
        for page in book_pages:
            for para in split_into_paragraphs(page["text"]):
                flat_paras.append((para, page["page_num"]))

        # Greedily pack paragraphs into chunks with overlap
        i = 0
        while i < len(flat_paras):
            current_paras = []
            current_tokens = 0
            start_idx = i
            page_start = flat_paras[i][1]

            while i < len(flat_paras) and current_tokens < target_tokens:
                para_text, page_num = flat_paras[i]
                current_paras.append(para_text)
                current_tokens += count_tokens(para_text)
                i += 1

            page_end = flat_paras[i - 1][1] if i > 0 else page_start
            chunk_text = "\n\n".join(current_paras)

            chunks.append({
                "chunk_id": f"{book.replace(' ', '_')}_{chunk_counter:05d}",
                "book": book,
                "page_start": page_start,
                "page_end": page_end,
                "text": chunk_text,
                "token_count": count_tokens(chunk_text),
            })
            chunk_counter += 1

            # Step back to create overlap: walk backwards from i until
            # we've "given back" roughly overlap_tokens worth of paragraphs
            if i >= len(flat_paras):
                break
            overlap_count = 0
            back = i - 1
            while back > start_idx and overlap_count < overlap_tokens:
                overlap_count += count_tokens(flat_paras[back][0])
                back -= 1
            i = max(back + 1, start_idx + 1)  # always make progress

    return chunks



