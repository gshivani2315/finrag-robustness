"""
passage_builder.py
------------------
Turns a TAT-QA *document* (one table + its paragraphs) into searchable passages.

CORPUS CONSTRUCTION STRATEGY (documented once, used everywhere)
---------------------------------------------------------------
For every document we create two kinds of passages:

1. PARAGRAPH passage - one passage per text paragraph, text kept as-is.
     id = "<table_uid>::p<order>"      e.g. "3ffd9053-...::p2"
   `order` is TAT-QA's own 1-based paragraph order, which is exactly what
   the `rel_paragraphs` annotation refers to. So gold evidence ids can be
   derived WITHOUT guessing.

2. TABLE-ROW passage - one passage per non-empty data row of the table.
     id = "<table_uid>::r<row_index>"  (row_index = 0-based index in the raw table)
   Text = row label + row values + the table's column-header text, so a row
   such as "Fixed Price | 2019 | 1,452.4" can be matched to "fixed price 2019".

Why these ids are stable: they are built only from dataset-provided fields
(table uid, paragraph order, raw row index). No randomness, no hashing of
mutable text, no dependence on iteration order of other documents.

IMPORTANT LIMITATION: TAT-QA labels evidence only for PARAGRAPHS
(`rel_paragraphs`). It does NOT label which table rows are relevant.
Table-row passages therefore have *unknown* relevance - they are not
"known irrelevant". evaluate.py handles this explicitly.
"""

from __future__ import annotations

PARAGRAPH = "paragraph"
TABLE_ROW = "table_row"


def paragraph_passage_id(doc_id: str, order) -> str:
    return f"{doc_id}::p{int(order)}"


def table_row_passage_id(doc_id: str, row_index: int) -> str:
    return f"{doc_id}::r{int(row_index)}"


def _clean(cell) -> str:
    """Collapse whitespace ('$  1,452.4' -> '$ 1,452.4'); None -> ''."""
    return " ".join(str(cell if cell is not None else "").split())


def detect_header_rows(table: list[list]) -> int:
    """
    Heuristic: TAT-QA header rows usually have an EMPTY first cell
    (e.g. ['', '2019', '2018']); data rows start with a label.
    - count the leading rows whose first cell is empty
    - if there are none, treat row 0 as the header
    - if that would leave no data rows, treat the table as having no header
    This is a heuristic (the dataset has no header annotation).
    """
    if not table:
        return 0
    n = 0
    for row in table:
        if _clean(row[0] if row else "") == "":
            n += 1
        else:
            break
    if n == 0 and len(table) > 1:
        n = 1
    if n >= len(table):
        n = 0
    return n


def build_header_text(table: list[list], n_header_rows: int) -> str:
    """Flatten header rows: cells joined by ' | ', header rows joined by ' ; '."""
    parts = []
    for row in table[:n_header_rows]:
        cells = [_clean(c) for c in row if _clean(c)]
        if cells:
            parts.append(" | ".join(cells))
    return " ; ".join(parts)


def linearize_row(row: list, header_text: str) -> str:
    """'Table row: <label>. Values: v1 | v2. Column headers: <header>.'"""
    cells = [_clean(c) for c in row]
    label = cells[0] if cells else ""
    values = [c for c in cells[1:] if c]
    text = f"Table row: {label}." if label else "Table row:"
    if values:
        text += " Values: " + " | ".join(values) + "."
    if header_text:
        text += f" Column headers: {header_text}."
    return text


def table_context_text(table: list[list], n_header_rows: int, max_labels: int = 20) -> str:
    """Header text + the first `max_labels` distinct row labels of the document's own table."""
    labels: list[str] = []
    for row in table[n_header_rows:]:
        lab = _clean(row[0] if row else "")
        if lab and lab not in labels:
            labels.append(lab)
        if len(labels) >= max_labels:
            break
    parts = [build_header_text(table, n_header_rows), "; ".join(labels)]
    return " ; ".join(p for p in parts if p)


def build_document_passages(doc: dict, split: str, doc_index: int = 0,
                            context: str = "none") -> list[dict]:
    """
    Passages for ONE TAT-QA document. Returns a list of passage dicts.
    context="none"         -> exactly the baseline passages.
    context="table_labels" -> paragraph passages get an extra field `search_text` =
        paragraph text + the document's own table header and row labels. The retriever
        indexes `search_text`; `text` stays the original paragraph for downstream use.
        Uses only document content, never the question, answer or gold labels.
    """
    if context not in ("none", "table_labels"):
        raise ValueError("context must be 'none' or 'table_labels'")
    table_obj = doc.get("table") or {}
    doc_id = table_obj.get("uid")
    if not doc_id:  # should not happen in TAT-QA; fail loudly rather than invent silently
        raise ValueError(f"{split} document #{doc_index} has no table uid")
    passages: list[dict] = []
    table = table_obj.get("table") or []
    n_hdr = detect_header_rows(table)
    ctx = table_context_text(table, n_hdr) if context == "table_labels" else ""

    # --- paragraph passages ---
    for pos, para in enumerate(doc.get("paragraphs") or []):
        text = _clean(para.get("text"))
        if not text:
            continue
        order = para.get("order", pos + 1)
        p = {
            "passage_id": paragraph_passage_id(doc_id, order),
            "text": text,
            "source_type": PARAGRAPH,
            "doc_id": doc_id,
            "split": split,
            "position": int(order),            # paragraph order (1-based)
            "original_uid": para.get("uid"),   # TAT-QA's own paragraph uid
        }
        if ctx:
            p["search_text"] = f"{text} Table context: {ctx}."
        passages.append(p)

    # --- table-row passages ---
    header_text = build_header_text(table, n_hdr)
    for idx in range(n_hdr, len(table)):
        row = table[idx]
        if not any(_clean(c) for c in row):
            continue  # fully empty row
        passages.append({
            "passage_id": table_row_passage_id(doc_id, idx),
            "text": linearize_row(row, header_text),
            "source_type": TABLE_ROW,
            "doc_id": doc_id,
            "split": split,
            "position": idx,                   # raw row index (0-based)
            "original_uid": None,
        })
    return passages


def build_corpus(documents: list[dict], split: str, context: str = "none") -> list[dict]:
    """
    Corpus for ONE split (never mixes splits -> no train/dev/test leakage).
    Asserts that passage ids are unique.
    """
    corpus: list[dict] = []
    for i, doc in enumerate(documents):
        corpus.extend(build_document_passages(doc, split, i, context))
    ids = [p["passage_id"] for p in corpus]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate passage ids in corpus")
    return corpus