"""Task 5: ablation study for the HW4 retrieval pipeline.

Starting from a naive baseline (B0, as in Tutorial 7), each variant changes ONE
component; every configuration is measured on public_test with the same
metrics as the runner (recall@5, kept/rejected, end-to-end answer accuracy).

    python3 ablation.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

import hw4
from hw4 import (MAX_RETRIEVED, _CORPUS, _abstain_reason, _answerable_with_llm, _expand_query,
                 _extract_meta, _question_words, _register_document, _row_labels, _tokens,
                 answer_correct, chunk_document, company_key, generate, load_corpus, load_json,
                 normalize, parse_query, retrieval_hit, table_to_text)

# ----------------------------------------------------------------------------- chunk builders

def _meta_dict(file_name: str, soup) -> dict:
    company, fiscal_year, doc_type = _extract_meta(soup)
    _register_document(company, fiscal_year, doc_type)
    return {"source": file_name, "company": company, "fiscal_year": fiscal_year,
            "doc_type": doc_type}


def chunks_b0(file_name: str, html: str, use_table_to_text: bool) -> list[Document]:
    """B0-style chunks: flat text, RecursiveCharacterTextSplitter, required metadata only."""
    soup = BeautifulSoup(html, "lxml")
    body = soup.body or soup
    meta = _meta_dict(file_name, soup)
    nav = body.find("nav")
    if nav:
        nav.decompose()
    if use_table_to_text:  # V1: tables through table_to_text, everything else flat
        parts = []
        for el in body.find_all(["table", "p", "h1", "h2", "h3", "ul", "ol"]):
            if el.name == "table":
                parts.append(table_to_text(str(el)))
            elif el.name in ("ul", "ol"):
                parts.append("\n".join(li.get_text(" ", strip=True) for li in el.find_all("li", recursive=False)))
            else:
                parts.append(el.get_text(" ", strip=True))
        text = "\n".join(p for p in parts if p)
    else:  # B0: the raw get_text() of the whole document
        text = body.get_text("\n", strip=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return [Document(page_content=t, metadata=dict(meta)) for t in splitter.split_text(text)]


# ----------------------------------------------------------------------------- retrievers

def retrieve_naive(index, question: str) -> list[Document]:
    """B0/V1/V2 retrieval: BM25 top-5 over the whole corpus, no parsing, never abstains."""
    scores = index["bm25"].get_scores(index["tokenize"](question))
    top = sorted(range(len(index["chunks"])), key=lambda i: -scores[i])[:MAX_RETRIEVED]
    return [index["chunks"][i] for i in top]


def retrieve_filtered(index, question: str, use_boost: bool, abstain: bool) -> list[Document]:
    """V3+: metadata routing via parse_query; V4 adds row-label boost + expansion; V5 adds abstention."""
    parsed = parse_query(question)
    if abstain and _abstain_reason(question, parsed, index):
        return []
    chunks = index["chunks"]
    query = _expand_query(question) if use_boost else index["tokenize"](question)
    base = index["bm25"].get_scores(query)
    q_words = _question_words(question, parsed.get("companies"))

    def bonus(i: int) -> float:
        if not use_boost:
            return 0.0
        return hw4.LABEL_BOOST * max((sum(1 for w in q_words if f" {w} " in lab)
                                      for lab in index["row_labels"][i]), default=0)

    def candidates(key: str) -> list:
        fiscal_year, doc_type = parsed.get("fiscal_year"), parsed.get("doc_type")
        out = []
        for i, chunk in enumerate(chunks):
            meta = chunk.metadata
            if company_key(meta.get("company", "")) != key or meta.get("doc_type") != doc_type:
                continue
            if doc_type == "annual_report" and fiscal_year is not None:
                years = sorted(y for y, dt in _CORPUS[key]["docs"] if dt == "annual_report")
                target = fiscal_year if fiscal_year in years else (
                    fiscal_year + 1 if fiscal_year + 1 in years else None)
                if target is None or meta.get("fiscal_year") != target:
                    continue
            out.append(i)
        out.sort(key=lambda i: -(base[i] + bonus(i)))
        return out

    keys = [company_key(c) for c in parsed.get("companies") or []]
    if not keys:
        return []
    if len(keys) >= 2:
        per = [candidates(k)[:hw4.TOP_PER_COMPANY] for k in keys]
        picked = []
        for pair in zip(*per):
            picked.extend(pair)
        rest = sorted((i for g in per for i in g), key=lambda i: -(base[i] + bonus(i)))
        for i in rest:
            if len(picked) >= MAX_RETRIEVED:
                break
            if i not in picked:
                picked.append(i)
    else:
        picked = candidates(keys[0])[:MAX_RETRIEVED]
    result = [chunks[i] for i in picked[:MAX_RETRIEVED]]
    if abstain and result and not _answerable_with_llm(question, result):
        return []
    return result


# ----------------------------------------------------------------------------- configurations

def make_index(chunks: list[Document], whitespace_tokens: bool) -> dict:
    tokenize = (lambda t: str(t).lower().split()) if whitespace_tokens else _tokens
    tokens = [tokenize(c.page_content) for c in chunks]
    vocab: dict = {}
    for chunk, toks in zip(chunks, tokens):
        vocab.setdefault(company_key(chunk.metadata.get("company", "")), set()).update(toks)
    return {"chunks": chunks, "bm25": BM25Okapi(tokens), "tokenize": tokenize,
            "row_labels": [_row_labels(c.page_content) for c in chunks], "vocab": vocab}


CONFIGS = [
    ("B0", "naive RAG: get_text(), RecursiveCharacterTextSplitter, BM25 top-5, no parsing, never abstains"),
    ("V1", "B0 + table_to_text for the tables"),
    ("V2", "V1 + section-aware chunking with contextual chunk headers"),
    ("V3", "V2 + parse_query routing: company/doc-type/year filtering, per-company retrieval for comparisons"),
    ("V4", "V3 + table row-label boost and query expansion"),
    ("V5", "V4 + abstention (rules + LLM answerability check) = the submitted pipeline"),
]


def build_chunks(config: str, corpus: dict[str, str]) -> list[Document]:
    by_file = {}
    for file_name, html in sorted(corpus.items()):
        if config == "B0":
            by_file[file_name] = chunks_b0(file_name, html, use_table_to_text=False)
        elif config == "V1":
            by_file[file_name] = chunks_b0(file_name, html, use_table_to_text=True)
        else:
            by_file[file_name] = chunk_document(file_name, html)
    return [c for f in sorted(by_file) for c in by_file[f]]


def run_config(config: str, corpus, questions, ground_truth) -> dict:
    chunks = build_chunks(config, corpus)
    whitespace = config in ("B0", "V1", "V2")
    index = make_index(chunks, whitespace_tokens=whitespace)

    def retrieve_one(question: str) -> list[Document]:
        if config in ("B0", "V1", "V2"):
            return retrieve_naive(index, question)
        if config == "V3":
            return retrieve_filtered(index, question, use_boost=False, abstain=False)
        if config == "V4":
            return retrieve_filtered(index, question, use_boost=True, abstain=False)
        return retrieve_filtered(index, question, use_boost=True, abstain=True)  # V5

    retrieved = {qid: retrieve_one(q) for qid, q in questions.items()}
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(lambda qid: _safe_generate(questions[qid], retrieved[qid]), questions)))

    graded = [qid for qid in questions if ground_truth and qid in ground_truth]
    answerable = [qid for qid in graded if ground_truth[qid]["evidence"]]
    unanswerable = [qid for qid in graded if not ground_truth[qid]["evidence"]]
    recall = sum(retrieval_hit(retrieved[qid], ground_truth[qid]) for qid in answerable) / max(len(answerable), 1)
    kept = sum(bool(retrieved[qid]) for qid in answerable) / max(len(answerable), 1)
    rejected = sum(not retrieved[qid] for qid in unanswerable) / max(len(unanswerable), 1)
    accuracy = sum(answer_correct(answers[qid], ground_truth[qid]) for qid in graded) / max(len(graded), 1)
    return {"config": config, "chunks": len(chunks), "recall@5": recall, "kept": kept,
            "rejected": rejected, "accuracy": accuracy, "answers": answers, "retrieved": retrieved}


def _safe_generate(question: str, chunks) -> str:
    try:
        return generate(question, chunks)
    except Exception as e:
        return f"error: {e!r}"


def main() -> None:
    parser = argparse.ArgumentParser(description="HW4 Task 5 ablation study")
    parser.add_argument("--data-folder", default="public_test")
    parser.add_argument("--output", default="ablation_results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")

    rows = []
    for config, description in CONFIGS:
        result = run_config(config, corpus, questions, ground_truth)
        rows.append({"config": config, "change": description, "chunks": result["chunks"],
                     "recall@5": round(result["recall@5"], 3), "kept": round(result["kept"], 3),
                     "rejected": round(result["rejected"], 3), "accuracy": round(result["accuracy"], 3)})
        print(f"{config}: recall {result['recall@5']:.1%} | kept {result['kept']:.1%} | "
              f"rejected {result['rejected']:.1%} | accuracy {result['accuracy']:.1%}")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "change", "chunks", "recall@5", "kept", "rejected", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")
    print(f"Your usage: {hw4.USAGE.summary()}")
    print(f"Generator usage: {hw4.GENERATOR_USAGE.summary()}")


if __name__ == "__main__":
    main()
