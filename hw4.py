"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================

from bs4 import BeautifulSoup
from rank_bm25 import BM25Okapi

TARGET_CHUNK_CHARS = 1250      # merge/split target, safely below MAX_CHUNK_CHARS
LABEL_BOOST = 5.0              # BM25 bonus per question word found in a table row label
TOP_PER_COMPANY = 3            # chunks kept per company for comparison questions

# Words that carry no information about what a question asks for.
QUESTION_STOPWORDS = set("""
a an the what was were did does do how much many in on at of for to from by with and or its it
is are as that this which who whom whose held hold report reported generate generated earn
earned value give gives set sets fy usd millions percent per share shares company companies
end ended fiscal year more most higher largest single had have has firm announce announced
announces city headquartered headquarters forecast forecasts spend spent purchases purchase
large size audit audited auditing
""".split())

# Light query expansion: terms analysts use vs terms the documents use.
QUERY_SYNONYMS = {
    "cash": ["equivalents"], "borrowings": ["debt"], "borrowing": ["borrowings"],
    "headcount": ["employees"], "revenue": ["sales"], "income": ["earnings"],
    "guide": ["guidance"], "repurchase": ["buyback", "programme"], "buyback": ["repurchase", "programme"],
    "inventories": ["inventory"],
}

# Corpus facts collected while chunking (chunk_document runs before parse_query/build_index).
_CORPUS: dict = {}          # company_key -> {"name": display name, "docs": {(fiscal_year, doc_type)}}
_PARSE_CACHE: dict = {}     # question -> parsed query (parse_query and retrieve share it)


def _tokens(text: str) -> list:
    """Lower-cased word tokens; numbers keep their separators ('2,710.4' stays one token)."""
    return re.findall(r"[a-z]+|\d[\d,.]*", str(text).lower())


def _expand_query(question: str) -> list:
    words = _tokens(question)
    extra = []
    for w in words:
        extra.extend(QUERY_SYNONYMS.get(w, []))
    return words + extra


def _extract_meta(soup: BeautifulSoup):
    """Company, fiscal year and document type from the first lines of a document."""
    h1 = soup.find("h1")
    h1_text = h1.get_text(" ", strip=True) if h1 else ""
    h2 = soup.find("h2")
    h2_text = h2.get_text(" ", strip=True) if h2 else ""
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    broker = soup.find("div", class_="broker")
    broker_text = broker.get_text(" ", strip=True) if broker else ""
    head = " ".join([title, h1_text, h2_text, broker_text]).lower()
    if "earnings conference call" in head or "earnings call" in head:
        doc_type = "earnings_call"
    elif "results review" in head or "equity research" in head:
        doc_type = "research_report"
    else:
        doc_type = "annual_report"
    company = re.sub(r"\s*\([^)]*\)\s*$", "", h1_text).strip()  # drop the "(NYSE: XXX)" part
    head_text = " ".join([title, h1_text, h2_text] +
                         [p.get_text(" ", strip=True) for p in soup.find_all("p")[:4]]).lower()
    m = (re.search(r"annual report (20\d\d)", head_text)
         or re.search(r"full year (20\d\d)", head_text)
         or re.search(r"fy(20\d\d) results review", head_text)
         or re.search(r"fiscal year ended [^\d]*(20\d\d)", head_text)
         or re.search(r"(20\d\d)", head_text))
    return company, (int(m.group(1)) if m else None), doc_type


def _register_document(company: str, fiscal_year, doc_type: str) -> None:
    key = company_key(company)
    entry = _CORPUS.setdefault(key, {"name": company, "docs": set()})
    if fiscal_year is not None and doc_type in DOC_TYPES:
        entry["docs"].add((fiscal_year, doc_type))


def _row_labels(text: str) -> list:
    """Normalised row labels of the table lines in a chunk (the part before the first ' | ')."""
    return [normalize(line.split(" | ")[0]) for line in text.splitlines() if " | " in line]


def _question_words(question: str, companies: list) -> list:
    """Content words of the question, without stopwords and company-name words."""
    company_words = set()
    for name in companies or []:
        company_words.update(company_key(name).split())
    return [w for w in re.findall(r"[a-z]+", question.lower())
            if w not in QUESTION_STOPWORDS and len(w) > 2 and w not in company_words]


def _word_known(word: str, vocab: set) -> bool:
    """The word, or a close inflection of it, appears in the vocabulary."""
    if word in vocab:
        return True
    if len(word) >= 5 and any(v.startswith(word) or word.startswith(v)
                              for v in vocab if abs(len(v) - len(word)) <= 4):
        return True
    return False


def _parse_query_rules(question: str) -> dict:
    """Deterministic query parsing: corpus-name matching, year regex, doc-type keywords."""
    normalised = normalize(question)
    companies = [info["name"] for key, info in _CORPUS.items() if f" {key} " in normalised]
    m = (re.search(r"fy\s?(20\d\d)", question, re.I)
         or re.search(r"31 december (20\d\d)", question, re.I)
         or re.search(r"\b(20\d\d)\b", question))
    fiscal_year = int(m.group(1)) if m else None
    ql = question.lower()
    if re.search(r"earnings call|conference call|\bq4\b|fourth.quarter", ql):
        doc_type = "earnings_call"
    elif re.search(r"\brating\b|target price|results review|\bforecast|\bequity research\b|"
                   r"\bsecurities\b|\bcapital\b|\bresearch\b|\bbroker|\bestimate", ql):
        doc_type = "research_report"
    else:
        doc_type = "annual_report"
    return {"companies": companies, "fiscal_year": fiscal_year, "doc_type": doc_type}


_PARSE_PROMPT = """Extract the companies, the fiscal year and the document type from an analyst's question.

Companies in the library: {companies}
The question may still name a company that is NOT in the library; then report the name as written.
doc_type is one of: annual_report, earnings_call, research_report.
- earnings_call: questions about the earnings call, guidance given on it, the share repurchase announced on it.
- research_report: questions about a broker's rating, target price, estimates or forecasts.
- annual_report: everything else (financial statements, segments, employees, officers, auditor, headquarters).
fiscal_year is the year the question asks about (e.g. "FY2023" or "at 31 December 2024"), or null if none.

Question: {question}

Reply with JSON only, e.g. {{"companies": ["Northgate Robotics"], "fiscal_year": 2024, "doc_type": "annual_report"}}"""


def _parse_query_llm(question: str) -> dict | None:
    """LLM fallback for questions whose company names the rules cannot match."""
    try:
        llm = make_llm(max_tokens=150)
        prompt = _PARSE_PROMPT.format(companies=", ".join(sorted(i["name"] for i in _CORPUS.values())),
                                      question=question)
        text = str(llm.invoke(prompt).content)
        match = re.search(r"\{.*\}", text, re.S)
        parsed = json.loads(match.group(0)) if match else None
        if not isinstance(parsed, dict):
            return None
        companies = parsed.get("companies") or []
        if isinstance(companies, str):
            companies = [companies]
        doc_type = parsed.get("doc_type")
        return {"companies": [str(c) for c in companies],
                "fiscal_year": parsed.get("fiscal_year") if parsed.get("fiscal_year") is None else int(parsed["fiscal_year"]),
                "doc_type": doc_type if doc_type in DOC_TYPES else None}
    except Exception:
        return None


_ANSWERABILITY_PROMPT = """Question: {question}

Passages:
{context}

Does any passage above state the exact figure, name or rating the question asks for?
Answer with one word: YES or NO."""


def _answerable_with_llm(question: str, chunks: list[Document]) -> bool:
    """Last abstention layer: do the retrieved passages actually state what is asked?

    Catches figures the company simply does not disclose (e.g. lobbying expenses): the
    company and the year exist, so the rule layers pass, but no passage contains the
    answer. Fails open (keeps the chunks) if the model cannot be called.
    """
    try:
        llm = make_llm(max_tokens=4)
        context = "\n\n".join(c.page_content[:900] for c in chunks)
        answer = str(llm.invoke(_ANSWERABILITY_PROMPT.format(question=question, context=context)).content)
        return answer.strip().upper().startswith("YES")
    except Exception:
        return True


def _abstain_reason(question: str, parsed: dict, index: dict) -> str | None:
    """Why the corpus cannot answer this question (None means it can)."""
    keys = [company_key(c) for c in parsed.get("companies") or []]
    if not keys:
        return "no company parsed"
    for key in keys:
        if key not in _CORPUS:
            return f"company not in corpus: {key}"
    fiscal_year, doc_type = parsed.get("fiscal_year"), parsed.get("doc_type")
    for key in keys:
        available = _CORPUS[key]["docs"]
        if doc_type == "annual_report":
            # a report covers its own year and, in the prior-year column, the year before
            if fiscal_year is not None and not any(
                    dt == "annual_report" and year in (fiscal_year, fiscal_year + 1) for year, dt in available):
                return f"no annual report covering FY{fiscal_year} for {key}"
        elif not any(dt == doc_type for _, dt in available):
            return f"no {doc_type} for {key}"
    # every distinctive word of the question must appear in the company's own documents
    unknown = [w for w in _question_words(question, parsed.get("companies"))
               if any(not _word_known(w, index["vocab"][key]) for key in keys)]
    if unknown:
        return f"terms not in the company's documents: {unknown}"
    return None


def table_to_text(table_html: str) -> str:
    """Task 1.1: turn one HTML <table> into text that keeps every value with its row and its column.

    Put each value on a line (at most 400 characters) together with its row label and its column
    header, e.g. "Revenue | 2025: 2,710.4 | 2024: 2,629.1", one table row per line, keep the text of
    every cell unchanged, and include the caption (it holds the unit) in the output. Do not call the
    LLM here.
    """
    ### YOUR CODE HERE
    soup = BeautifulSoup(table_html, "lxml")
    table = soup.find("table") or soup
    caption_tag = table.find("caption")
    caption = caption_tag.get_text(" ", strip=True) if caption_tag else ""

    headers = []
    thead = table.find("thead")
    if thead:
        for row in thead.find_all("tr"):
            headers = [c.get_text(" ", strip=True) for c in row.find_all(["th", "td"])]
            if any(headers):
                break

    lines = [caption] if caption else []
    tbody = table.find("tbody") or table
    for row in tbody.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        # skip divider rows such as "Current assets" that span the whole table
        if len(cells) == 1 or any(c.get("colspan") and int(c.get("colspan")) >= 2 and len(cells) <= 2
                                  for c in cells):
            continue
        label = cells[0].get_text(" ", strip=True)
        values = [c.get_text(" ", strip=True) for c in cells[1:]]
        if len(headers) >= len(values) + 1:
            cols = headers[1:1 + len(values)]
        else:
            cols = headers[-len(values):] if headers else [""] * len(values)
        first_header = headers[0] if headers else ""
        label_text = f"{first_header}: {label}" if first_header else label
        parts = [f"{col}: {val}" if col else val for col, val in zip(cols, values)]
        lines.append(f"{label_text} | " + " | ".join(parts))
    return "\n".join(lines)


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Task 1.2: split one document into chunks of at most MAX_CHUNK_CHARS characters.

    Every chunk needs the metadata "source" (= file_name), "company", "fiscal_year" (an int) and
    "doc_type" (one of DOC_TYPES). Do not call the LLM here.
    """
    ### YOUR CODE HERE
    soup = BeautifulSoup(html, "lxml")
    body = soup.body or soup
    company, fiscal_year, doc_type = _extract_meta(soup)
    _register_document(company, fiscal_year, doc_type)
    meta = {"source": file_name, "company": company, "fiscal_year": fiscal_year, "doc_type": doc_type}
    header = f"{company} | fiscal year {fiscal_year} | {doc_type.replace('_', ' ')}"

    # collect (section, text) items in reading order; tables go through table_to_text
    items = []
    section = "Introduction"
    nav = body.find("nav")
    if nav:
        nav.decompose()
    for el in body.find_all(["h1", "h2", "h3", "p", "ul", "ol", "table"]):
        if el.name in ("h1", "h2", "h3"):
            section = el.get_text(" ", strip=True)
            continue
        if el.name == "table":
            text = table_to_text(str(el))
        elif el.name in ("ul", "ol"):
            text = "\n".join(li.get_text(" ", strip=True) for li in el.find_all("li", recursive=False))
        else:
            text = el.get_text(" ", strip=True)
        text = re.sub(r"[ \t]+", " ", text).strip()
        if not text:
            continue
        # skip the table-of-contents list (it duplicates the section headings)
        if text.lower().startswith("letter to shareholders") and "risk factors" in text.lower() and len(text) < 400:
            continue
        items.append((section, text))

    # split oversized items into units, keeping every table row on its own line
    pieces = []
    for section, text in items:
        units = []
        for part in text.split("\n"):
            if len(part) > TARGET_CHUNK_CHARS:
                units.extend(s for s in re.split(r"(?<=[.!?])\s+", part) if s)
            else:
                units.append(part)
        buf = ""
        for unit in units:
            if buf and len(buf) + len(unit) + 1 > TARGET_CHUNK_CHARS:
                pieces.append((section, buf))
                buf = unit
            else:
                buf = f"{buf}\n{unit}" if buf else unit
        if buf:
            pieces.append((section, buf))

    # merge small consecutive pieces of the same section into chunks
    merged, cur_section, cur = [], None, ""
    for section, text in pieces:
        if section != cur_section or (cur and len(cur) + len(text) + 1 > TARGET_CHUNK_CHARS):
            if cur:
                merged.append((cur_section, cur))
            cur_section, cur = section, text
        else:
            cur = f"{cur}\n{text}" if cur else text
    if cur:
        merged.append((cur_section, cur))

    chunks = []
    for section, text in merged:
        page = f"{header} | section: {section}\n{text}"
        if len(page) > MAX_CHUNK_CHARS:  # safety net, normally unreachable
            page = page[:MAX_CHUNK_CHARS]
        chunk_meta = dict(meta)
        chunk_meta["section"] = section
        chunks.append(Document(page_content=page, metadata=chunk_meta))
    return chunks


def parse_query(question: str) -> dict:
    """Task 2.2: {"companies": [names], "fiscal_year": int or None, "doc_type": one of DOC_TYPES}.

    The rules for each field are under Task 2.2 in the handout. You may use make_llm() here.
    """
    ### YOUR CODE HERE
    if question in _PARSE_CACHE:
        return _PARSE_CACHE[question]
    parsed = _parse_query_rules(question)
    if not parsed["companies"]:
        llm_parsed = _parse_query_llm(question)
        if llm_parsed:
            if llm_parsed["companies"]:
                parsed["companies"] = llm_parsed["companies"]
            if llm_parsed["doc_type"]:
                parsed["doc_type"] = llm_parsed["doc_type"]
            if llm_parsed["fiscal_year"] is not None or parsed["fiscal_year"] is None:
                parsed["fiscal_year"] = llm_parsed["fiscal_year"]
    _PARSE_CACHE[question] = parsed
    return parsed


def build_index(chunks: list[Document]) -> Any:
    """Task 2: index the chunks of the whole corpus (the output of chunk_document), once per run.

    Return anything retrieve() needs: a BM25 or TF-IDF index, the chunks grouped by metadata, ...
    """
    ### YOUR CODE HERE
    tokens = [_tokens(c.page_content) for c in chunks]
    vocab: dict = {}
    for chunk, toks in zip(chunks, tokens):
        vocab.setdefault(company_key(chunk.metadata.get("company", "")), set()).update(toks)
    return {
        "chunks": list(chunks),
        "bm25": BM25Okapi(tokens),
        "row_labels": [_row_labels(c.page_content) for c in chunks],
        "vocab": vocab,
    }


def retrieve(index: Any, question: str) -> list[Document]:
    """Tasks 2 and 3: at most MAX_RETRIEVED of the chunks given to build_index, best first.

    Return [] when the corpus cannot answer the question: the generator then answers NOT_FOUND.
    """
    ### YOUR CODE HERE
    try:
        parsed = parse_query(question)
        if _abstain_reason(question, parsed, index):
            return []

        chunks = index["chunks"]
        base = index["bm25"].get_scores(_expand_query(question))
        q_words = _question_words(question, parsed.get("companies"))

        def label_bonus(i: int) -> float:
            best = 0
            for label in index["row_labels"][i]:
                best = max(best, sum(1 for w in q_words if f" {w} " in label))
            return LABEL_BOOST * best

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
            out.sort(key=lambda i: -(base[i] + label_bonus(i)))
            return out

        keys = [company_key(c) for c in parsed.get("companies") or []]
        if len(keys) >= 2:
            # comparison: guarantee chunks of every company named in the question
            per_company = [candidates(key)[:TOP_PER_COMPANY] for key in keys]
            picked = []
            for pair in zip(*per_company):
                picked.extend(pair)
            rest = sorted((i for group in per_company for i in group),
                          key=lambda i: -(base[i] + label_bonus(i)))
            for i in rest:
                if len(picked) >= MAX_RETRIEVED:
                    break
                if i not in picked:
                    picked.append(i)
        else:
            picked = candidates(keys[0])[:MAX_RETRIEVED]
        result = [chunks[i] for i in picked[:MAX_RETRIEVED]]
        if result and not _answerable_with_llm(question, result):
            return []
        return result
    except Exception:
        traceback.print_exc()
        return []


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
