"""Michelin Roanne case knowledge tool (Builder B).

Parses the two extracted documents into (snippet, citation) chunks:
  knowledge/site_dossier.md            → "[page N]" markers → [site_dossier p.N]
  knowledge/rapport_socio_economique.md → ##/### headings   → [rapport_socio_eco §Heading]

Retrieval is deliberately simple (hackathon-grade, no deps): lowercase token
overlap with a light prefix match (cost ~ costliest), plus a bonus for the
exact query phrase. gbrain pattern: results feed SYNTHESIS with citations,
and a zero-score lookup returns an explicit gap marker — never silence.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_KNOW_DIR = Path(__file__).resolve().parent / "knowledge"

_PAGE_RE = re.compile(r"^\[page (\d+)\]\s*$")
_HEAD_RE = re.compile(r"^#{2,4}\s+(.+?)\s*$")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_STOP = {
    "the", "a", "an", "is", "are", "of", "for", "on", "in", "and", "or", "to",
    "what", "how", "with", "it", "at", "by", "be", "this", "that", "does",
    "de", "la", "le", "les", "des", "du", "et", "en", "un", "une", "est",
    "que", "qui", "pour", "sur", "dans", "au", "aux",
}

_CHUNK_TARGET = 700   # pack paragraphs up to ~this many chars
_MIN_CHUNK = 40       # drop layout crumbs


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP]


def _pack(paragraphs: list[str], citation: str) -> list[tuple[str, str]]:
    """Pack consecutive paragraphs into ~_CHUNK_TARGET-char chunks so short
    table rows (e.g. 'Direct loss / hour ≈€20-30 k') stay retrievable."""
    chunks: list[tuple[str, str]] = []
    buf: list[str] = []
    size = 0
    for p in paragraphs:
        p = re.sub(r"\s+", " ", p).strip()
        if not p:
            continue
        buf.append(p)
        size += len(p)
        if size >= _CHUNK_TARGET:
            chunks.append((" · ".join(buf), citation))
            buf, size = [], 0
    if buf:
        text = " · ".join(buf)
        if len(text) >= _MIN_CHUNK:
            chunks.append((text, citation))
        elif chunks:  # glue a tiny tail onto the previous chunk
            prev_text, prev_cit = chunks[-1]
            chunks[-1] = (prev_text + " · " + text, prev_cit)
    return chunks


class Knowledge:
    """(section_text, citation) retrieval over the case documents."""

    def __init__(self, knowledge_dir: Optional[Path] = None) -> None:
        d = Path(knowledge_dir) if knowledge_dir else _KNOW_DIR
        self.chunks: list[tuple[str, str]] = []
        dossier = d / "site_dossier.md"
        rapport = d / "rapport_socio_economique.md"
        if dossier.exists():
            self.chunks += self._parse_dossier(dossier.read_text(encoding="utf-8"))
        if rapport.exists():
            self.chunks += self._parse_rapport(rapport.read_text(encoding="utf-8"))
        # precompute token sets
        self._chunk_tokens: list[set[str]] = [set(_tokens(t)) for t, _ in self.chunks]

    # ------------------------------------------------------------- parsing
    @staticmethod
    def _parse_dossier(text: str) -> list[tuple[str, str]]:
        chunks: list[tuple[str, str]] = []
        page: Optional[str] = None
        paras: list[str] = []
        cur: list[str] = []

        def flush_para() -> None:
            if cur:
                paras.append(" ".join(cur))
                cur.clear()

        def flush_page() -> None:
            nonlocal paras
            flush_para()
            if page is not None and paras:
                chunks.extend(_pack(paras, f"[site_dossier p.{page}]"))
            paras = []

        for line in text.splitlines():
            m = _PAGE_RE.match(line.strip())
            if m:
                flush_page()
                page = m.group(1)
                continue
            if page is None:
                continue
            if line.strip() in ("", "---"):
                flush_para()
            else:
                cur.append(line.strip())
        flush_page()
        return chunks

    @staticmethod
    def _parse_rapport(text: str) -> list[tuple[str, str]]:
        chunks: list[tuple[str, str]] = []
        heading: Optional[str] = None
        paras: list[str] = []
        cur: list[str] = []

        def flush_para() -> None:
            if cur:
                paras.append(" ".join(cur))
                cur.clear()

        def flush_section() -> None:
            nonlocal paras
            flush_para()
            if heading and paras:
                chunks.extend(_pack(paras, f"[rapport_socio_eco §{heading}]"))
            paras = []

        for line in text.splitlines():
            m = _HEAD_RE.match(line)
            if m:
                flush_section()
                heading = m.group(1).strip()
                continue
            if line.strip() == "":
                flush_para()
            else:
                cur.append(line.strip())
        flush_section()
        return chunks

    # ------------------------------------------------------------- lookup
    def lookup(self, query: str, k: int = 3) -> list[tuple[str, str]]:
        """Top-k (snippet, citation) by keyword score; explicit gap marker on 0."""
        q_tokens = _tokens(query)
        phrase = re.sub(r"\s+", " ", query.lower()).strip()
        scored: list[tuple[float, int]] = []
        for i, (text, _cit) in enumerate(self.chunks):
            ctoks = self._chunk_tokens[i]
            score = 0.0
            for qt in q_tokens:
                if qt in ctoks:
                    score += 1.0
                elif len(qt) >= 4 and any(
                    (c.startswith(qt) or qt.startswith(c)) and len(c) >= 4
                    for c in ctoks
                ):
                    score += 0.5  # stemming-lite: cost ~ costliest, stop ~ stoppage
            if phrase and len(phrase) > 6 and phrase in text.lower():
                score += 3.0
            if score > 0:
                scored.append((score, i))
        if not scored:
            return [(f"no dossier coverage for '{query}'", "[knowledge gap]")]
        scored.sort(key=lambda s: (-s[0], s[1]))
        out = []
        for _score, i in scored[:k]:
            text, cit = self.chunks[i]
            if len(text) > 900:
                text = text[:900] + " …"
            out.append((text, cit))
        return out

    def downtime_context(self) -> list[tuple[str, str]]:
        """The cost-of-downtime snippets from the site dossier (economics page)."""
        hits: list[tuple[str, str]] = []
        for text, cit in self.chunks:
            if "site_dossier" not in cit:
                continue
            low = text.lower()
            if ("direct loss" in low or ("cost" in low and "hour" in low)
                    or ("€" in text and ("loss" in low or "stop" in low))):
                hits.append((text if len(text) <= 900 else text[:900] + " …", cit))
        if not hits:
            return [("no dossier coverage for 'cost of downtime'", "[knowledge gap]")]
        return hits[:3]
