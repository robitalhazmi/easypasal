"""Structure-aware legal chunker and fixed-size baseline chunker for EasyPasal.

Decomposes Indonesian regulatory documents (UU, PP, POJK, SEOJK) into atomic,
semantically coherent chunks while maintaining regulatory context breadcrumbs
(Doc -> Bab -> Bagian -> Paragraf -> Pasal -> Ayat / Angka). Also provides a
fixed-size sliding window chunker as a baseline for Experiment E1.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from easypasal.ingest.parser import ParsedDocument

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("easypasal.ingest.chunker")


@dataclass
class Chunk:
    """Represents an atomic text chunk and its regulatory metadata."""

    chunk_id: str
    doc_id: str
    doc_number: str
    doc_title: str
    year: int
    issuer: str
    status: str
    source_url: str
    chunk_type: str  # "structure_aware" | "fixed_size"
    text: str  # Context-enriched text for sparse/dense retrieval
    raw_content: str  # Clean clause/article body without breadcrumb header
    bab: str | None = None
    bab_title: str | None = None
    bagian: str | None = None
    paragraf: str | None = None
    pasal: str | None = None
    ayat: str | None = None
    citation_label: str = ""
    char_count: int = 0
    token_count_est: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to a serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        """Instantiate Chunk from a dictionary."""
        return cls(**data)


def estimate_tokens(text: str) -> int:
    """Estimate token count for Indonesian text using standard character heuristic."""
    return max(1, len(text) // 4)


def split_oversized(text: str, max_chars: int = 2000, overlap: int = 200) -> list[str]:
    """Splits an oversized legal text block by paragraphs or sliding window.

    Ensures single long clauses or annexes never exceed embedding context limits
    while keeping natural paragraphs intact whenever possible.
    """
    if len(text) <= max_chars:
        return [text]

    # Try splitting by paragraph boundaries
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    chunks: list[str] = []
    curr: list[str] = []
    curr_len = 0

    for p in paragraphs:
        if curr_len + len(p) + 2 > max_chars and curr:
            chunks.append("\n\n".join(curr))
            curr = [p]
            curr_len = len(p)
        else:
            curr.append(p)
            curr_len += len(p) + 2
    if curr:
        chunks.append("\n\n".join(curr))

    final_chunks: list[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            final_chunks.append(c)
        else:
            # Sub-split oversized single paragraph with sliding window
            start = 0
            stride = max_chars - overlap
            while start < len(c):
                end = min(start + max_chars, len(c))
                final_chunks.append(c[start:end])
                if end == len(c):
                    break
                start += stride
    return final_chunks


def build_breadcrumb(
    doc_number: str,
    bab: str | None = None,
    bab_title: str | None = None,
    bagian: str | None = None,
    paragraf: str | None = None,
    pasal: str | None = None,
    ayat: str | None = None,
    angka: str | None = None,
    is_penjelasan: bool = False,
) -> str:
    """Constructs a statutory breadcrumb header for Indonesian regulations."""
    parts: list[str] = []
    prefix = f"Penjelasan {doc_number}" if is_penjelasan else doc_number
    parts.append(prefix)

    if bab:
        b_str = f"{bab} {bab_title}".strip() if bab_title else bab
        parts.append(b_str)
    if bagian:
        parts.append(bagian)
    if paragraf:
        parts.append(paragraf)

    if pasal and ayat:
        parts.append(f"Pasal {pasal} ayat ({ayat})")
    elif pasal and angka:
        parts.append(f"Pasal {pasal} angka {angka}")
    elif pasal:
        parts.append(f"Pasal {pasal}")

    return " | ".join(parts)


def is_catchword(text: str) -> bool:
    """Detects whether a text snippet is a page-turn catchword artifact from PDF layout."""
    clean = text.strip()
    if len(clean) < 35 and clean.endswith("..."):
        return True
    return bool(re.match(r"^\(\d+[a-z]?\)\s+[A-Za-z0-9]+\.{2,}$", clean))


class StructureAwareChunker:
    """Decomposes Indonesian regulations into hierarchy-aware, context-enriched chunks."""

    def __init__(
        self,
        max_chunk_size: int = 2000,
        overlap: int = 200,
        include_breadcrumbs: bool = True,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.include_breadcrumbs = include_breadcrumbs
        self.penjelasan_re = re.compile(
            r"\n(?=PENJELASAN\s+(?:ATAS|UNDANG-UNDANG|PERATURAN)\b|\n\s*PENJELASAN\s*\n)",
            re.IGNORECASE,
        )
        self.heading_re = re.compile(
            r"\n(?=(?:BAB\s+[IVXLCDM]+|Bagian\s+[A-Za-z]+|Paragraf\s+\d+|[\"'\u201c\u201d]?\s*Pasal\s+(?:[0-9]+[A-Za-z]*|[IVXLCDM]+))\s*:?[\"'\u201c\u201d]?$)",
            re.MULTILINE,
        )
        self.penj_heading_re = re.compile(
            r"\n(?=(?:[IVXLCDM]+\.\s+[A-Z\s]+|[\"'\u201c\u201d]?\s*Pasal\s+(?:[0-9]+[A-Za-z]*|[IVXLCDM]+))\s*:?[\"'\u201c\u201d]?$)",
            re.MULTILINE,
        )

    def _format_chunk_text(self, breadcrumb: str, body: str) -> str:
        """Format chunk text with or without statutory breadcrumbs."""
        if not self.include_breadcrumbs or not breadcrumb:
            return body if body else breadcrumb
        return f"[{breadcrumb}]\n{body}" if body else f"[{breadcrumb}]"

    def chunk(self, doc: ParsedDocument | dict[str, Any]) -> list[Chunk]:
        """Process a parsed document into structure-aware chunks."""
        doc_id = doc.doc_id if isinstance(doc, ParsedDocument) else doc["doc_id"]
        meta = doc.metadata if isinstance(doc, ParsedDocument) else doc.get("metadata", {})
        doc_number = meta.get("doc_number", doc_id)
        doc_title = meta.get("title", "")
        year_val = meta.get("year", 0)
        try:
            year = int(year_val)
        except (ValueError, TypeError):
            year = 0
        issuer = meta.get("issuer", "")
        status = meta.get("status", "berlaku")
        source_url = meta.get("url", "")
        clean_text = (
            doc.clean_text if isinstance(doc, ParsedDocument) else doc.get("clean_text", "")
        )

        if not clean_text:
            return []

        # Normalize OCR lowercase 'l.' at line start in definition lists to digit '1.'
        clean_text = re.sub(r"(?<=\n)l\.\s+", "1. ", clean_text)

        if doc_id.startswith("SEOJK") or doc_number.startswith("SEOJK"):
            raw_chunks = self._chunk_seojk(
                doc_id, doc_number, doc_title, year, issuer, status, source_url, clean_text
            )
        else:
            raw_chunks = self._chunk_statute(
                doc_id, doc_number, doc_title, year, issuer, status, source_url, clean_text
            )

        # Ensure deterministic, collision-free unique chunk IDs
        final_chunks: list[Chunk] = []
        seen_ids: dict[str, int] = {}
        for c in raw_chunks:
            cid = c.chunk_id
            if cid in seen_ids:
                count = seen_ids[cid] + 1
                seen_ids[cid] = count
                c.chunk_id = f"{cid}_{count}"
            else:
                seen_ids[cid] = 1
            final_chunks.append(c)

        return final_chunks

    def _chunk_seojk(
        self,
        doc_id: str,
        doc_number: str,
        doc_title: str,
        year: int,
        issuer: str,
        status: str,
        source_url: str,
        text: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        sec_splits = re.split(r"\n(?=[IVXLCDM]+\.\s+)", text)

        # Process Preamble/Opening
        preamble = sec_splits[0].strip()
        if len(preamble) > 50:
            for i, p_part in enumerate(
                split_oversized(preamble, self.max_chunk_size, self.overlap)
            ):
                suffix = f"_part_{i + 1}" if i > 0 else ""
                cit_suffix = f" (bagian {i + 1})" if i > 0 else ""
                c_text = f"[{doc_number} | Pembukaan]\n{p_part}"
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}_pembukaan{suffix}",
                        doc_id=doc_id,
                        doc_number=doc_number,
                        doc_title=doc_title,
                        year=year,
                        issuer=issuer,
                        status=status,
                        source_url=source_url,
                        chunk_type="structure_aware",
                        text=c_text,
                        raw_content=p_part,
                        citation_label=f"{doc_number}, Pembukaan{cit_suffix}",
                        char_count=len(c_text),
                        token_count_est=estimate_tokens(c_text),
                    )
                )

        # Process Roman sections
        for sec in sec_splits[1:]:
            lines = sec.strip().splitlines()
            if not lines:
                continue
            first_line = lines[0]
            m_rom = re.match(r"^([IVXLCDM]+)\.\s*(.*)", first_line)
            roman = m_rom.group(1) if m_rom else "I"
            roman_title = m_rom.group(2).strip() if m_rom else ""

            body = "\n".join(lines[1:]).strip()
            items = re.split(r"\n(?=\d+\.\s+)", "\n" + body)
            item_list = [it.strip() for it in items if it.strip()]

            if len(item_list) > 1 or (item_list and re.match(r"^\d+\.", item_list[0])):
                for it in item_list:
                    if is_catchword(it):
                        continue
                    m_num = re.match(r"^(\d+)\.", it)
                    num_str = m_num.group(1) if m_num else "1"
                    for idx, part_text in enumerate(
                        split_oversized(it, self.max_chunk_size, self.overlap)
                    ):
                        suffix = f"_part_{idx + 1}" if idx > 0 else ""
                        cit_suffix = f" (bagian {idx + 1})" if idx > 0 else ""
                        cit = f"{doc_number}, Romawi {roman} angka {num_str}{cit_suffix}"
                        cid = f"{doc_id}_romawi_{roman}_angka_{num_str}{suffix}"
                        c_hdr = f"{doc_number} | Romawi {roman} {roman_title} | Angka {num_str}"
                        c_text = self._format_chunk_text(c_hdr, part_text)
                        chunks.append(
                            Chunk(
                                chunk_id=cid,
                                doc_id=doc_id,
                                doc_number=doc_number,
                                doc_title=doc_title,
                                year=year,
                                issuer=issuer,
                                status=status,
                                source_url=source_url,
                                chunk_type="structure_aware",
                                text=c_text,
                                raw_content=part_text,
                                bab=f"Romawi {roman}",
                                bab_title=roman_title,
                                citation_label=cit,
                                char_count=len(c_text),
                                token_count_est=estimate_tokens(c_text),
                            )
                        )
            else:
                for idx, part_text in enumerate(
                    split_oversized(body, self.max_chunk_size, self.overlap)
                ):
                    suffix = f"_part_{idx + 1}" if idx > 0 else ""
                    cit_suffix = f" (bagian {idx + 1})" if idx > 0 else ""
                    cit = f"{doc_number}, Romawi {roman}{cit_suffix}"
                    cid = f"{doc_id}_romawi_{roman}{suffix}"
                    c_hdr = f"{doc_number} | Romawi {roman} {roman_title}"
                    c_text = self._format_chunk_text(c_hdr, part_text)
                    chunks.append(
                        Chunk(
                            chunk_id=cid,
                            doc_id=doc_id,
                            doc_number=doc_number,
                            doc_title=doc_title,
                            year=year,
                            issuer=issuer,
                            status=status,
                            source_url=source_url,
                            chunk_type="structure_aware",
                            text=c_text,
                            raw_content=part_text,
                            bab=f"Romawi {roman}",
                            bab_title=roman_title,
                            citation_label=cit,
                            char_count=len(c_text),
                            token_count_est=estimate_tokens(c_text),
                        )
                    )

        return chunks

    def _chunk_statute(
        self,
        doc_id: str,
        doc_number: str,
        doc_title: str,
        year: int,
        issuer: str,
        status: str,
        source_url: str,
        text: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []

        penj_match = self.penjelasan_re.search(text)
        if penj_match:
            body_text = text[: penj_match.start()]
            penj_text = text[penj_match.start() :]
        else:
            body_text = text
            penj_text = ""

        parts = self.heading_re.split(body_text)

        # Process Preamble (Menimbang, Mengingat, Memutuskan)
        preamble = parts[0].strip() if parts else ""
        if len(preamble) > 50:
            for idx, p_part in enumerate(
                split_oversized(preamble, self.max_chunk_size, self.overlap)
            ):
                suffix = f"_part_{idx + 1}" if idx > 0 else ""
                cit_suffix = f" (bagian {idx + 1})" if idx > 0 else ""
                c_text = self._format_chunk_text(f"{doc_number} | Pembukaan", p_part)
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}_pembukaan{suffix}",
                        doc_id=doc_id,
                        doc_number=doc_number,
                        doc_title=doc_title,
                        year=year,
                        issuer=issuer,
                        status=status,
                        source_url=source_url,
                        chunk_type="structure_aware",
                        text=c_text,
                        raw_content=p_part,
                        bab="PEMBUKAAN",
                        bab_title="Dasar Hukum dan Pertimbangan",
                        citation_label=f"{doc_number}, Pembukaan{cit_suffix}",
                        char_count=len(c_text),
                        token_count_est=estimate_tokens(c_text),
                    )
                )

        cur_bab: str | None = None
        cur_bab_title: str | None = None
        cur_bagian: str | None = None
        cur_paragraf: str | None = None

        for part in parts[1:]:
            s = part.strip()
            if not s:
                continue
            lines = s.splitlines()
            first_line = lines[0].strip()

            m_bab = re.match(r"^BAB\s+([IVXLCDM]+)$", first_line, re.IGNORECASE)
            if m_bab:
                cur_bab = f"BAB {m_bab.group(1).upper()}"
                cur_bab_title = " ".join(line.strip() for line in lines[1:] if line.strip()) or None
                cur_bagian = None
                cur_paragraf = None
                continue

            m_bagian = re.match(r"^Bagian\s+([A-Za-z]+)$", first_line, re.IGNORECASE)
            if m_bagian:
                cur_bagian = f"Bagian {m_bagian.group(1).capitalize()}"
                cur_paragraf = None
                continue

            m_paragraf = re.match(r"^Paragraf\s+(\d+)$", first_line, re.IGNORECASE)
            if m_paragraf:
                cur_paragraf = f"Paragraf {m_paragraf.group(1)}"
                continue

            m_pasal = re.match(
                r"^[\"'\u201c\u201d]?\s*Pasal\s+([0-9]+[A-Za-z]*|[IVXLCDM]+)\s*:?[\"'\u201c\u201d]?$",
                first_line,
                re.IGNORECASE,
            )
            if m_pasal:
                pasal_num = m_pasal.group(1)
                pasal_body = "\n".join(lines[1:]).strip()
                # If this is an amendment article, strip any trailing introductory clause for the next amendment point
                pasal_body = re.sub(
                    r"\n+\d+\.\s*(?:Ketentuan|Menambah|Di antara|Mengubah|Menghapus)\b.*$",
                    "",
                    pasal_body,
                    flags=re.DOTALL,
                ).strip()
                pasal_chunks = self._chunk_pasal(
                    doc_id=doc_id,
                    doc_number=doc_number,
                    doc_title=doc_title,
                    year=year,
                    issuer=issuer,
                    status=status,
                    source_url=source_url,
                    pasal_num=pasal_num,
                    pasal_body=pasal_body,
                    cur_bab=cur_bab,
                    cur_bab_title=cur_bab_title,
                    cur_bagian=cur_bagian,
                    cur_paragraf=cur_paragraf,
                    is_penjelasan=False,
                )
                chunks.extend(pasal_chunks)

        if penj_text:
            penj_chunks = self._chunk_penjelasan(
                doc_id=doc_id,
                doc_number=doc_number,
                doc_title=doc_title,
                year=year,
                issuer=issuer,
                status=status,
                source_url=source_url,
                penj_text=penj_text,
            )
            chunks.extend(penj_chunks)

        return chunks

    def _chunk_pasal(
        self,
        doc_id: str,
        doc_number: str,
        doc_title: str,
        year: int,
        issuer: str,
        status: str,
        source_url: str,
        pasal_num: str,
        pasal_body: str,
        cur_bab: str | None,
        cur_bab_title: str | None,
        cur_bagian: str | None,
        cur_paragraf: str | None,
        is_penjelasan: bool,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []

        # Normalize OCR clause numbering artifacts:
        # e.g., missing closing paren '(21 ' -> '(2) ', or clauses preceded by punctuation without newline
        pasal_body = re.sub(r"(?<=[.;\n])\s*\(([0-9]{1,2})[1l|\]]\s+", r"\n(\1) ", pasal_body)
        pasal_body = re.sub(r"(?<=[.;])\s*\(([0-9]{1,2}[a-z]?)\)\s+", r"\n(\1) ", pasal_body)

        # Check for clauses (1), (2)
        has_clauses = bool(re.search(r"\n\(\d+[a-z]?\)\s+", "\n" + pasal_body))
        if has_clauses:
            clause_parts = re.split(r"\n(?=\(\d+[a-z]?\)\s+)", "\n" + pasal_body)
            for cp in clause_parts:
                cp_clean = cp.strip()
                if not cp_clean or is_catchword(cp_clean):
                    continue
                m_ayat = re.match(r"^\((\d+[a-z]?)\)", cp_clean)
                if m_ayat:
                    ayat_num = m_ayat.group(1)
                    bc = build_breadcrumb(
                        doc_number=doc_number,
                        bab=cur_bab,
                        bab_title=cur_bab_title,
                        bagian=cur_bagian,
                        paragraf=cur_paragraf,
                        pasal=pasal_num,
                        ayat=ayat_num,
                        is_penjelasan=is_penjelasan,
                    )
                    prefix = "penjelasan_" if is_penjelasan else ""
                    cit_prefix = f"Penjelasan {doc_number}" if is_penjelasan else doc_number
                    for idx, c_split in enumerate(
                        split_oversized(cp_clean, self.max_chunk_size, self.overlap)
                    ):
                        suffix = f"_part_{idx + 1}" if idx > 0 else ""
                        cit_suffix = f" (bagian {idx + 1})" if idx > 0 else ""
                        cid = f"{doc_id}_{prefix}pasal_{pasal_num}_ayat_{ayat_num}{suffix}"
                        cit = f"{cit_prefix}, Pasal {pasal_num} ayat ({ayat_num}){cit_suffix}"
                        c_text = self._format_chunk_text(bc, c_split)
                        chunks.append(
                            Chunk(
                                chunk_id=cid,
                                doc_id=doc_id,
                                doc_number=doc_number,
                                doc_title=doc_title,
                                year=year,
                                issuer=issuer,
                                status=status,
                                source_url=source_url,
                                chunk_type="structure_aware",
                                text=c_text,
                                raw_content=c_split,
                                bab=cur_bab,
                                bab_title=cur_bab_title,
                                bagian=cur_bagian,
                                paragraf=cur_paragraf,
                                pasal=pasal_num,
                                ayat=ayat_num,
                                citation_label=cit,
                                char_count=len(c_text),
                                token_count_est=estimate_tokens(c_text),
                            )
                        )
            return chunks

        # Check for definition items: 1. , 2.
        has_definitions = bool(re.search(r"^\s*1\.\s+", pasal_body, re.MULTILINE))
        if has_definitions:
            def_parts = re.split(r"\n(?=\d+\.\s+)", "\n" + pasal_body)
            for dp in def_parts:
                dp_clean = dp.strip()
                if not dp_clean or is_catchword(dp_clean):
                    continue
                m_angka = re.match(r"^(\d+)\.", dp_clean)
                if m_angka:
                    angka_num = m_angka.group(1)
                    bc = build_breadcrumb(
                        doc_number=doc_number,
                        bab=cur_bab,
                        bab_title=cur_bab_title,
                        bagian=cur_bagian,
                        paragraf=cur_paragraf,
                        pasal=pasal_num,
                        angka=angka_num,
                        is_penjelasan=is_penjelasan,
                    )
                    prefix = "penjelasan_" if is_penjelasan else ""
                    cit_prefix = f"Penjelasan {doc_number}" if is_penjelasan else doc_number
                    for idx, d_split in enumerate(
                        split_oversized(dp_clean, self.max_chunk_size, self.overlap)
                    ):
                        suffix = f"_part_{idx + 1}" if idx > 0 else ""
                        cit_suffix = f" (bagian {idx + 1})" if idx > 0 else ""
                        cid = f"{doc_id}_{prefix}pasal_{pasal_num}_angka_{angka_num}{suffix}"
                        cit = f"{cit_prefix}, Pasal {pasal_num} angka {angka_num}{cit_suffix}"
                        c_text = self._format_chunk_text(bc, d_split)
                        chunks.append(
                            Chunk(
                                chunk_id=cid,
                                doc_id=doc_id,
                                doc_number=doc_number,
                                doc_title=doc_title,
                                year=year,
                                issuer=issuer,
                                status=status,
                                source_url=source_url,
                                chunk_type="structure_aware",
                                text=c_text,
                                raw_content=d_split,
                                bab=cur_bab,
                                bab_title=cur_bab_title,
                                bagian=cur_bagian,
                                paragraf=cur_paragraf,
                                pasal=pasal_num,
                                ayat=None,
                                citation_label=cit,
                                char_count=len(c_text),
                                token_count_est=estimate_tokens(c_text),
                            )
                        )
            if chunks:
                return chunks

        # Single unnumbered clause
        bc = build_breadcrumb(
            doc_number=doc_number,
            bab=cur_bab,
            bab_title=cur_bab_title,
            bagian=cur_bagian,
            paragraf=cur_paragraf,
            pasal=pasal_num,
            is_penjelasan=is_penjelasan,
        )
        prefix = "penjelasan_" if is_penjelasan else ""
        cit_prefix = f"Penjelasan {doc_number}" if is_penjelasan else doc_number
        for idx, u_split in enumerate(
            split_oversized(pasal_body, self.max_chunk_size, self.overlap)
        ):
            suffix = f"_part_{idx + 1}" if idx > 0 else ""
            cit_suffix = f" (bagian {idx + 1})" if idx > 0 else ""
            cid = f"{doc_id}_{prefix}pasal_{pasal_num}{suffix}"
            cit = f"{cit_prefix}, Pasal {pasal_num}{cit_suffix}"
            c_text = self._format_chunk_text(bc, u_split)
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=doc_id,
                    doc_number=doc_number,
                    doc_title=doc_title,
                    year=year,
                    issuer=issuer,
                    status=status,
                    source_url=source_url,
                    chunk_type="structure_aware",
                    text=c_text,
                    raw_content=u_split,
                    bab=cur_bab,
                    bab_title=cur_bab_title,
                    bagian=cur_bagian,
                    paragraf=cur_paragraf,
                    pasal=pasal_num,
                    ayat=None,
                    citation_label=cit,
                    char_count=len(c_text),
                    token_count_est=estimate_tokens(c_text),
                )
            )
        return chunks

    def _chunk_penjelasan(
        self,
        doc_id: str,
        doc_number: str,
        doc_title: str,
        year: int,
        issuer: str,
        status: str,
        source_url: str,
        penj_text: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        parts = self.penj_heading_re.split(penj_text)

        for part in parts:
            s = part.strip()
            if not s:
                continue
            lines = s.splitlines()
            first_line = lines[0].strip()

            if re.match(r"^I\.\s*UMUM", first_line, re.IGNORECASE):
                body = "\n".join(lines[1:]).strip()
                for idx, u_split in enumerate(
                    split_oversized(body, self.max_chunk_size, self.overlap)
                ):
                    suffix = f"_part_{idx + 1}" if idx > 0 else ""
                    cit_suffix = f" (bagian {idx + 1})" if idx > 0 else ""
                    c_text = self._format_chunk_text(f"Penjelasan {doc_number} | I. Umum", u_split)
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc_id}_penjelasan_umum{suffix}",
                            doc_id=doc_id,
                            doc_number=doc_number,
                            doc_title=doc_title,
                            year=year,
                            issuer=issuer,
                            status=status,
                            source_url=source_url,
                            chunk_type="structure_aware",
                            text=c_text,
                            raw_content=u_split,
                            bab="PENJELASAN",
                            bab_title="I. UMUM",
                            citation_label=f"Penjelasan {doc_number}, I. Umum{cit_suffix}",
                            char_count=len(c_text),
                            token_count_est=estimate_tokens(c_text),
                        )
                    )
                continue

            m_pasal = re.match(
                r"^[\"'\u201c\u201d]?\s*Pasal\s+([0-9]+[A-Za-z]*|[IVXLCDM]+)\s*:?[\"'\u201c\u201d]?$",
                first_line,
                re.IGNORECASE,
            )
            if m_pasal:
                pasal_num = m_pasal.group(1)
                body = "\n".join(lines[1:]).strip()
                # Normalize OCR clause numbering artifacts in explanation sections
                body = re.sub(
                    r"(?:^|\n)\s*Ayat\s+\(([0-9]{1,2})[1l|\]]\b",
                    r"\nAyat (\1)",
                    body,
                    flags=re.IGNORECASE,
                )
                body = re.sub(
                    r"(?<=[.;])\s*Ayat\s+\(([0-9]{1,2}[a-z]?)\)",
                    r"\nAyat (\1)",
                    body,
                    flags=re.IGNORECASE,
                )
                has_ayats = bool(re.search(r"(?:^|\n)Ayat\s+\((\d+[a-z]?)\)", body, re.IGNORECASE))
                if has_ayats:
                    ayat_parts = re.split(
                        r"\n(?=Ayat\s+\(\d+[a-z]?\))", "\n" + body, flags=re.IGNORECASE
                    )
                    for ap in ayat_parts:
                        ap_clean = ap.strip()
                        if not ap_clean or is_catchword(ap_clean):
                            continue
                        m_a = re.search(r"Ayat\s+\((\d+[a-z]?)\)", ap_clean, re.IGNORECASE)
                        if m_a:
                            ayat_num = m_a.group(1)
                            for idx, a_split in enumerate(
                                split_oversized(ap_clean, self.max_chunk_size, self.overlap)
                            ):
                                suffix = f"_part_{idx + 1}" if idx > 0 else ""
                                cit_suffix = f" (bagian {idx + 1})" if idx > 0 else ""
                                cid = (
                                    f"{doc_id}_penjelasan_pasal_{pasal_num}_ayat_{ayat_num}{suffix}"
                                )
                                cit = f"Penjelasan {doc_number}, Pasal {pasal_num} ayat ({ayat_num}){cit_suffix}"
                                bc = (
                                    f"Penjelasan {doc_number} | Pasal {pasal_num} ayat ({ayat_num})"
                                )
                                c_text = self._format_chunk_text(bc, a_split)
                                chunks.append(
                                    Chunk(
                                        chunk_id=cid,
                                        doc_id=doc_id,
                                        doc_number=doc_number,
                                        doc_title=doc_title,
                                        year=year,
                                        issuer=issuer,
                                        status=status,
                                        source_url=source_url,
                                        chunk_type="structure_aware",
                                        text=c_text,
                                        raw_content=a_split,
                                        bab="PENJELASAN",
                                        pasal=pasal_num,
                                        ayat=ayat_num,
                                        citation_label=cit,
                                        char_count=len(c_text),
                                        token_count_est=estimate_tokens(c_text),
                                    )
                                )
                else:
                    for idx, b_split in enumerate(
                        split_oversized(body, self.max_chunk_size, self.overlap)
                    ):
                        suffix = f"_part_{idx + 1}" if idx > 0 else ""
                        cit_suffix = f" (bagian {idx + 1})" if idx > 0 else ""
                        cid = f"{doc_id}_penjelasan_pasal_{pasal_num}{suffix}"
                        cit = f"Penjelasan {doc_number}, Pasal {pasal_num}{cit_suffix}"
                        bc = f"Penjelasan {doc_number} | Pasal {pasal_num}"
                        c_text = self._format_chunk_text(bc, b_split)
                        chunks.append(
                            Chunk(
                                chunk_id=cid,
                                doc_id=doc_id,
                                doc_number=doc_number,
                                doc_title=doc_title,
                                year=year,
                                issuer=issuer,
                                status=status,
                                source_url=source_url,
                                chunk_type="structure_aware",
                                text=c_text,
                                raw_content=b_split,
                                bab="PENJELASAN",
                                pasal=pasal_num,
                                ayat=None,
                                citation_label=cit,
                                char_count=len(c_text),
                                token_count_est=estimate_tokens(c_text),
                            )
                        )

        return chunks


class FixedSizeChunker:
    """Sliding-window character chunker baseline for Experiment E1."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, doc: ParsedDocument | dict[str, Any]) -> list[Chunk]:
        """Split document text into fixed-size sliding windows."""
        doc_id = doc.doc_id if isinstance(doc, ParsedDocument) else doc["doc_id"]
        meta = doc.metadata if isinstance(doc, ParsedDocument) else doc.get("metadata", {})
        doc_number = meta.get("doc_number", doc_id)
        doc_title = meta.get("title", "")
        year_val = meta.get("year", 0)
        try:
            year = int(year_val)
        except (ValueError, TypeError):
            year = 0
        issuer = meta.get("issuer", "")
        status = meta.get("status", "berlaku")
        source_url = meta.get("url", "")
        clean_text = (
            doc.clean_text if isinstance(doc, ParsedDocument) else doc.get("clean_text", "")
        )

        if not clean_text:
            return []

        stride = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(clean_text):
            end = min(start + self.chunk_size, len(clean_text))
            window = clean_text[start:end]
            if window.strip():
                idx += 1
                cid = f"{doc_id}_fix_{idx:04d}"
                cit = f"{doc_number}, Chunk #{idx}"
                chunks.append(
                    Chunk(
                        chunk_id=cid,
                        doc_id=doc_id,
                        doc_number=doc_number,
                        doc_title=doc_title,
                        year=year,
                        issuer=issuer,
                        status=status,
                        source_url=source_url,
                        chunk_type="fixed_size",
                        text=window,
                        raw_content=window,
                        citation_label=cit,
                        char_count=len(window),
                        token_count_est=estimate_tokens(window),
                    )
                )
            if end >= len(clean_text):
                break
            start += stride

        return chunks


def chunk_document(
    doc: ParsedDocument | dict[str, Any],
    strategy: str = "structure_aware",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    max_chunk_size: int = 2000,
    include_breadcrumbs: bool = True,
) -> list[Chunk]:
    """Chunk a single parsed document using the specified strategy."""
    if strategy in ("structure_aware", "structure_aware_enriched"):
        chunker = StructureAwareChunker(
            max_chunk_size=max_chunk_size,
            overlap=chunk_overlap,
            include_breadcrumbs=include_breadcrumbs,
        )
        return chunker.chunk(doc)
    elif strategy == "structure_aware_no_header":
        chunker = StructureAwareChunker(
            max_chunk_size=max_chunk_size,
            overlap=chunk_overlap,
            include_breadcrumbs=False,
        )
        return chunker.chunk(doc)
    elif strategy == "fixed_size":
        baseline_chunker = FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return baseline_chunker.chunk(doc)
    else:
        raise ValueError(
            f"Unsupported chunking strategy: {strategy!r}. Must be 'structure_aware', 'structure_aware_no_header', or 'fixed_size'."
        )


def chunk_corpus(
    input_dir: Path | str = "data/parsed",
    output_dir: Path | str = "data/chunks",
    strategy: str = "structure_aware",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    max_chunk_size: int = 2000,
) -> list[Chunk]:
    """Process all parsed documents in input_dir and export chunks to JSONL."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_path.glob("*.json"))
    if not json_files:
        logger.warning("No parsed JSON documents found in %s", input_path)
        return []

    logger.info(
        "Chunking %d documents using strategy=%r (output_dir=%s)...",
        len(json_files),
        strategy,
        output_path,
    )

    all_chunks: list[Chunk] = []
    for file_path in json_files:
        try:
            with open(file_path, encoding="utf-8") as f:
                doc_dict = json.load(f)
            doc_chunks = chunk_document(
                doc_dict,
                strategy=strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_chunk_size=max_chunk_size,
            )
            all_chunks.extend(doc_chunks)
        except Exception:
            logger.exception("Failed to chunk document %s", file_path)

    output_file = output_path / f"{strategy}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    logger.info(
        "Successfully wrote %d %s chunks to %s (%.2f MB)",
        len(all_chunks),
        strategy,
        output_file,
        output_file.stat().st_size / (1024 * 1024),
    )
    return all_chunks


def main() -> None:
    """CLI runner for chunking the EasyPasal regulatory corpus."""
    parser = argparse.ArgumentParser(
        description="Chunk parsed Indonesian regulatory documents into structure-aware or fixed-size JSONL."
    )
    parser.add_argument(
        "--strategy",
        choices=["structure_aware", "structure_aware_no_header", "fixed_size", "all"],
        default="structure_aware",
        help="Chunking strategy to execute (default: structure_aware)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/parsed"),
        help="Directory containing parsed document JSON files (default: data/parsed)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/chunks"),
        help="Directory to save generated JSONL files (default: data/chunks)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk character size for fixed_size strategy (default: 500)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=100,
        help="Chunk character overlap for fixed_size strategy (default: 100)",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=2000,
        help="Max character size before splitting for structure_aware strategy (default: 2000)",
    )

    args = parser.parse_args()

    if args.strategy in ("structure_aware", "all"):
        chunk_corpus(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            strategy="structure_aware",
            max_chunk_size=args.max_chunk_size,
        )

    if args.strategy in ("structure_aware_no_header", "all"):
        chunk_corpus(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            strategy="structure_aware_no_header",
            max_chunk_size=args.max_chunk_size,
        )

    if args.strategy in ("fixed_size", "all"):
        chunk_corpus(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            strategy="fixed_size",
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
        )


if __name__ == "__main__":
    main()
