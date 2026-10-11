"""PDF layout parsing and text cleaning module for Indonesian regulatory documents.

Extracts text from PDF files using PyMuPDF, strips running headers/footers,
normalizes line breaks and whitespace, resolves Indonesian hyphenation splits,
and preserves legal structural hierarchy (Bab, Bagian, Paragraf, Pasal, Ayat).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("easypasal.ingest.parser")

KNOWN_COMPOUNDS: set[tuple[str, str]] = {
    ("perundang", "undangan"),
    ("tata", "tertib"),
    ("serah", "terima"),
    ("tanya", "jawab"),
    ("hak", "kewajiban"),
    ("surat", "menyurat"),
    ("tanggung", "jawab"),
}

# Regex to match structural legal headings that must be preserved on their own lines
HEADING_RE = re.compile(
    r"^(BAB\s+[IVXLCDM]+|"
    r"Bagian\s+[A-Za-z]+|"
    r"Paragraf\s+\d+|"
    r"Pasal\s+\d+|"
    r"\(\d+\)|"
    r"[a-z]\.|"
    r"\d+\.|"
    r"MEMUTUSKAN:|"
    r"Menetapkan\s*:|"
    r"Menimbang\s*:|"
    r"Mengingat\s*:)",
    re.IGNORECASE,
)

STANDALONE_HEADING_RE = re.compile(
    r"^(BAB\s+[IVXLCDM]+|" r"Bagian\s+[A-Za-z]+|" r"Paragraf\s+\d+|" r"Pasal\s+\d+)$",
    re.IGNORECASE,
)


@dataclass
class ParsedPage:
    """Represents text and layout metadata extracted from a single PDF page."""

    page_number: int  # 1-indexed
    raw_text: str
    clean_text: str
    char_count: int
    block_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert page to serializable dictionary."""
        return asdict(self)


@dataclass
class ParsedDocument:
    """Represents a fully extracted regulatory document."""

    doc_id: str
    filename: str
    total_pages: int
    pages: list[ParsedPage] = field(default_factory=list)
    clean_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert document to serializable dictionary."""
        return asdict(self)


def dehyphenate_text(text: str) -> str:
    """Join words split across line breaks while preserving reduplications and compounds."""

    def repl(m: re.Match[str]) -> str:
        w1 = m.group(1)
        w2 = m.group(2)
        if w1.lower() == w2.lower() or (w1.lower(), w2.lower()) in KNOWN_COMPOUNDS:
            return f"{w1}-{w2}"
        return f"{w1}{w2}"

    return re.sub(r"(\b[A-Za-z0-9]+)-\s*\n\s*([A-Za-z0-9]+\b)", repl, text)


def normalize_whitespace(text: str) -> str:
    """Normalize non-breaking spaces and excessive inline spaces."""
    # Replace non-breaking spaces
    text = text.replace("\u00a0", " ").replace("\ufeff", "")
    # Normalize multiple spaces per line to a single space
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(lines)


def is_header_block(text: str, y1: float, page_height: float, page_idx: int) -> bool:
    """Check if a text block located near the top of the page is a running header."""
    if page_idx == 0:
        # Never strip title headings on the first page
        return False

    cleaned = text.strip()
    if not cleaned:
        return True

    # Page number formats: "- 2 -", "-2 -", "- 2-", "-2-", "2", "Halaman 2"
    is_page_number = bool(
        re.match(r"^-\s*\d+\s*-?$", cleaned)
        or re.match(r"^\d+$", cleaned)
        or re.match(r"^[Hh]alaman\s+\d+", cleaned)
    )
    if is_page_number and y1 <= page_height * 0.32:
        return True

    if y1 > page_height * 0.25:
        return False

    # Gazette running header: "2022, No.5/OJK \n-2-"
    if re.match(r"^\d{4},\s*No\.\s*\d+", cleaned, re.IGNORECASE):
        return True

    # Institutional running headers
    return bool(
        re.match(
            r"^(salinan|presiden republik indonesia|otoritas jasa keuangan|"
            r"menteri keuangan republik indonesia)$",
            cleaned,
            re.IGNORECASE,
        )
    )


def is_footer_block(text: str, y0: float, page_height: float) -> bool:
    """Check if a text block located near the bottom of the page is a running footer."""
    if y0 < page_height * 0.85:
        return False

    cleaned = text.strip()
    if not cleaned:
        return True

    # Page number formats: "- 2 -", "2"
    if re.match(r"^-\s*\d+\s*-$", cleaned) or re.match(r"^\d+$", cleaned):
        return True

    # Security serial stamp: "SK No 017000 A"
    if re.match(r"^SK\s*No\.?\s*[0-9A-Z\s]+$", cleaned, re.IGNORECASE):
        return True

    # Website watermark / link
    if re.search(r"www\.[a-z0-9\.-]+", cleaned, re.IGNORECASE):
        return True

    # Catchwords (kata alihan): "5. Prosesor . . ." or "8. Informasi..."
    return bool(re.match(r"^\d*\.?\s*[\w\(\)\-]+\s*\.\s*\.\s*\.?$", cleaned))


def unwrap_legal_text(raw_text: str) -> str:
    """Reconstruct wrapped lines into natural clauses while preserving legal hierarchy."""
    lines = [line.strip() for line in raw_text.splitlines()]
    out: list[str] = []
    current_line = ""

    for line in lines:
        if not line:
            if current_line:
                out.append(current_line)
                current_line = ""
            continue

        if HEADING_RE.match(line):
            if current_line:
                out.append(current_line)
                current_line = ""
            current_line = line
            if STANDALONE_HEADING_RE.match(line):
                out.append(current_line)
                current_line = ""
        else:
            if not current_line:
                current_line = line
            else:
                if current_line.endswith(":"):
                    out.append(current_line)
                    current_line = line
                else:
                    current_line = f"{current_line} {line}"

    if current_line:
        out.append(current_line)

    return "\n".join(out)


def clean_page_text(page: pymupdf.Page, page_idx: int) -> tuple[str, str, int]:
    """Extract and clean text blocks from a page, returning (raw_text, clean_text, block_count)."""
    blocks = page.get_text("blocks")
    page_height = page.rect.height

    raw_blocks: list[str] = []
    kept_blocks: list[str] = []
    text_block_count = 0

    for b in blocks:
        # Block tuple: (x0, y0, x1, y1, text, block_no, block_type)
        if b[6] != 0:  # Skip image blocks
            continue

        block_text = b[4].strip()
        if not block_text:
            continue

        text_block_count += 1
        raw_blocks.append(b[4])

        y0, y1 = b[1], b[3]
        if is_header_block(block_text, y1, page_height, page_idx):
            continue
        if is_footer_block(block_text, y0, page_height):
            continue

        kept_blocks.append(b[4])

    raw_text = "\n\n".join(raw_blocks)
    intermediate_text = "\n\n".join(kept_blocks)

    # Apply dehyphenation across line wraps
    dehyphenated = dehyphenate_text(intermediate_text)

    # Reconstruct wrapped lines into coherent clauses
    unwrapped = unwrap_legal_text(dehyphenated)

    # Final whitespace cleanup
    clean_text = normalize_whitespace(unwrapped)

    return raw_text, clean_text, text_block_count


def parse_pdf(file_path: Path, doc_id: str | None = None) -> ParsedDocument:
    """Parse a single PDF file and extract cleaned text per page."""
    if not file_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    actual_doc_id = doc_id or file_path.stem
    doc = pymupdf.open(str(file_path))

    parsed_pages: list[ParsedPage] = []
    all_clean_chunks: list[str] = []

    try:
        total_pages = len(doc)
        for idx in range(total_pages):
            page = doc[idx]
            raw_text, clean_text, block_count = clean_page_text(page, page_idx=idx)

            parsed_page = ParsedPage(
                page_number=idx + 1,
                raw_text=raw_text,
                clean_text=clean_text,
                char_count=len(clean_text),
                block_count=block_count,
            )
            parsed_pages.append(parsed_page)
            if clean_text:
                all_clean_chunks.append(clean_text)

        full_clean_text = "\n\n".join(all_clean_chunks)
        return ParsedDocument(
            doc_id=actual_doc_id,
            filename=file_path.name,
            total_pages=total_pages,
            pages=parsed_pages,
            clean_text=full_clean_text,
            metadata={
                "source_path": str(file_path),
                "total_chars": len(full_clean_text),
                "total_pages": total_pages,
            },
        )
    finally:
        doc.close()


def load_manifest_lookup(manifest_path: Path | None) -> dict[str, dict[str, Any]]:
    """Build a filename-to-metadata lookup dict from corpus_manifest.csv."""
    if not manifest_path or not manifest_path.is_file():
        return {}

    lookup: dict[str, dict[str, Any]] = {}
    with open(manifest_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_number = row["doc_number"].strip()
            clean_num = re.sub(r"[^\w\.-]", "_", doc_number)
            clean_num = re.sub(r"_+", "_", clean_num)
            filename = f"{clean_num}.pdf"
            lookup[filename] = dict(row)
    return lookup


def parse_corpus(
    raw_dir: Path,
    output_dir: Path,
    manifest_path: Path | None = None,
    limit: int | None = None,
) -> list[ParsedDocument]:
    """Parse all or a subset of PDFs in raw_dir and save structured JSON to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(raw_dir.glob("*.pdf"))

    if limit is not None:
        pdf_files = pdf_files[:limit]

    manifest_lookup = load_manifest_lookup(manifest_path)
    parsed_docs: list[ParsedDocument] = []

    for pdf_path in pdf_files:
        logger.info("Parsing PDF: %s", pdf_path.name)
        doc = parse_pdf(pdf_path)

        # Attach manifest metadata if available
        if pdf_path.name in manifest_lookup:
            doc.metadata.update(manifest_lookup[pdf_path.name])

        parsed_docs.append(doc)

        # Save individual structured JSON artifact
        json_path = output_dir / f"{doc.doc_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(
            "Parsed %s: %d pages, %d chars -> %s",
            pdf_path.name,
            doc.total_pages,
            len(doc.clean_text),
            json_path.name,
        )

    return parsed_docs


def spot_check_documents(
    doc_filenames: list[str],
    raw_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Extract and perform detailed structural validation on a specified set of documents."""
    results: dict[str, Any] = {}
    for filename in doc_filenames:
        pdf_path = raw_dir / filename
        if not pdf_path.is_file():
            logger.warning("Spot-check file missing: %s", pdf_path)
            results[filename] = {"status": "missing"}
            continue

        doc = parse_pdf(pdf_path)

        # Verification checks
        has_pasal = bool(re.search(r"\bPasal\s+\d+\b", doc.clean_text))
        has_bab = bool(re.search(r"\bBAB\s+[IVXLCDM]+\b", doc.clean_text))
        page_nums_in_text = bool(re.search(r"\n-\s*\d+\s*-\n", doc.clean_text))

        summary = {
            "status": "success",
            "pages": doc.total_pages,
            "chars": len(doc.clean_text),
            "has_pasal": has_pasal,
            "has_bab": has_bab,
            "header_stripped": not page_nums_in_text,
        }
        results[filename] = summary

        # Save parsed JSON
        json_path = output_dir / f"{doc.doc_id}.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(
            "SPOT-CHECK [%s]: %d pages, %d chars, BAB=%s, Pasal=%s, HeadersClean=%s",
            filename,
            summary["pages"],
            summary["chars"],
            summary["has_bab"],
            summary["has_pasal"],
            summary["header_stripped"],
        )

    return results


def main() -> None:
    """CLI runner for PDF parsing and extraction spot-checks."""
    parser = argparse.ArgumentParser(
        description="Parse regulatory PDFs using PyMuPDF and clean layout artifacts."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing raw PDF documents",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/parsed"),
        help="Directory to store parsed JSON outputs",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/corpus_manifest.csv"),
        help="Path to corpus_manifest.csv",
    )
    parser.add_argument(
        "--spot-check",
        action="store_true",
        help="Run spot-check on 5 key representative documents",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to parse",
    )
    args = parser.parse_args()

    if args.spot_check:
        spot_check_files = [
            "POJK_22_Tahun_2023.pdf",
            "POJK_11_POJK.03_2022.pdf",
            "POJK_8_Tahun_2023.pdf",
            "UU_Nomor_27_Tahun_2022.pdf",
            "POJK_31_POJK.04_2015.pdf",
        ]
        logger.info("Running spot-check extraction on 5 documents...")
        results = spot_check_documents(spot_check_files, args.input_dir, args.output_dir)
        for fname, res in results.items():
            print(f"  {fname}: {res}")
    else:
        logger.info("Starting corpus PDF parsing from %s...", args.input_dir)
        docs = parse_corpus(
            raw_dir=args.input_dir,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            limit=args.limit,
        )
        logger.info("Parsing complete: %d documents processed.", len(docs))


if __name__ == "__main__":
    main()
