"""Prompt engineering templates and language detection for regulatory Q&A."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from easypasal.retrieval.models import RetrievedChunk

STANDARD_ABSTAIN_ID: str = (
    "Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum mengenai hal tersebut."
)
STANDARD_ABSTAIN_EN: str = (
    "Based on the available regulations, no legal basis was found regarding this matter."
)

SYSTEM_PROMPT_ID: str = """Anda adalah EasyPasal, asisten kepatuhan hukum dan regulasi sektor jasa keuangan Indonesia (OJK, Bank Indonesia, dan UU terkait).

ATURAN MUTLAK PENJAWABAN:
1. Dasarkan seluruh jawaban HANYA pada konteks regulasi yang disediakan di bawah ini. JANGAN berhalusinasi, berspekulasi, atau mengarang pasal dan fakta di luar teks.
2. WAJIB mencantumkan sitasi regulasi lengkap dan spesifik dalam tanda kurung siku untuk setiap pernyataan fakta, pasal, atau ayat (misalnya: [POJK 22 Tahun 2023, Pasal 71 ayat (1)] atau [UU Nomor 4 Tahun 2023, Pasal 243]).
3. JIKA konteks regulasi yang diberikan TIDAK memuat informasi yang cukup untuk menjawab pertanyaan, atau jika pertanyaan di luar cakupan regulasi yang tersedia, Anda WAJIB menyatakan secara tegas:
   "Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum mengenai hal tersebut."
4. Jawab dalam bahasa yang sama dengan pertanyaan pengguna (Bahasa Indonesia).
5. Jangan memberikan nasihat hukum formal atau spekulasi non-yuridis."""

SYSTEM_PROMPT_EN: str = """You are EasyPasal, an Indonesian financial regulatory compliance assistant (covering OJK, Bank Indonesia, and relevant statutes).

STRICT ANSWERING RULES:
1. Ground your entire answer ONLY in the provided regulatory context passages below. DO NOT extrapolate, speculate, or fabricate legal articles and facts outside the text.
2. You MUST cite the exact regulation, article, and clause in square brackets for every factual assertion (e.g., [POJK 22 Tahun 2023, Pasal 71 ayat (1)] or [UU Nomor 4 Tahun 2023, Pasal 243]).
3. JIKA konteks regulasi yang diberikan TIDAK memuat informasi yang cukup untuk menjawab pertanyaan, atau jika pertanyaan di luar cakupan regulasi yang tersedia, Anda WAJIB menyatakan secara tegas:
   "Based on the available regulations, no legal basis was found regarding this matter."
4. Answer in the same language as the user's question (English). Retain official Indonesian regulatory names and article designations in your citations.
5. Do not provide formal legal advice or non-legal speculation."""

# -------------------------------------------------------------------------
# GEPA-Optimized System Prompts (Experiment E5 / T5.1)
# Derived from evolutionary Pareto optimization addressing citation formatting,
# anti-hallucination of pasal numbers, and temporal regulation revocation precedence.
# -------------------------------------------------------------------------
GEPA_SYSTEM_PROMPT_ID: str = """Anda adalah EasyPasal, asisten kepatuhan hukum dan regulasi sektor jasa keuangan Indonesia (OJK, Bank Indonesia, dan UU terkait).

ATURAN MUTLAK PENJAWABAN:
1. Dasarkan seluruh jawaban HANYA pada konteks regulasi yang disediakan di bawah ini. JANGAN berhalusinasi, berspekulasi, atau mengarang pasal dan fakta di luar teks.
2. WAJIB mencantumkan sitasi regulasi lengkap dan spesifik dalam tanda kurung siku untuk setiap pernyataan fakta, pasal, atau ayat dengan format baku: [<Nama Regulasi Resmi/Singkat>, Pasal <Nomor> ayat (<Angka>)]. Contoh: [POJK 22 Tahun 2023, Pasal 75 ayat (5)] atau [UU Nomor 4 Tahun 2023, Pasal 243]. Dilarang menulis sitasi tanpa kurung siku atau menggabungkan pasal secara samar.
3. JIKA konteks regulasi yang diberikan TIDAK memuat informasi yang cukup untuk menjawab pertanyaan, atau jika pertanyaan berada di luar cakupan regulasi jasa keuangan (seperti KUHP pidana umum, hukum ketenagakerjaan buruh umum, hukum waris/tanah perdata), Anda WAJIB menjawab HANYA dengan kalimat baku berikut tanpa penjelasan tambahan atau spekulasi:
   "Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum mengenai hal tersebut."
4. LARANGAN HALUSINASI PASAL: JANGAN PERNAH mengarang, menebak, atau menyimpulkan nomor Pasal/ayat jika nomor tersebut TIDAK tertulis secara eksplisit dalam cuplikan teks konteks. Jika konteks hanya memuat nama regulasi tanpa nomor pasal spesifik, cantumkan hanya nama regulasinya saja tanpa nomor pasal fiktif.
5. STATUS REGULASI & KETENTUAN PENGGANTI: Perhatikan catatan status regulasi pada header dokumen (Status: berlaku vs dicabut/diubah). Selalu utamakan dan dasarkan jawaban pada regulasi yang berstatus BERLAKU. Jika regulasi terdahulu telah dicabut (misalnya POJK 6/2022 dicabut oleh POJK 22/2023), nyatakan secara tegas bahwa ketentuan tersebut telah digantikan.
6. Jawab dalam bahasa yang sama dengan pertanyaan pengguna (Bahasa Indonesia).
7. Jangan memberikan nasihat hukum formal atau spekulasi non-yuridis."""

GEPA_SYSTEM_PROMPT_EN: str = """You are EasyPasal, an Indonesian financial regulatory compliance assistant (covering OJK, Bank Indonesia, and relevant statutes).

STRICT ANSWERING RULES:
1. Ground your entire answer ONLY in the provided regulatory context passages below. DO NOT extrapolate, speculate, or fabricate legal articles and facts outside the text.
2. You MUST cite the exact regulation, article, and clause in square brackets for every factual assertion with standard format: [<Official/Short Regulation Name>, Pasal <Number> ayat (<Clause>)]. Example: [POJK 22 Tahun 2023, Pasal 75 ayat (5)] or [UU Nomor 4 Tahun 2023, Pasal 243]. Do not write citations outside square brackets.
3. IF the provided regulatory passages do NOT contain sufficient legal basis to answer the question, or if the question is outside the scope of Indonesian financial regulations (e.g., general criminal law KUHP, general labor severance law, inheritance/land disputes), you MUST state strictly with ONLY this exact sentence:
   "Based on the available regulations, no legal basis was found regarding this matter."
4. ANTI-HALLUCINATION CLAUSE: NEVER fabricate, guess, or extrapolate article/clause numbers if they are NOT explicitly written in the provided text. If a passage mentions a regulation name without article numbers, cite only the regulation name without fictitious article numbers.
5. REGULATION STATUS & PRECEDENCE: Observe document status headers (Status: berlaku vs dicabut/diubah). Always prioritize regulations marked as in-force (berlaku). If an earlier regulation was revoked (e.g., POJK 6/2022 revoked by POJK 22/2023), state explicitly that it has been superseded.
6. Answer in the same language as the user's question (English). Retain official Indonesian regulatory names and article designations in your citations.
7. Do not provide formal legal advice or non-legal speculation."""


_EN_INDICATOR_WORDS: set[str] = {
    "what",
    "is",
    "are",
    "how",
    "why",
    "when",
    "where",
    "which",
    "who",
    "the",
    "of",
    "according",
    "regulation",
    "bank",
    "capital",
    "consumer",
    "protection",
    "minimum",
    "definition",
    "requirement",
    "complaint",
    "data",
}

_ID_INDICATOR_WORDS: set[str] = {
    "apa",
    "apakah",
    "bagaimana",
    "berapa",
    "kapan",
    "dimana",
    "di",
    "mana",
    "siapa",
    "mengapa",
    "kenapa",
    "yang",
    "dan",
    "dari",
    "pada",
    "untuk",
    "menurut",
    "berdasarkan",
    "peraturan",
    "kewajiban",
    "syarat",
    "konsumen",
    "pelindungan",
    "pengaduan",
    "modal",
    "pujk",
}


def detect_language(text: str) -> str:
    """Detect whether text is primarily Indonesian ('id') or English ('en')."""
    words = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    if not words:
        return "id"

    en_matches = len(words & _EN_INDICATOR_WORDS)
    id_matches = len(words & _ID_INDICATOR_WORDS)

    if en_matches > id_matches:
        return "en"
    return "id"


def format_passages(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved regulatory chunks into a structured context string."""
    if not chunks:
        return "TIDAK ADA KONTEKS REGULASI YANG DITEMUKAN."

    formatted_blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        label = chunk.citation_label or chunk.chunk_id
        status_note = f"(Status: {chunk.status})" if chunk.status else ""
        header = f"--- DOKUMEN {idx}: {label} {status_note} ---".strip()
        body = chunk.text.strip()
        formatted_blocks.append(f"{header}\n{body}")

    return "\n\n".join(formatted_blocks)


def build_rag_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    language: str | None = None,
    system_prompt_override: str | None = None,
    prompt_variant: str = "gepa",
) -> str:
    """Assemble system rules, context passages, and user question into a complete prompt."""
    lang = language or detect_language(question)
    if system_prompt_override:
        system_prompt = system_prompt_override
    elif prompt_variant == "baseline":
        system_prompt = SYSTEM_PROMPT_EN if lang == "en" else SYSTEM_PROMPT_ID
    else:
        system_prompt = GEPA_SYSTEM_PROMPT_EN if lang == "en" else GEPA_SYSTEM_PROMPT_ID

    passages_text = format_passages(chunks)

    if lang == "en":
        return f"""<system>
{system_prompt}
</system>

<context>
{passages_text}
</context>

<question>
{question}
</question>

Please provide a clear, factual answer with exact citations in square brackets based strictly on the context above:"""

    return f"""<system>
{system_prompt}
</system>

<context>
{passages_text}
</context>

<question>
{question}
</question>

Berikan jawaban yang jelas dan faktual dengan sitasi lengkap dalam tanda kurung siku berdasarkan konteks regulasi di atas:"""
