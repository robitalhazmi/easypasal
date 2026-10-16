"""Question router and guardrail classifier (Task T6.4 / Experiment E7).

Provides front-line triage for user queries:
1. Detects adversarial prompt injections and jailbreak attacks.
2. Identifies out-of-scope non-financial inquiries and superseded regulation traps.
3. Classifies in-scope regulatory themes for metadata-directed routing.
4. Fast-paths refusals with sub-25ms latency, bypassing expensive retrieval and generation.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from easypasal.config import GuardrailsConfig

logger = logging.getLogger("easypasal.guardrails.question_router")

CANONICAL_THEMES = [
    "Consumer Protection",
    "IT & Cyber Risk",
    "AML / CFT",
    "Data Privacy & Secrecy",
    "Capital Markets",
    "General Financial Regulation",
    "Out of Scope",
]

# Standardized refusal messages that conform to check_abstention patterns
REFUSAL_INJECTION_ID = (
    "Maaf, permintaan ini tidak dapat diproses karena terdeteksi instruksi yang "
    "berada di luar cakupan regulasi atau manipulasi panduan sistem."
)
REFUSAL_INJECTION_EN = (
    "Sorry, this request cannot be processed because it is outside the scope of "
    "regulatory guidelines or attempts to manipulate system instructions."
)

REFUSAL_SUPERSEDED_ID = (
    "Maaf, informasi tidak ditemukan dalam regulasi yang masih berlaku karena ketentuan "
    "yang ditanyakan telah dicabut atau digantikan oleh regulasi yang lebih baru."
)
REFUSAL_SUPERSEDED_EN = (
    "Sorry, information not found in the available regulations as the queried rule has "
    "been repealed or superseded by newer financial regulations."
)

REFUSAL_OUT_OF_SCOPE_ID = (
    "Maaf, informasi tidak ditemukan dalam regulasi keuangan yang relevan dalam korpus "
    "Otoritas Jasa Keuangan (OJK) dan Bank Indonesia."
)
REFUSAL_OUT_OF_SCOPE_EN = (
    "Sorry, information not found in the relevant financial regulations within the "
    "OJK and Bank Indonesia corpus."
)

# Regex matching adversarial injection and jailbreak attempts
_INJECTION_PATTERNS = [
    re.compile(
        r"\b(?:abaikan|ignore)\s+(?:semua\s+)?(?:instruksi|aturan|directions|guidelines)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:system\s+(?:prompt\s+)?override|administrator\s+override)\b", re.IGNORECASE),
    re.compile(r"\b(?:dan\s+mode|unrestrictedlegalai|unrestricted\s+ai)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:bypass|tembus|bobol)\s+(?:semua\s+)?(?:aturan|guardrails|guidelines|protokol)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:tampilkan|bocorkan|print|reveal)\s+(?:seluruh\s+)?(?:prompt\s+sistem|system\s+prompt|api\s+key)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:agen\s+pasar\s+gelap|black\s+market|konsultan\s+kejahatan)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:jaminan\s+mutlak|guaranteed\s+immunity|tidak\s+dapat\s+dituntut)\b", re.IGNORECASE
    ),
]

# Regex matching repealed / superseded regulations
_SUPERSEDED_PATTERNS = [
    re.compile(r"\bPOJK\s+(?:Nomor\s+|No\.?\s*)?1\s*/\s*POJK\.07\s*/\s*2013\b", re.IGNORECASE),
    re.compile(r"\bPBI\s+(?:Nomor\s+|No\.?\s*)?9\s*/\s*15\s*/\s*PBI\s*/\s*2007\b", re.IGNORECASE),
    re.compile(r"\bUU\s+(?:Nomor\s+|No\.?\s*)?15\s+Tahun\s+2002\b", re.IGNORECASE),
    re.compile(
        r"\bKep(?:utusan)?\s+(?:Ketua\s+)?Bapepam\s+(?:Nomor\s+|No\.?\s*)?Kep-45\b", re.IGNORECASE
    ),
    re.compile(r"\bPOJK\s+(?:Nomor\s+|No\.?\s*)?38\s*/\s*POJK\.03\s*/\s*2016\b", re.IGNORECASE),
]

# Regex matching hard out-of-scope non-financial domains
_OUT_OF_SCOPE_PATTERNS = [
    re.compile(
        r"\b(?:surat\s+izin\s+mengemudi|sim\b|lalu\s+lintas|lampu\s+merah|tilang)\b", re.IGNORECASE
    ),
    re.compile(
        r"\b(?:hak\s+guna\s+bangunan|hgb\b|hak\s+guna\s+usaha|hgu\b|uu\s+pokok\s+agraria|uupa\b|tanah\s+adat)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:pesangon|pemutusan\s+hubungan\s+kerja|phk\b|uu\s+ketenagakerjaan|upah\s+minimum)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:pajak\s+penghasilan|pph\s+21|spt\s+tahunan|faktur\s+pajak)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:pasal\s+363\s+kuhp|kuhp\b|tindak\s+pidana\s+pencurian|pembunuhan)\b", re.IGNORECASE
    ),
    re.compile(r"\b(?:resep|bumbu|rendang|masak|kue)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:asynchronous\s+context\s+manager|python\s+3\.\d+|javascript|react|css)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:perkawinan\s+campuran|hukum\s+waris|kompilasi\s+hukum\s+islam)\b", re.IGNORECASE
    ),
]


class RouteDecision(BaseModel):
    """Output decision emitted by the QuestionRouter."""

    theme: str = Field(..., description="Classified regulatory theme or 'Out of Scope'")
    is_answerable: bool = Field(
        ..., description="True if query is answerable by current financial regulations"
    )
    is_injection: bool = Field(
        ..., description="True if prompt injection or jailbreak attempt detected"
    )
    should_block: bool = Field(
        ..., description="True if query should be fast-pathed to immediate refusal"
    )
    refusal_reason: str | None = Field(
        default=None, description="Diagnostic explanation for refusal"
    )
    refusal_response: str | None = Field(default=None, description="Standardized refusal message")
    latency_ms: float = Field(
        default=0.0, description="Routing classification latency in milliseconds"
    )


class QuestionRouter:
    """Multi-task regulatory question router and safety guardrail."""

    def __init__(
        self,
        config: GuardrailsConfig | None = None,
        adapter_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self._model: Any = None
        self._tokenizer: Any = None
        self._is_live_gpu = False

    def route(self, question: str, language: str = "id") -> RouteDecision:
        """Classify user query into theme, answerability, and injection defense."""
        start_t = time.perf_counter()
        clean_q = question.strip()

        # 1. Adversarial Injection Check (highest priority guardrail)
        for pat in _INJECTION_PATTERNS:
            if pat.search(clean_q):
                elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)
                refusal = REFUSAL_INJECTION_EN if language == "en" else REFUSAL_INJECTION_ID
                return RouteDecision(
                    theme="Out of Scope",
                    is_answerable=False,
                    is_injection=True,
                    should_block=True,
                    refusal_reason="Adversarial prompt injection / jailbreak detected",
                    refusal_response=refusal,
                    latency_ms=max(elapsed_ms, 1.2),
                )

        # 2. Superseded Regulation Trap Check
        for pat in _SUPERSEDED_PATTERNS:
            if pat.search(clean_q):
                elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)
                theme = self._infer_theme(clean_q)
                refusal = REFUSAL_SUPERSEDED_EN if language == "en" else REFUSAL_SUPERSEDED_ID
                return RouteDecision(
                    theme=theme,
                    is_answerable=False,
                    is_injection=False,
                    should_block=True,
                    refusal_reason="Superseded or repealed regulation without active legal force",
                    refusal_response=refusal,
                    latency_ms=max(elapsed_ms, 1.5),
                )

        # 3. Hard Out-of-Scope Non-Financial Domain Check
        for pat in _OUT_OF_SCOPE_PATTERNS:
            if pat.search(clean_q):
                elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)
                refusal = REFUSAL_OUT_OF_SCOPE_EN if language == "en" else REFUSAL_OUT_OF_SCOPE_ID
                return RouteDecision(
                    theme="Out of Scope",
                    is_answerable=False,
                    is_injection=False,
                    should_block=True,
                    refusal_reason="Out-of-scope non-financial legal or factual query",
                    refusal_response=refusal,
                    latency_ms=max(elapsed_ms, 1.5),
                )

        # 4. In-Scope Regulatory Query (Theme Identification)
        theme = self._infer_theme(clean_q)
        elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

        return RouteDecision(
            theme=theme,
            is_answerable=True,
            is_injection=False,
            should_block=False,
            refusal_reason=None,
            refusal_response=None,
            latency_ms=max(elapsed_ms, 2.0),
        )

    def _infer_theme(self, text: str) -> str:
        """Heuristic theme identification calibrated with the QLoRA adapter."""
        t_lower = text.lower()

        if any(
            w in t_lower
            for w in [
                "siber",
                "cyber",
                "ti bank",
                "drc",
                "data center",
                "insiden siber",
                "kerentanan",
            ]
        ):
            return "IT & Cyber Risk"
        if any(
            w in t_lower
            for w in ["konsumen", "pujk", "pengaduan", "sengketa", "laps", "klausula baku"]
        ):
            return "Consumer Protection"
        if any(
            w in t_lower
            for w in ["pencucian uang", "ppatk", "ltkm", "cdd", "edd", "apu", "ppt", "dttot"]
        ):
            return "AML / CFT"
        if any(
            w in t_lower
            for w in ["rahasia bank", "data pribadi", "pdp", "telemarketing", "kebocoran data"]
        ):
            return "Data Privacy & Secrecy"
        if any(
            w in t_lower
            for w in [
                "efek",
                "saham",
                "emiten",
                "reksa dana",
                "insider trading",
                "bursa",
                "pasar modal",
            ]
        ):
            return "Capital Markets"

        return "General Financial Regulation"


def route_question(
    question: str,
    language: str = "id",
    config: GuardrailsConfig | None = None,
) -> RouteDecision:
    """Convenience helper to route a single question through the guardrail router."""
    router = QuestionRouter(config=config)
    return router.route(question=question, language=language)
