"""Reflective candidate proposer for GEPA prompt optimization (T5.1).

Analyzes diagnostic execution traces from EasyPasalGEPAAdapter and proposes
targeted prompt refinements addressing citation formatting, hallucination,
abstention boundary precision, and regulatory revocation precedence.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from easypasal.generation.prompts import SYSTEM_PROMPT_ID

logger = logging.getLogger("easypasal.experiments.e5_gepa.proposer")

# Targeted prompt modification blocks derived from legal failure mode reflections
DIRECTIVE_CITATION_SYNTAX = """- FORMAT SITASI WAJIB: Setiap kutipan dasar hukum WAJIB ditulis dalam tanda kurung siku dengan format standar: [<Nama Regulasi Resmi/Singkat>, Pasal <Nomor> ayat (<Angka>)]. Contoh: [POJK 22 Tahun 2023, Pasal 75 ayat (5)] atau [UU Nomor 4 Tahun 2023, Pasal 243]. Dilarang menulis sitasi tanpa kurung siku atau menggabungkan pasal secara samar."""

DIRECTIVE_ANTI_HALLUCINATION = """- LARANGAN HALUSINASI PASAL: JANGAN PERNAH mengarang, menebak, atau menyimpulkan nomor Pasal/ayat jika nomor tersebut TIDAK tertulis secara eksplisit dalam cuplikan teks konteks. Jika konteks hanya memuat nama regulasi tanpa nomor pasal spesifik, cantumkan hanya nama regulasinya tanpa nomor pasal fiktif."""

DIRECTIVE_STRICT_ABSTENTION = """- BATAS ABSTENSI TEGAS: Jika pertanyaan berada di luar cakupan regulasi jasa keuangan (seperti KUHP pidana umum, ketenagakerjaan buruh umum, hukum waris/tanah perdata), ATAU jika cuplikan teks regulasi yang tersedia tidak memuat informasi yang cukup, Anda WAJIB menjawab HANYA dengan kalimat baku berikut tanpa penjelasan tambahan:
  "Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum mengenai hal tersebut.\""""

DIRECTIVE_TEMPORAL_REVOCATION = """- STATUS REGULASI & KETENTUAN PENGGANTI: Perhatikan catatan status regulasi pada header dokumen (Status: berlaku vs dicabut/diubah). Selalu utamakan dan dasarkan jawaban pada regulasi yang berstatus BERLAKU. Jika regulasi terdahulu telah dicabut (misal POJK 6/2022 dicabut oleh POJK 22/2023), nyatakan bahwa ketentuan tersebut telah digantikan."""

DIRECTIVE_FACTUAL_CONCISENESS = """- STRUKTUR JAWABAN: Berikan jawaban yang langsung pada pokok persoalan, padat fakta hukum, tanpa kalimat pembuka obrolan (seperti 'Tentu, saya akan membantu...'), dan pisahkan poin-poin kewajiban secara terstruktur jika terdapat lebih dari satu syarat."""


class EasyPasalReflectiveProposer:
    """Proposer that mutates system prompts based on diagnostic error traces."""

    def __init__(self, base_prompt: str = SYSTEM_PROMPT_ID) -> None:
        self.base_prompt = base_prompt
        self._iteration = 0

    def propose(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Propose next prompt variation by analyzing failure patterns in reflective traces."""
        self._iteration += 1
        current_prompt = candidate.get("system_prompt", self.base_prompt)

        # Collect failure indicators from diagnostic traces
        traces: Sequence[Mapping[str, Any]] = []
        for comp in components_to_update:
            if comp in reflective_dataset:
                traces = reflective_dataset[comp]
                break

        feedback_texts = [str(t.get("Feedback", "")) for t in traces]
        all_feedback = " ".join(feedback_texts)

        needs_abstention = "FAILED ABSTENTION" in all_feedback or "FALSE ABSTENTION" in all_feedback
        needs_citations = "MISSED CITATIONS" in all_feedback
        needs_grounding = "UNGROUNDED CITATIONS" in all_feedback or "HALLUCINATION" in all_feedback

        logger.info(
            "GEPA Proposer Iteration %d: traces=%d, abstention_issue=%s, citation_issue=%s, grounding_issue=%s",
            self._iteration,
            len(traces),
            needs_abstention,
            needs_citations,
            needs_grounding,
        )

        # Build candidate prompt variant based on active needs and iteration
        mutated_prompt = current_prompt

        directives_to_add: list[str] = []
        if needs_abstention and DIRECTIVE_STRICT_ABSTENTION not in mutated_prompt:
            directives_to_add.append(DIRECTIVE_STRICT_ABSTENTION)

        if (
            needs_citations or self._iteration >= 2
        ) and DIRECTIVE_CITATION_SYNTAX not in mutated_prompt:
            directives_to_add.append(DIRECTIVE_CITATION_SYNTAX)

        if (
            needs_grounding or self._iteration >= 3
        ) and DIRECTIVE_ANTI_HALLUCINATION not in mutated_prompt:
            directives_to_add.append(DIRECTIVE_ANTI_HALLUCINATION)

        if self._iteration >= 4 and DIRECTIVE_TEMPORAL_REVOCATION not in mutated_prompt:
            directives_to_add.append(DIRECTIVE_TEMPORAL_REVOCATION)

        if self._iteration >= 5 and DIRECTIVE_FACTUAL_CONCISENESS not in mutated_prompt:
            directives_to_add.append(DIRECTIVE_FACTUAL_CONCISENESS)

        if directives_to_add:
            # Append new directives under ATURAN MUTLAK PENJAWABAN
            additions = "\n\nPETUNJUK KEPATUHAN TAMBAHAN:\n" + "\n\n".join(directives_to_add)
            mutated_prompt = mutated_prompt.strip() + "\n" + additions

        result: dict[str, str] = {}
        for comp in components_to_update:
            result[comp] = mutated_prompt
        return result
