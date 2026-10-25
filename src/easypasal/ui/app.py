"""Gradio Web User Interface for EasyPasal (Plan §8.3 / Task T7.6).

Provides an interactive regulatory compliance assistant with bilingual support,
collapsible statutory citation inspector, user feedback collection,
and transparent abstention handling.
"""

from __future__ import annotations

import argparse
import asyncio.selector_events
import contextlib
import logging
import os
import time
from typing import Any

# Silence benign BaseEventLoop.__del__ ValueError on Linux container GC
try:
    if hasattr(asyncio.selector_events.BaseSelectorEventLoop, "_close_self_pipe"):
        _orig_close_self_pipe = asyncio.selector_events.BaseSelectorEventLoop._close_self_pipe

        def _safe_close_self_pipe(self: Any) -> None:
            with contextlib.suppress(ValueError, KeyError, OSError):
                _orig_close_self_pipe(self)

        asyncio.selector_events.BaseSelectorEventLoop._close_self_pipe = _safe_close_self_pipe
except Exception:
    pass

# Import spaces before heavy AI libraries for Hugging Face ZeroGPU compatibility
try:
    import spaces  # type: ignore[import-not-found]
except ImportError:

    class _MockSpaces:
        @staticmethod
        def GPU(func: Any = None, **_kwargs: Any) -> Any:
            if func is not None:
                return func
            return lambda f: f

    spaces = _MockSpaces()

import gradio as gr

from easypasal.generation.pipeline import RAGPipeline
from easypasal.monitoring.logger import RequestLogger
from easypasal.monitoring.schemas import FeedbackLogRecord, RequestLogRecord

logger = logging.getLogger("easypasal.ui.app")

DEFAULT_EXAMPLES = [
    [
        "Di mana Bank Umum wajib menempatkan Pusat Data dan Pusat Pemulihan Bencana menurut regulasi OJK?",
        "id",
        "gepa_v0.2",
    ],
    [
        "Berapa batas maksimum manfaat ekonomi atau bunga pinjaman pada fintech P2P lending?",
        "id",
        "gepa_v0.2",
    ],
    [
        "Bagaimana kewajiban Lembaga Jasa Keuangan dalam menangani dan menyelesaikan pengaduan konsumen?",
        "id",
        "baseline_v0.1",
    ],
    [
        "Apa saja pilar ketahanan dan tata kelola siber yang wajib diterapkan oleh perbankan?",
        "id",
        "gepa_v0.2",
    ],
    [
        "What are the principal consumer protection obligations for digital financial services under OJK rules?",
        "en",
        "baseline_v0.1",
    ],
    [
        "Berapa modal minimum yang dibutuhkan untuk membuka kedai kopi di Jakarta Selatan?",
        "id",
        "baseline_v0.1",
    ],
]


def format_citations_markdown(output: Any) -> str:
    """Format retrieved statutory passages into structured markdown for review."""
    abstained = getattr(output, "abstained", False)
    language = getattr(output, "language", "id")

    if abstained:
        if language == "en":
            return (
                "*This query does not have a governing statutory basis within the available "
                "regulatory corpus. No articles were cited.*"
            )
        return (
            "*Pertanyaan ini tidak memiliki dasar hukum yang berlaku dalam korpus regulasi yang tersedia. "
            "Tidak ada pasal yang disitir.*"
        )

    retrieved_chunks = getattr(output, "retrieved_chunks", [])
    if not retrieved_chunks:
        if language == "en":
            return "*No reference statutory provisions retrieved for this query.*"
        return "*Tidak ada pasal rujukan yang ditarik untuk pertanyaan ini.*"

    md_lines: list[str] = [
        "### 📚 Naskah dan Pasal Terkait",
        "",
    ]
    for idx, chunk in enumerate(retrieved_chunks, 1):
        doc_id = getattr(chunk, "doc_id", "Regulasi OJK/BI")
        pasal = getattr(chunk, "pasal", "")
        ayat = getattr(chunk, "ayat", "")
        score = getattr(chunk, "score", None)
        text = getattr(chunk, "text", "")

        pasal_info = f"Pasal {pasal}" if pasal else ""
        if ayat:
            pasal_info += f" Ayat ({ayat})" if not pasal_info else f" ayat ({ayat})"

        header = f"**{idx}. [{doc_id}] {pasal_info}**"
        if score is not None:
            header += f" *(Relevance Score: {score:.3f})*"

        md_lines.append(header)
        clean_text = text.strip().replace("\n", " ")
        if len(clean_text) > 400:
            clean_text = clean_text[:400] + "..."
        md_lines.append(f'> "{clean_text}"')
        md_lines.append("")

    return "\n".join(md_lines)


def build_ui(
    pipeline: RAGPipeline | None = None,
    request_logger: RequestLogger | None = None,
) -> gr.Blocks:
    """Build and configure the Gradio Blocks user interface for EasyPasal."""
    active_pipeline = pipeline

    custom_css = """
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        max-width: 1320px !important;
        margin: 0 auto !important;
    }
    .disclaimer-banner {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 16px;
        font-size: 0.85rem;
        color: #94a3b8;
        line-height: 1.4;
    }
    .meta-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 0.85rem;
    }
    #query-panel,
    #examples-panel,
    #response-panel,
    #feedback-panel {
        background: #0f172a !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        padding: 16px !important;
        margin-bottom: 14px !important;
    }
    #response-panel {
        min-height: 220px !important;
    }
    /* Example Queries - Full Visibility, Word-wrapped, Zero Truncation */
    .example-queries-container,
    #example-queries-list {
        overflow: visible !important;
        max-height: none !important;
        height: auto !important;
        width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 8px !important;
    }
    .example-query-btn {
        display: block !important;
        width: 100% !important;
        white-space: normal !important;
        word-break: break-word !important;
        overflow-wrap: break-word !important;
        text-overflow: clip !important;
        overflow: visible !important;
        text-align: left !important;
        font-size: 0.86rem !important;
        line-height: 1.45 !important;
        font-weight: 400 !important;
        padding: 10px 14px !important;
        margin-bottom: 6px !important;
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        color: #cbd5e1 !important;
        box-shadow: none !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
        height: auto !important;
        min-height: unset !important;
        max-height: none !important;
        box-sizing: border-box !important;
    }
    .example-query-btn:hover {
        background-color: #1e293b !important;
        border-color: #38bdf8 !important;
        color: #38bdf8 !important;
        transform: translateY(-1px);
    }
    .example-query-btn * {
        white-space: normal !important;
        word-break: break-word !important;
        overflow-wrap: break-word !important;
        text-overflow: clip !important;
        overflow: visible !important;
        text-align: left !important;
        line-height: 1.45 !important;
    }

    #example-queries,
    #example-queries .dataset,
    #example-queries .table-wrap,
    #example-queries .gallery,
    #example-queries table,
    .example-queries-container .dataset,
    .example-queries-container .table-wrap,
    .example-queries-container .gallery,
    .example-queries-container table {
        overflow: visible !important;
        max-height: none !important;
        height: auto !important;
        width: 100% !important;
    }

    /* Gallery card layout (Gradio single visible column mode) */
    #example-queries .gallery,
    .example-queries-container .gallery {
        display: flex !important;
        flex-direction: column !important;
        gap: 8px !important;
        width: 100% !important;
    }
    #example-queries .gallery-item,
    .example-queries-container .gallery-item {
        display: block !important;
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
        min-height: unset !important;
        max-height: none !important;
        overflow: visible !important;
        background: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        text-align: left !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
        box-sizing: border-box !important;
    }
    #example-queries .gallery-item:hover,
    .example-queries-container .gallery-item:hover {
        background: #1e293b !important;
        border-color: #38bdf8 !important;
        transform: translateY(-1px);
    }
    #example-queries .gallery-item *,
    #example-queries .gallery-item div,
    #example-queries .gallery-item span,
    #example-queries .gallery-item p,
    .example-queries-container .gallery-item *,
    .example-queries-container .gallery-item div,
    .example-queries-container .gallery-item span,
    .example-queries-container .gallery-item p,
    #example-queries button,
    #example-queries button div,
    #example-queries button span,
    .example-queries-container button,
    .example-queries-container button div,
    .example-queries-container button span {
        white-space: normal !important;
        word-break: break-word !important;
        overflow-wrap: break-word !important;
        text-overflow: clip !important;
        overflow: visible !important;
        line-height: 1.45 !important;
        font-size: 0.875rem !important;
        text-align: left !important;
        max-width: 100% !important;
        width: 100% !important;
        display: block !important;
        color: #cbd5e1 !important;
    }
    #example-queries .gallery-item:hover *,
    #example-queries .gallery-item:hover div,
    .example-queries-container .gallery-item:hover *,
    .example-queries-container .gallery-item:hover div {
        color: #38bdf8 !important;
    }

    /* Table layout fallback (if rendered as table) */
    #example-queries thead,
    .example-queries-container thead {
        display: none !important;
    }
    #example-queries table,
    .example-queries-container table {
        border-collapse: separate !important;
        border-spacing: 0 6px !important;
        width: 100% !important;
        table-layout: auto !important;
    }
    #example-queries tbody tr,
    .example-queries-container tbody tr {
        background: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        transition: all 0.15s ease-in-out !important;
    }
    #example-queries tbody tr:hover,
    .example-queries-container tbody tr:hover {
        background: #1e293b !important;
        border-color: #38bdf8 !important;
    }
    #example-queries td,
    .example-queries-container td {
        max-width: none !important;
        width: 100% !important;
        text-align: left !important;
        white-space: normal !important;
        word-break: break-word !important;
        overflow-wrap: break-word !important;
        text-overflow: clip !important;
        overflow: visible !important;
        padding: 10px 14px !important;
        font-size: 0.875rem !important;
        line-height: 1.45 !important;
        border: none !important;
        color: #cbd5e1 !important;
    }
    #example-queries td *,
    .example-queries-container td * {
        white-space: normal !important;
        word-break: break-word !important;
        overflow-wrap: break-word !important;
        text-overflow: clip !important;
        overflow: visible !important;
        line-height: 1.45 !important;
        text-align: left !important;
        max-width: none !important;
        width: 100% !important;
        display: block !important;
    }
    #example-queries tbody tr:hover td,
    #example-queries tbody tr:hover td *,
    .example-queries-container tbody tr:hover td,
    .example-queries-container tbody tr:hover td * {
        color: #38bdf8 !important;
    }
    #example-queries table th:nth-child(n+2),
    #example-queries table td:nth-child(n+2),
    .example-queries-container table th:nth-child(n+2),
    .example-queries-container table td:nth-child(n+2) {
        display: none !important;
    }
    #example-queries .paginate,
    .example-queries-container .paginate {
        display: none !important;
    }
    """

    with gr.Blocks(title="EasyPasal — Indonesian Regulatory Compliance AI") as demo:
        gr.HTML(f"<style>{custom_css}</style>")
        gr.HTML(
            """
            <div style="text-align: center; margin-top: 10px; margin-bottom: 16px;">
                <h1 style="font-size: 2.1rem; font-weight: 700; margin-bottom: 6px;">
                    ⚖️ EasyPasal
                </h1>
                <p style="font-size: 1.05rem; color: #64748b; margin-bottom: 12px;">
                    Asisten Kepatuhan Regulasi Sektor Jasa Keuangan Indonesia (OJK &amp; Bank Indonesia)
                </p>
                <div style="display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;">
                    <span style="background: #e0f2fe; color: #0369a1; padding: 2px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;">OJK / BI Corpus</span>
                    <span style="background: #ecfdf5; color: #047857; padding: 2px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;">Citation Grounded</span>
                    <span style="background: #fef3c7; color: #b45309; padding: 2px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;">Abstain on Out-of-Scope</span>
                    <span style="background: #f3e8ff; color: #6b21a8; padding: 2px 10px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;">Bilingual (ID/EN)</span>
                </div>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="disclaimer-banner">
                ⚠️ <strong>Pemberitahuan / Legal Disclaimer:</strong> EasyPasal adalah purwarupa AI riset regulasi kepatuhan hukum dan <em>tidak memberikan nasihat hukum mengikat (no legal advice)</em>. Seluruh jawaban dihasilkan dengan merujuk pasal dan ayat dari regulasi publik terkurasi. Selalu verifikasi ketentuan resmi pada JDIH OJK / Bank Indonesia sebelum pengambilan keputusan kepatuhan institusional.
            </div>
            """
        )

        # State storage for active request ID and generation output
        current_request_id = gr.State(value="")
        last_output_state = gr.State(value=None)

        with gr.Row(elem_id="main-workspace-row"):
            # Left Column: Input, settings, and example queries
            with gr.Column(
                scale=5,
                min_width=380,
                elem_id="query-column",
                elem_classes=["query-column-container"],
            ):
                with gr.Group(elem_id="query-panel"):
                    gr.Markdown("### 📝 Input Pertanyaan / Query")
                    question_input = gr.Textbox(
                        label="Tulis Pertanyaan Regulasi / Enter Compliance Question:",
                        placeholder="Contoh: Di mana Bank Umum wajib menempatkan Pusat Data menurut regulasi OJK?",
                        lines=3,
                        autofocus=True,
                    )
                    with gr.Row():
                        lang_toggle = gr.Radio(
                            choices=[("Bahasa Indonesia", "id"), ("English", "en")],
                            value="id",
                            label="Bahasa Respon / Language",
                        )
                        variant_selector = gr.Dropdown(
                            choices=["baseline_v0.1", "gepa_v0.2"],
                            value="baseline_v0.1",
                            label="Model Variant",
                        )

                    with gr.Row():
                        submit_btn = gr.Button(
                            "🔍 Cari & Jawab / Ask EasyPasal", variant="primary", scale=2
                        )
                        clear_btn = gr.Button("Bersihkan / Clear", scale=1)

                with gr.Group(elem_id="examples-panel"):
                    gr.Markdown("### ⚡ Pertanyaan Uji / Example Queries")
                    gr.Markdown(
                        "<span style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 6px; display: block;'>"
                        "Pilih pertanyaan untuk langsung menguji kepatuhan:"
                        "</span>"
                    )
                    with gr.Column(elem_id="example-queries-list"):
                        for ex_q, ex_l, ex_v in DEFAULT_EXAMPLES:
                            ex_btn = gr.Button(
                                ex_q,
                                size="sm",
                                elem_classes=["example-query-btn"],
                            )
                            ex_btn.click(
                                fn=lambda query_val=ex_q, lang_val=ex_l, var_val=ex_v: (
                                    query_val,
                                    lang_val,
                                    var_val,
                                ),
                                inputs=[],
                                outputs=[question_input, lang_toggle, variant_selector],
                            )

            # Right Column: Grounded Answer, Citations & Feedback
            with gr.Column(
                scale=6,
                min_width=420,
                elem_id="response-column",
                elem_classes=["response-column-container"],
            ):
                with gr.Group(elem_id="response-panel"):
                    gr.Markdown("### 📋 Jawaban Kepatuhan / Grounded Answer")
                    with gr.Row():
                        status_badge = gr.HTML(
                            value="<span style='color: #64748b;'>Status: Menunggu pertanyaan</span>"
                        )
                        telemetry_info = gr.HTML(value="")

                    answer_output = gr.Markdown(
                        value="*Silakan masukkan pertanyaan kepatuhan di sebelah kiri atau pilih salah satu pertanyaan uji untuk melihat jawaban berdasar pasal.*"
                    )

                # Collapsible Citations Inspector
                with gr.Accordion("🔎 Rincian Sitasi & Potongan Naskah Regulasi", open=True):
                    citations_output = gr.Markdown(value="*Belum ada data sitasi.*")

                # Feedback Section
                with gr.Group(elem_id="feedback-panel"):
                    with gr.Row():
                        thumbs_up_btn = gr.Button("👍 Jawaban Tepat & Membantu", size="sm")
                        thumbs_down_btn = gr.Button("👎 Kurang Tepat / Salah Sitasi", size="sm")

                    feedback_comment = gr.Textbox(
                        label="Catatan atau masukan (opsional):",
                        placeholder="Misal: Nomor pasal berbeda dengan ketentuan terbaru...",
                        lines=1,
                    )
                    feedback_submit_btn = gr.Button("Kirim Feedback / Submit Feedback", size="sm")
                    feedback_status = gr.HTML(value="")

        # Interaction Handlers
        @spaces.GPU(duration=120)
        def on_ask(
            query: str,
            lang: str,
            variant: str,
        ) -> tuple[str, str, str, str, str, Any]:
            nonlocal active_pipeline
            if not query or not query.strip():
                return (
                    "⚠️ *Harap masukkan pertanyaan terlebih dahulu.*",
                    "<span style='color: #f59e0b;'>⚠️ Pertanyaan kosong</span>",
                    "",
                    "*Tidak ada data sitasi.*",
                    "",
                    None,
                )

            clean_q = query.strip()
            req_id = f"demo-{int(time.time() * 1000)}"

            # Initialize pipeline lazily if needed
            if active_pipeline is None:
                try:
                    logger.info("Initializing EasyPasal RAGPipeline for UI...")
                    active_pipeline = RAGPipeline.from_defaults(
                        provider=os.getenv("EASYPASAL_LLM_PROVIDER", "auto")
                    )
                except Exception as exc:
                    logger.warning("Pipeline fallback to mock in UI: %s", exc)
                    active_pipeline = RAGPipeline.from_defaults(provider="mock")

            try:
                max_tokens_env = os.getenv("EASYPASAL_MAX_TOKENS")
                eff_max_tokens = int(max_tokens_env) if max_tokens_env else None
                strict_env_val = os.getenv("EASYPASAL_STRICT_GROUNDING")
                strict_env = (
                    strict_env_val.lower() in ("1", "true", "yes")
                    if strict_env_val is not None
                    else False
                )
                status_filter_env = os.getenv("EASYPASAL_STATUS_FILTER", "berlaku")
                output = active_pipeline.ask(
                    clean_q,
                    language=lang,
                    variant=variant,
                    max_tokens=eff_max_tokens,
                    strict_grounding=strict_env,
                    status_filter=status_filter_env,
                )
            except Exception as exc:
                logger.error("Error during answer generation: %s", exc)
                return (
                    f"⚠️ Terjadi kendala teknis saat memproses jawaban: `{exc}`",
                    "<span style='color: #ef4444;'>❌ Error Sistem</span>",
                    "",
                    "*Tidak ada data sitasi.*",
                    req_id,
                    None,
                )

            # Record in RequestLogger if available
            if request_logger is not None:
                try:
                    top_scores = [c.score for c in output.retrieved_chunks]
                    doc_ids = [c.doc_id for c in output.retrieved_chunks]
                    top1 = top_scores[0] if top_scores else None
                    rec = RequestLogRecord(
                        request_id=req_id,
                        variant=variant,
                        language=lang,
                        question_length=len(clean_q),
                        question_redacted=clean_q,
                        retrieval_score_top1=top1,
                        retrieval_scores_topk=top_scores,
                        retrieved_doc_ids=doc_ids,
                        abstained=output.abstained,
                        routed=bool(output.metadata.get("routed", False)),
                        theme=output.metadata.get("theme"),
                        retrieval_latency_ms=output.metadata.get("retrieval_ms"),
                        generation_latency_ms=output.metadata.get("generation_ms"),
                        total_latency_ms=output.latency_ms,
                        prompt_tokens=output.prompt_tokens,
                        completion_tokens=output.completion_tokens,
                        citation_count=len(output.citations),
                        model_name=output.model_name,
                    )
                    request_logger.log_request(rec)
                except Exception as log_exc:
                    logger.warning("Failed to log request in UI: %s", log_exc)

            # Determine status badge based on abstention and post-hoc citation verification
            v_data = (
                output.metadata.get("citation_verification")
                if isinstance(output.metadata, dict)
                else None
            )
            is_valid_cit = v_data.get("is_valid", True) if v_data else True

            if output.abstained or len(output.citations) == 0:
                badge_html = (
                    "<span style='background:#fee2e2; color:#b91c1c; padding:3px 10px; "
                    "border-radius:4px; font-weight:600;'>⚠️ Tidak Ada Dasar Hukum / Abstained</span>"
                )
            elif not is_valid_cit:
                badge_html = (
                    "<span style='background:#fef3c7; color:#b45309; padding:3px 10px; "
                    "border-radius:4px; font-weight:600;'>⚠️ Sitasi Belum Terverifikasi / Perlu Tinjauan</span>"
                )
            else:
                badge_html = (
                    "<span style='background:#dcfce7; color:#15803d; padding:3px 10px; "
                    "border-radius:4px; font-weight:600;'>✅ Dijawab Berdasar Regulasi</span>"
                )

            telemetry_html = (
                f"<span style='font-size:0.8rem; color:#64748b;'>"
                f"Latency: {output.latency_ms:.1f}ms &bull; "
                f"Model: {output.model_name or 'EasyPasal'} &bull; "
                f"Variant: <code>{variant}</code>"
                f"</span>"
            )

            citations_md = format_citations_markdown(output)
            return (
                output.answer,
                badge_html,
                telemetry_html,
                citations_md,
                req_id,
                output,
            )

        def on_clear() -> tuple[str, str, str, str, str, str, Any]:
            return (
                "",
                "*Silakan masukkan pertanyaan kepatuhan di atas untuk melihat jawaban berdasar pasal.*",
                "<span style='color: #64748b;'>Status: Menunggu pertanyaan</span>",
                "",
                "*Belum ada data sitasi.*",
                "",
                None,
            )

        def on_feedback_rating(
            req_id: str,
            rating: int,
        ) -> str:
            if not req_id:
                return "<span style='color: #f59e0b;'>⚠️ Ajukan pertanyaan terlebih dahulu sebelum memberi rating.</span>"

            if request_logger is not None:
                try:
                    rating_val: Any = 1 if rating > 0 else -1
                    rec = FeedbackLogRecord(
                        request_id=req_id,
                        rating=rating_val,
                        comment_redacted=None,
                        tags=["ui_rating"],
                    )
                    request_logger.log_feedback(rec)
                except Exception as e:
                    logger.warning("Feedback logging failed: %s", e)

            icon = "👍" if rating > 0 else "👎"
            return f"<span style='color: #10b981;'>{icon} Terima kasih atas penilaian Anda!</span>"

        def on_feedback_submit(
            req_id: str,
            comment: str,
        ) -> str:
            if not req_id:
                return "<span style='color: #f59e0b;'>⚠️ Ajukan pertanyaan terlebih dahulu sebelum mengirim masukan.</span>"

            clean_comment = comment.strip() if comment else ""
            if request_logger is not None and clean_comment:
                try:
                    rating_val: Any = -1 if "salah" in clean_comment.lower() else 1
                    rec = FeedbackLogRecord(
                        request_id=req_id,
                        rating=rating_val,
                        comment_redacted=clean_comment,
                        tags=["user_comment"],
                    )
                    request_logger.log_feedback(rec)
                except Exception as e:
                    logger.warning("Feedback comment logging failed: %s", e)

            return "<span style='color: #10b981;'>✅ Masukan berhasil dikirim. Terima kasih telah membantu menyempurnakan EasyPasal!</span>"

        # Wire UI event triggers
        submit_btn.click(
            fn=on_ask,
            inputs=[question_input, lang_toggle, variant_selector],
            outputs=[
                answer_output,
                status_badge,
                telemetry_info,
                citations_output,
                current_request_id,
                last_output_state,
            ],
        )

        question_input.submit(
            fn=on_ask,
            inputs=[question_input, lang_toggle, variant_selector],
            outputs=[
                answer_output,
                status_badge,
                telemetry_info,
                citations_output,
                current_request_id,
                last_output_state,
            ],
        )

        clear_btn.click(
            fn=on_clear,
            inputs=[],
            outputs=[
                question_input,
                answer_output,
                status_badge,
                telemetry_info,
                citations_output,
                current_request_id,
                last_output_state,
            ],
        )

        thumbs_up_btn.click(
            fn=lambda r_id: on_feedback_rating(r_id, 1),
            inputs=[current_request_id],
            outputs=[feedback_status],
        )

        thumbs_down_btn.click(
            fn=lambda r_id: on_feedback_rating(r_id, -1),
            inputs=[current_request_id],
            outputs=[feedback_status],
        )

        feedback_submit_btn.click(
            fn=on_feedback_submit,
            inputs=[current_request_id, feedback_comment],
            outputs=[feedback_status],
        )

    return demo  # type: ignore[no-any-return]


def main() -> None:
    """CLI launcher for the EasyPasal Gradio Web UI."""
    parser = argparse.ArgumentParser(description="Launch EasyPasal Gradio UI.")
    parser.add_argument(
        "--host",
        default=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        help="Network interface host to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        help="Port to listen on (default: 7860 for Hugging Face Spaces)",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "mock", "ollama", "llamacpp", "transformers", "hf"],
        default=os.getenv("EASYPASAL_LLM_PROVIDER", "auto"),
        help="LLM provider for RAGPipeline (default: auto or EASYPASAL_LLM_PROVIDER)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable gradio.live link",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(
        "Initializing EasyPasal UI with provider=%s on %s:%d...",
        args.provider,
        args.host,
        args.port,
    )

    # Initialize RAGPipeline with requested provider
    pipeline = None
    try:
        pipeline = RAGPipeline.from_defaults(provider=args.provider)
        logger.info("RAGPipeline loaded successfully.")
    except Exception as exc:
        logger.error("Failed to initialize RAGPipeline: %s", exc)
        pipeline = None

    # Initialize RequestLogger
    request_logger = RequestLogger(
        collector_url=os.getenv("EASYPASAL_COLLECTOR_URL"),
        collector_token=os.getenv("EASYPASAL_COLLECTOR_TOKEN", "easypasal-vps-secret-token-v1"),
    )

    app = build_ui(pipeline=pipeline, request_logger=request_logger)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
