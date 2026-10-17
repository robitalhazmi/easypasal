"""Gradio Web User Interface for EasyPasal (Plan §8.3 / Task T7.6).

Provides an interactive regulatory compliance assistant with bilingual support,
collapsible statutory citation inspector, user feedback collection,
and transparent abstention handling.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any

import gradio as gr

from easypasal.generation.pipeline import RAGPipeline
from easypasal.monitoring.logger import RequestLogger
from easypasal.monitoring.schemas import FeedbackLogRecord, RequestLogRecord

logger = logging.getLogger("easypasal.ui.app")

DEFAULT_EXAMPLES = [
    [
        "Apa syarat modal disetor minimum untuk pendirian Bank Umum konvensional?",
        "id",
        "baseline_v0.1",
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
    retrieved_chunks = getattr(output, "retrieved_chunks", [])
    if not retrieved_chunks:
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
        max-width: 1100px !important;
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

        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="Tulis Pertanyaan Regulasi / Enter Compliance Question:",
                    placeholder="Contoh: Apa syarat modal disetor minimum untuk pendirian Bank Umum?",
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

            with gr.Column(scale=2):
                gr.Markdown("### ⚡ Pertanyaan Uji / Example Queries")
                gr.Examples(
                    examples=DEFAULT_EXAMPLES,
                    inputs=[question_input, lang_toggle, variant_selector],
                    label="Pilih pertanyaan untuk menguji sistem:",
                )

        # Response Section
        with gr.Group():
            gr.Markdown("### 📋 Jawaban Kepatuhan / Grounded Answer")
            answer_output = gr.Markdown(
                value="*Silakan masukkan pertanyaan kepatuhan di atas untuk melihat jawaban berdasar pasal.*"
            )

            with gr.Row():
                status_badge = gr.HTML(
                    value="<span style='color: #64748b;'>Status: Menunggu pertanyaan</span>"
                )
                telemetry_info = gr.HTML(value="")

        # Collapsible Citations Inspector
        with gr.Accordion("🔎 Rincian Sitasi & Potongan Naskah Regulasi", open=False):
            citations_output = gr.Markdown(value="*Belum ada data sitasi.*")

        # Feedback Section
        with gr.Group():
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
                    active_pipeline = RAGPipeline.from_defaults(provider="auto")
                except Exception as exc:
                    logger.warning("Pipeline fallback to mock in UI: %s", exc)
                    active_pipeline = RAGPipeline.from_defaults(provider="mock")

            try:
                output = active_pipeline.ask(
                    clean_q,
                    language=lang,
                    variant=variant,
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

            # Determine status badge
            if output.abstained:
                badge_html = (
                    "<span style='background:#fee2e2; color:#b91c1c; padding:3px 10px; "
                    "border-radius:4px; font-weight:600;'>⚠️ Tidak Ada Dasar Hukum / Abstained</span>"
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
        choices=["auto", "mock", "ollama", "llamacpp"],
        default="auto",
        help="LLM provider for RAGPipeline (default: auto)",
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
