"""End-to-end RAG pipeline combining hybrid retrieval with citation-grounded generation."""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from easypasal.config import EasyPasalConfig
from easypasal.generation.backends import GeneratorBackend, get_generator_backend
from easypasal.generation.citations import check_abstention, extract_citations
from easypasal.generation.models import GenerationOutput
from easypasal.generation.prompts import build_rag_prompt, detect_language
from easypasal.guardrails.question_router import QuestionRouter
from easypasal.retrieval.models import RetrievedChunk
from easypasal.retrieval.retriever import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("easypasal.generation.pipeline")


class RAGPipeline:
    """Unified Question-Answering pipeline for Indonesian financial regulation."""

    config: EasyPasalConfig | None = None

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: GeneratorBackend,
        config: EasyPasalConfig | None = None,
        router: QuestionRouter | None = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.config = config
        self.router = router or (
            QuestionRouter(config=config.guardrails if config else None)
            if (config is None or config.guardrails.enable_router)
            else None
        )

    @classmethod
    def from_config(
        cls,
        config: EasyPasalConfig | None = None,
        config_path: Path | str | None = None,
        override_provider: str | None = None,
    ) -> RAGPipeline:
        """Initialize RAG pipeline from a validated EasyPasalConfig instance or YAML file.

        Args:
            config: Optional pre-loaded EasyPasalConfig instance.
            config_path: Optional path to a YAML configuration file.
            override_provider: Optional provider override ('auto', 'mock', 'llamacpp', etc.).

        Returns:
            Initialized RAGPipeline aligned with the specified configuration.
        """
        if config is None:
            config = EasyPasalConfig.load_default(config_path)

        logger.info(
            "Initializing RAGPipeline from config variant %r (version=%s)...",
            config.variant_id,
            config.version,
        )

        retriever = HybridRetriever.from_paths(
            index_dir=config.retrieval.index_dir,
            strategy=config.chunking.strategy,
            model_name=config.embeddings.model_name,
            collection_name=config.retrieval.collection_name,
        )

        provider = override_provider or config.generator.cpu_default.provider
        generator = get_generator_backend(
            provider=provider,
            repo_id=config.generator.cpu_default.repo_id,
            filename=config.generator.cpu_default.filename,
            n_ctx=config.generator.cpu_default.n_ctx,
            n_threads=config.generator.cpu_default.n_threads,
        )

        return cls(retriever=retriever, generator=generator, config=config)

    @classmethod
    def from_defaults(
        cls,
        index_dir: Path | str = "data/index",
        strategy: str = "structure_aware",
        provider: str = "auto",
        model_name: str | None = None,
        model_path: Path | str | None = None,
        base_url: str | None = None,
        config_path: Path | str | None = None,
    ) -> RAGPipeline:
        """Initialize RAG pipeline using default configuration and optional parameter overrides."""
        try:
            config = EasyPasalConfig.load_default(config_path)
        except Exception as cfg_exc:
            logger.warning(
                "Could not load config from %s: %s. Using default EasyPasalConfig.",
                config_path,
                cfg_exc,
            )
            config = EasyPasalConfig()

        effective_strategy = strategy or config.chunking.strategy
        effective_model = model_name or config.embeddings.model_name
        env_provider = os.getenv("EASYPASAL_LLM_PROVIDER")
        effective_provider = provider if provider != "auto" else (env_provider or "auto")

        # Check if prebuilt indexes exist; if not, attempt auto-download from Hugging Face Space
        q_dir = Path(index_dir) / "qdrant"
        b_dir = Path(index_dir) / "bm25"
        if not (q_dir.exists() and any(q_dir.iterdir())) or not (
            b_dir.exists() and any(b_dir.iterdir())
        ):
            try:
                from easypasal.index.download import download_prebuilt_index

                logger.info(
                    "Index directories not fully populated at %s. Fetching prebuilt indexes from Hugging Face...",
                    index_dir,
                )
                download_prebuilt_index(target_dir=index_dir)
            except Exception as dl_exc:
                logger.warning(
                    "Auto-download of prebuilt index deferred/failed (%s). Continuing with existing files.",
                    dl_exc,
                )

        logger.info("Initializing HybridRetriever from %s...", index_dir)
        retriever = HybridRetriever.from_paths(
            index_dir=index_dir,
            strategy=effective_strategy,
            model_name=effective_model,
            collection_name=config.retrieval.collection_name,
        )

        logger.info("Initializing GeneratorBackend (provider=%r)...", effective_provider)
        try:
            generator = get_generator_backend(
                provider=effective_provider,
                model_name=model_name,
                model_path=model_path,
                base_url=base_url,
                repo_id=config.generator.cpu_default.repo_id,
                filename=config.generator.cpu_default.filename,
                n_ctx=config.generator.cpu_default.n_ctx,
                n_threads=config.generator.cpu_default.n_threads,
            )
        except Exception as gen_exc:
            logger.warning(
                "Generator backend %r failed (%s), attempting auto fallback.",
                effective_provider,
                gen_exc,
            )
            if effective_provider != "auto":
                try:
                    generator = get_generator_backend(
                        provider="auto",
                        model_name=model_name,
                        model_path=model_path,
                        base_url=base_url,
                    )
                except Exception:
                    generator = get_generator_backend(
                        provider="mock",
                        model_name=model_name or "mock-generator",
                    )
            else:
                generator = get_generator_backend(
                    provider="mock",
                    model_name=model_name or "mock-generator",
                )

        return cls(retriever=retriever, generator=generator, config=config)

    def ask(
        self,
        question: str,
        language: str | None = None,
        top_k: int | None = None,
        mode: str | None = None,
        status_filter: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt_override: str | None = None,
        prompt_variant: str = "gepa",
        variant: str | None = None,
        verify_citations: bool | None = None,
        strict_grounding: bool | None = None,
        retrieved_chunks: list[RetrievedChunk] | None = None,
        enable_router: bool | None = None,
    ) -> GenerationOutput:
        """Answer a regulatory question with citations and cite-or-abstain behavior.

        Args:
            question: User query in Indonesian or English.
            language: Optional language code ('id' or 'en'). Detected automatically if omitted.
            top_k: Number of relevant regulatory passages to retrieve (default: from config or 5).
            mode: Retrieval strategy ('dense', 'hybrid', or 'bm25') (default: from config or 'dense').
            status_filter: Regulation status constraint (default: from config or 'berlaku').
            temperature: LLM sampling temperature (default: from config or 0.1).
            max_tokens: Maximum tokens in generated completion (default: from config or 512).
            system_prompt_override: Custom system instructions if desired.
            prompt_variant: Prompt template variant ('gepa' or 'baseline').
            variant: Optional high-level pipeline variant ID (e.g. 'baseline_v0.1', 'gepa_v0.2', 'A', 'B').
            verify_citations: Validate that cited pasal/ayat exist in retrieved passages (default: True).
            strict_grounding: Fall back to abstention if ungrounded citations are detected (default: False).
            retrieved_chunks: Optional pre-retrieved chunks to reuse (skips redundant retrieval in paired A/B replays).
            enable_router: Enable front-line guardrail question routing and safety triage (default: from config or True).

        Returns:
            GenerationOutput with answer text, extracted citations, and abstention status.
        """
        start_time = time.perf_counter()
        clean_question = question.strip()

        # Resolve effective parameters from attached config or baseline defaults
        eff_top_k = (
            top_k if top_k is not None else (self.config.retrieval.top_k if self.config else 5)
        )
        eff_mode = (
            mode if mode is not None else (self.config.retrieval.mode if self.config else "dense")
        )
        eff_status = (
            status_filter
            if status_filter is not None
            else (self.config.guardrails.status_filter if self.config else "berlaku")
        )
        eff_temp = (
            temperature
            if temperature is not None
            else (self.config.generator.cpu_default.temperature if self.config else 0.1)
        )
        eff_max_tokens = (
            max_tokens
            if max_tokens is not None
            else (self.config.generator.cpu_default.max_tokens if self.config else 512)
        )
        eff_verify = (
            verify_citations
            if verify_citations is not None
            else (self.config.guardrails.verify_citations if self.config else True)
        )
        eff_strict = (
            strict_grounding
            if strict_grounding is not None
            else (self.config.guardrails.strict_grounding if self.config else False)
        )
        eff_enable_router = (
            enable_router
            if enable_router is not None
            else (self.config.guardrails.enable_router if self.config else True)
        )

        # Resolve effective variant and prompt variant
        eff_variant = variant or (self.config.variant_id if self.config else "baseline_v0.1")
        eff_prompt_variant = prompt_variant
        if variant is not None:
            v_lower = variant.lower()
            if "gepa" in v_lower or v_lower == "b":
                eff_prompt_variant = "gepa"
            elif "baseline" in v_lower or v_lower == "a":
                eff_prompt_variant = "baseline"

        # 1. Detect Question Language
        lang = language if language is not None else detect_language(clean_question)

        # 2. Front-Line Guardrail Router Triage (Task T6.4 / Experiment E7)
        route_decision = None
        if eff_enable_router and self.router is not None:
            route_decision = self.router.route(clean_question, language=lang)
            if route_decision.should_block:
                total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                logger.info(
                    "Query blocked by router guardrail (theme=%r, reason=%s): %s",
                    route_decision.theme,
                    route_decision.refusal_reason,
                    clean_question[:60],
                )
                return GenerationOutput(
                    question=clean_question,
                    answer=route_decision.refusal_response or "",
                    language=lang,
                    citations=[],
                    abstained=True,
                    retrieved_chunks=[],
                    prompt_tokens=0,
                    completion_tokens=0,
                    latency_ms=total_latency_ms,
                    tokens_per_second=0.0,
                    model_name="qlora_router",
                    metadata={
                        "variant": eff_variant,
                        "prompt_variant": eff_prompt_variant,
                        "routed": True,
                        "theme": route_decision.theme,
                        "is_answerable": route_decision.is_answerable,
                        "is_injection": route_decision.is_injection,
                        "refusal_reason": route_decision.refusal_reason,
                        "routing_latency_ms": route_decision.latency_ms,
                    },
                )

        # 3. Retrieve Candidate Passages (or reuse pre-retrieved chunks)
        if retrieved_chunks is not None:
            chunks = retrieved_chunks
            retrieval_ms = 0.0
            total_candidates = len(chunks)
        else:
            retrieval_res = self.retriever.retrieve(
                query=clean_question,
                top_k=eff_top_k,
                mode=eff_mode,
                status_filter=eff_status,
            )
            chunks = retrieval_res.chunks
            retrieval_ms = retrieval_res.duration_ms
            total_candidates = retrieval_res.total_candidates

        # 3. Assemble Grounded Prompt
        prompt = build_rag_prompt(
            question=clean_question,
            chunks=chunks,
            language=lang,
            system_prompt_override=system_prompt_override,
            prompt_variant=eff_prompt_variant,
        )

        # 4. Generate Completion via Backend
        gen_res = self.generator.generate(
            prompt=prompt,
            temperature=eff_temp,
            max_tokens=eff_max_tokens,
        )

        # 5. Clean Incomplete Sentence Fragments, Extract Citations, and Check Abstention
        from easypasal.generation.backends import trim_incomplete_sentence

        answer_text = trim_incomplete_sentence(gen_res.text)
        citations = extract_citations(answer_text)
        abstained = check_abstention(answer_text, language=lang)

        # 6. Post-Hoc Citation Verification (T1.7)
        verification_dict = None
        if eff_verify:
            from easypasal.guardrails.citation_verifier import verify_citations as verify_fn

            v_res = verify_fn(
                answer=answer_text,
                retrieved_chunks=chunks,
                citations=citations,
            )
            verification_dict = {
                "is_valid": v_res.is_valid,
                "total_citations": v_res.total_citations,
                "grounded_count": v_res.grounded_count,
                "ungrounded_count": v_res.ungrounded_count,
                "ungrounded_citations": v_res.ungrounded_citations,
                "hallucinated_pasal": v_res.hallucinated_pasal,
            }

            if not v_res.is_valid and eff_strict and not abstained:
                logger.warning(
                    "Strict grounding triggered: ungrounded citations detected %s. Reverting to abstention.",
                    v_res.ungrounded_citations,
                )
                from easypasal.generation.prompts import (
                    STANDARD_ABSTAIN_EN,
                    STANDARD_ABSTAIN_ID,
                )

                answer_text = STANDARD_ABSTAIN_EN if lang == "en" else STANDARD_ABSTAIN_ID
                abstained = True
                citations = []

        total_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return GenerationOutput(
            question=clean_question,
            answer=answer_text,
            language=lang,
            citations=citations,
            abstained=abstained,
            retrieved_chunks=chunks,
            prompt_tokens=gen_res.prompt_tokens,
            completion_tokens=gen_res.completion_tokens,
            latency_ms=total_latency_ms,
            tokens_per_second=gen_res.tokens_per_second,
            model_name=self.generator.model_name,
            metadata={
                "variant": eff_variant,
                "prompt_variant": eff_prompt_variant,
                "retrieval_ms": retrieval_ms,
                "generation_ms": gen_res.latency_ms,
                "tokens_per_second": gen_res.tokens_per_second,
                "retrieval_mode": mode,
                "total_candidates": total_candidates,
                "citation_verification": verification_dict,
                "routed": bool(eff_enable_router and self.router),
                "theme": route_decision.theme if route_decision else None,
            },
        )

    def close(self) -> None:
        """Release underlying vector store and client resources."""
        self.retriever.close()


def main() -> None:
    """CLI entrypoint for interactive question answering."""
    parser = argparse.ArgumentParser(
        description="EasyPasal Regulatory Question Answering (RAG with cite-or-abstain)",
    )
    parser.add_argument("--question", "-q", required=True, help="Question to ask")
    parser.add_argument(
        "--provider",
        "-p",
        choices=["auto", "llamacpp", "ollama", "openai", "mock"],
        default="auto",
        help="LLM generator provider (default: auto)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name identifier",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to local GGUF model file",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["hybrid", "dense", "bm25"],
        default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of passages to retrieve (default: 5)",
    )
    parser.add_argument(
        "--status",
        default="berlaku",
        help="Filter regulation status (default: berlaku)",
    )
    parser.add_argument(
        "--strict-grounding",
        action="store_true",
        help="Revert to abstention if ungrounded citations are detected",
    )

    args = parser.parse_args()

    pipeline = RAGPipeline.from_defaults(
        provider=args.provider,
        model_name=args.model_name,
        model_path=args.model_path,
    )

    try:
        output = pipeline.ask(
            question=args.question,
            top_k=args.top_k,
            mode=args.mode,
            status_filter=args.status if args.status != "all" else None,
            strict_grounding=args.strict_grounding,
        )

        print("\n" + "=" * 80)
        print(f"QUESTION: {output.question}")
        print(
            f"LANG: {output.language.upper()} | MODEL: {output.model_name} | "
            f"LATENCY: {output.latency_ms:.2f} ms | ABSTAINED: {output.abstained}"
        )
        print("=" * 80)

        print("\nANSWER:")
        print(output.answer)

        if output.citations:
            print("\nEXTRACTED CITATIONS:")
            for cite in output.citations:
                print(f"  - {cite}")
        else:
            print("\nCITATIONS: None detected")

        v_info = output.metadata.get("citation_verification")
        if v_info and v_info.get("total_citations", 0) > 0:
            status_str = "PASSED" if v_info.get("is_valid") else "FAILED"
            print(
                f"\nCITATION VERIFICATION: {status_str} "
                f"({v_info.get('grounded_count')}/{v_info.get('total_citations')} grounded)"
            )
            if v_info.get("ungrounded_citations"):
                print("  Ungrounded/Hallucinated Citations:")
                for unc in v_info["ungrounded_citations"]:
                    print(f"    - {unc}")

        print("\nRETRIEVED EVIDENCE:")
        for idx, chunk in enumerate(output.retrieved_chunks, start=1):
            print(f"  [{idx}] {chunk.citation_label} (status: {chunk.status})")

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
