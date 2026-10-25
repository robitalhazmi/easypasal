"""Generation module: prompt assembly, LLM backends, citation extraction, and RAG pipeline."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from easypasal.generation.backends import (
        GeneratorBackend,
        LlamaCppBackend,
        LLMGenerationResult,
        MockGeneratorBackend,
        OllamaBackend,
        OpenAICompatibleBackend,
        get_generator_backend,
        get_process_peak_memory_mb,
        trim_incomplete_sentence,
    )
    from easypasal.generation.citations import (
        check_abstention,
        extract_citations,
        verify_citation_coverage,
    )
    from easypasal.generation.models import (
        GenerationInput,
        GenerationOutput,
    )
    from easypasal.generation.pipeline import RAGPipeline
    from easypasal.generation.prompts import (
        STANDARD_ABSTAIN_EN,
        STANDARD_ABSTAIN_ID,
        build_rag_prompt,
        detect_language,
        format_passages,
    )

__all__ = [
    "GenerationInput",
    "GenerationOutput",
    "GeneratorBackend",
    "LLMGenerationResult",
    "LlamaCppBackend",
    "MockGeneratorBackend",
    "OllamaBackend",
    "OpenAICompatibleBackend",
    "RAGPipeline",
    "STANDARD_ABSTAIN_EN",
    "STANDARD_ABSTAIN_ID",
    "build_rag_prompt",
    "check_abstention",
    "detect_language",
    "extract_citations",
    "format_passages",
    "get_generator_backend",
    "get_process_peak_memory_mb",
    "trim_incomplete_sentence",
    "verify_citation_coverage",
]

_BACKEND_SYMBOLS = {
    "GeneratorBackend",
    "LLMGenerationResult",
    "LlamaCppBackend",
    "MockGeneratorBackend",
    "OllamaBackend",
    "OpenAICompatibleBackend",
    "get_generator_backend",
    "get_process_peak_memory_mb",
    "trim_incomplete_sentence",
}

_CITATION_SYMBOLS = {
    "check_abstention",
    "extract_citations",
    "verify_citation_coverage",
}

_MODEL_SYMBOLS = {
    "GenerationInput",
    "GenerationOutput",
}

_PIPELINE_SYMBOLS = {
    "RAGPipeline",
}

_PROMPT_SYMBOLS = {
    "STANDARD_ABSTAIN_EN",
    "STANDARD_ABSTAIN_ID",
    "build_rag_prompt",
    "detect_language",
    "format_passages",
}


def __getattr__(name: str) -> object:
    if name in _BACKEND_SYMBOLS:
        import easypasal.generation.backends as _backends

        return getattr(_backends, name)
    if name in _CITATION_SYMBOLS:
        import easypasal.generation.citations as _citations

        return getattr(_citations, name)
    if name in _MODEL_SYMBOLS:
        import easypasal.generation.models as _models

        return getattr(_models, name)
    if name in _PIPELINE_SYMBOLS:
        import easypasal.generation.pipeline as _pipeline

        return getattr(_pipeline, name)
    if name in _PROMPT_SYMBOLS:
        import easypasal.generation.prompts as _prompts

        return getattr(_prompts, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
