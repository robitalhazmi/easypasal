"""Unit and regression tests for GEPA Prompt Optimization (T5.1 / Experiment E5)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from eval.labeling_helper import GoldItem
from eval.split_eval import load_gold_items
from experiments.e5_gepa.adapter import EasyPasalGEPAAdapter, compute_lexical_f1
from experiments.e5_gepa.dataset import split_dev_for_gepa
from experiments.e5_gepa.proposer import (
    DIRECTIVE_CITATION_SYNTAX,
    DIRECTIVE_STRICT_ABSTENTION,
    EasyPasalReflectiveProposer,
)
from experiments.e5_gepa.run_experiment import (
    PromptItemEval,
    SplitEvaluationSummary,
    combine_split_evaluations,
)

from easypasal.generation.backends import MockGeneratorBackend
from easypasal.generation.pipeline import RAGPipeline
from easypasal.generation.prompts import (
    SYSTEM_PROMPT_ID,
    build_rag_prompt,
)
from easypasal.retrieval.models import RetrievedChunk


@pytest.fixture
def sample_gold_items() -> list[GoldItem]:
    """Sample real gold items from dev.jsonl for realistic testing."""
    return load_gold_items("data/gold/dev.jsonl")[:6]


def test_split_dev_for_gepa(sample_gold_items: list[GoldItem]) -> None:
    """Verify stratified partitioning into e5_train and e5_val."""
    train_items, val_items = split_dev_for_gepa(sample_gold_items, train_ratio=0.6, seed=42)
    assert len(train_items) + len(val_items) == len(sample_gold_items)
    assert len(train_items) >= 2
    assert len(val_items) >= 1

    # Check IDs are strictly disjoint
    train_ids = {it.id for it in train_items}
    val_ids = {it.id for it in val_items}
    assert train_ids.isdisjoint(val_ids)


def test_adapter_evaluation_and_reflection(sample_gold_items: list[GoldItem]) -> None:
    """Test EasyPasalGEPAAdapter scoring, trajectory collection, and reflective dataset generation."""
    mock_retriever = MagicMock()
    mock_chunk = RetrievedChunk(
        chunk_id="chunk_01",
        score=0.92,
        citation_label="POJK 22 Tahun 2023, Pasal 75 ayat (5)",
        doc_id="pojk_22_2023",
        doc_number="POJK 22/2023",
        text="PUJK wajib menyelesaikan pengaduan dalam jangka waktu 10 hari kerja. [POJK 22 Tahun 2023, Pasal 75 ayat (5)]",
        status="berlaku",
    )
    mock_retriever.retrieve.return_value.chunks = [mock_chunk]
    mock_retriever.retrieve.return_value.duration_ms = 5.0
    mock_retriever.retrieve.return_value.total_candidates = 1

    generator = MockGeneratorBackend()
    pipeline = RAGPipeline(retriever=mock_retriever, generator=generator)

    adapter = EasyPasalGEPAAdapter(pipeline=pipeline)
    candidate = {"system_prompt": SYSTEM_PROMPT_ID}

    batch_eval = adapter.evaluate(sample_gold_items[:3], candidate, capture_traces=True)
    assert len(batch_eval.scores) == 3
    assert len(batch_eval.outputs) == 3
    assert batch_eval.trajectories is not None
    assert len(batch_eval.trajectories) == 3

    # Validate reflective dataset schema
    refl_data = adapter.make_reflective_dataset(candidate, batch_eval, ["system_prompt"])
    assert "system_prompt" in refl_data
    for record in refl_data["system_prompt"]:
        assert "Inputs" in record
        assert "Generated Outputs" in record
        assert "Feedback" in record


def test_reflective_proposer_mutation() -> None:
    """Verify that EasyPasalReflectiveProposer mutates prompts incorporating compliance directives."""
    proposer = EasyPasalReflectiveProposer(base_prompt=SYSTEM_PROMPT_ID)
    seed = {"system_prompt": SYSTEM_PROMPT_ID}

    # Simulate traces highlighting abstention and citation failures
    mock_refl = {
        "system_prompt": [
            {
                "Inputs": {"question": "Resep rendang?"},
                "Generated Outputs": {"answer": "Resep rendang adalah...", "abstained": False},
                "Feedback": "FAILED ABSTENTION: Query is out-of-scope",
            },
            {
                "Inputs": {"question": "Berapa modal minimum?"},
                "Generated Outputs": {"answer": "Modal minimum Rp3T", "citations": []},
                "Feedback": "MISSED CITATIONS: Expected [POJK 12/2020]",
            },
        ]
    }

    mutated = proposer.propose(seed, mock_refl, ["system_prompt"])
    new_prompt = mutated["system_prompt"]
    assert len(new_prompt) > len(SYSTEM_PROMPT_ID)
    assert "PETUNJUK KEPATUHAN TAMBAHAN" in new_prompt
    assert DIRECTIVE_STRICT_ABSTENTION in new_prompt
    assert DIRECTIVE_CITATION_SYNTAX in new_prompt


def test_combine_split_evaluations() -> None:
    """Test combining two disjoint split evaluations into a unified summary."""
    item_a = PromptItemEval(
        item_id="1",
        category="CatA",
        question="Q1",
        gold_answer="A1",
        gold_citations=["Cit1"],
        should_abstain=False,
        generated_answer="Gen1",
        extracted_citations=["Cit1"],
        did_abstain=False,
        citation_f1_article=1.0,
        citation_f1_exact=1.0,
        citation_precision_article=1.0,
        citation_recall_article=1.0,
        abstention_correct=True,
        lexical_f1=0.8,
        grounded_ratio=1.0,
        hallucinated_pasal=[],
        ungrounded_citations=[],
        composite_score=0.9,
        latency_ms=100.0,
    )
    item_b = PromptItemEval(
        item_id="2",
        category="CatB",
        question="Q2",
        gold_answer="A2",
        gold_citations=[],
        should_abstain=True,
        generated_answer="Berdasarkan regulasi...",
        extracted_citations=[],
        did_abstain=True,
        citation_f1_article=0.0,
        citation_f1_exact=0.0,
        citation_precision_article=0.0,
        citation_recall_article=0.0,
        abstention_correct=True,
        lexical_f1=1.0,
        grounded_ratio=1.0,
        hallucinated_pasal=[],
        ungrounded_citations=[],
        composite_score=1.0,
        latency_ms=80.0,
    )

    sum_a = SplitEvaluationSummary(
        split_name="train",
        variant_name="test_var",
        num_items=1,
        mean_citation_f1_article=1.0,
        mean_citation_f1_exact=1.0,
        mean_citation_precision=1.0,
        mean_citation_recall=1.0,
        abstention_accuracy=1.0,
        mean_lexical_f1=0.8,
        grounding_fidelity=1.0,
        mean_composite_score=0.9,
        mean_latency_ms=100.0,
        items=[item_a],
    )
    sum_b = SplitEvaluationSummary(
        split_name="val",
        variant_name="test_var",
        num_items=1,
        mean_citation_f1_article=0.0,
        mean_citation_f1_exact=0.0,
        mean_citation_precision=0.0,
        mean_citation_recall=0.0,
        abstention_accuracy=1.0,
        mean_lexical_f1=1.0,
        grounding_fidelity=1.0,
        mean_composite_score=1.0,
        mean_latency_ms=80.0,
        items=[item_b],
    )

    combined = combine_split_evaluations(sum_a, sum_b, "dev", "test_var", num_bootstrap=50)
    assert combined.num_items == 2
    assert combined.split_name == "dev"
    assert combined.mean_citation_f1_article == 0.5
    assert combined.abstention_accuracy == 1.0
    assert combined.mean_composite_score == 0.95
    assert len(combined.items) == 2


def test_build_rag_prompt_variants() -> None:
    """Verify build_rag_prompt selects GEPA vs Baseline prompts correctly."""
    chunk = RetrievedChunk(
        chunk_id="c1",
        score=0.9,
        citation_label="POJK 1/2020",
        doc_id="d1",
        doc_number="POJK 1/2020",
        text="Isi pasal",
    )

    # 1. Default variant ('gepa')
    prompt_gepa_id = build_rag_prompt("Pertanyaan?", [chunk], language="id", prompt_variant="gepa")
    assert "LARANGAN HALUSINASI PASAL" in prompt_gepa_id
    assert "STATUS REGULASI & KETENTUAN PENGGANTI" in prompt_gepa_id

    prompt_gepa_en = build_rag_prompt("Question?", [chunk], language="en", prompt_variant="gepa")
    assert "ANTI-HALLUCINATION CLAUSE" in prompt_gepa_en
    assert "REGULATION STATUS & PRECEDENCE" in prompt_gepa_en

    # 2. Baseline variant
    prompt_base_id = build_rag_prompt(
        "Pertanyaan?", [chunk], language="id", prompt_variant="baseline"
    )
    assert "LARANGAN HALUSINASI PASAL" not in prompt_base_id
    assert "ATURAN MUTLAK PENJAWABAN" in prompt_base_id

    # 3. Explicit override
    prompt_override = build_rag_prompt(
        "Pertanyaan?", [chunk], system_prompt_override="CUSTOM SYSTEM PROMPT"
    )
    assert "CUSTOM SYSTEM PROMPT" in prompt_override


def test_compute_lexical_f1() -> None:
    """Verify lexical token F1 computation."""
    assert compute_lexical_f1("modal minimum bank", "modal minimum bank") == 1.0
    assert compute_lexical_f1("modal minimum bank", "berdasarkan hukum pidana") == 0.0
    assert 0.0 < compute_lexical_f1("modal minimum bank umum", "modal minimum bank") < 1.0
