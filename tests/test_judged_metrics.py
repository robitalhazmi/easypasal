"""Unit tests for LLM-judged metrics (Faithfulness and Answer Correctness) (T3.3)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from eval.labeling_helper import GoldItem
from eval.metrics.judge import (
    CorrectnessResult,
    FaithfulnessResult,
    JudgedRecord,
    MockJudgeBackend,
    OpenAICompatibleJudgeBackend,
    build_correctness_prompt,
    build_faithfulness_prompt,
    evaluate_correctness,
    evaluate_faithfulness,
    evaluate_judged_item,
    extract_json_payload,
    is_abstention_text,
)
from eval.run_judged_eval import (
    export_calibration_sample,
    load_chunk_text_map,
    main,
    print_judged_report,
    run_judged_evaluation,
)

# ==============================================================================
# 1. Prompt Construction and JSON Parsing Tests
# ==============================================================================


class TestPromptAndParsing:
    """Tests for prompt building and robust JSON extraction."""

    def test_build_faithfulness_prompt(self) -> None:
        """Verify faithfulness prompt incorporates question, context, and answer."""
        prompt = build_faithfulness_prompt(
            question="Berapa modal disetor P2P?",
            context="Modal disetor P2P lending minimal Rp25 miliar.",
            answer="Minimal Rp25 miliar berdasarkan POJK 10/2022.",
        )
        assert "QUESTION:" in prompt
        assert "Berapa modal disetor P2P?" in prompt
        assert "RETRIEVED CONTEXT:" in prompt
        assert "Modal disetor P2P lending minimal Rp25 miliar." in prompt
        assert "GENERATED ANSWER:" in prompt
        assert "Minimal Rp25 miliar berdasarkan POJK 10/2022." in prompt
        assert "faithfulness_score" in prompt

    def test_build_correctness_prompt_answerable(self) -> None:
        """Verify correctness prompt for standard answerable queries."""
        prompt = build_correctness_prompt(
            question="Pertanyaan A?",
            gold_answer="Emas 10 gram.",
            answer="Emas 10 gram.",
            should_abstain=False,
        )
        assert "This question is ANSWERABLE" in prompt
        assert "REFERENCE GOLD ANSWER:" in prompt
        assert "Emas 10 gram." in prompt
        assert "tp_facts" in prompt

    def test_build_correctness_prompt_abstain(self) -> None:
        """Verify correctness prompt for out-of-scope / adversarial queries."""
        prompt = build_correctness_prompt(
            question="Pertanyaan B?",
            gold_answer="Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum.",
            answer="Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum.",
            should_abstain=True,
        )
        assert "This question is OUT OF SCOPE / ADVERSARIAL" in prompt
        assert "The system MUST refuse/abstain" in prompt

    def test_extract_json_payload_clean(self) -> None:
        """Verify extraction from standard JSON strings."""
        raw = '{"faithfulness_score": 1.0, "reasoning": "Grounding verified."}'
        data = extract_json_payload(raw)
        assert data["faithfulness_score"] == 1.0
        assert data["reasoning"] == "Grounding verified."

    def test_extract_json_payload_markdown_codeblock(self) -> None:
        """Verify extraction from fenced markdown json blocks."""
        raw = """Here is my evaluation:
```json
{
  "tp_facts": ["fact A"],
  "correctness_score": 0.85
}
```
Hope this helps!"""
        data = extract_json_payload(raw)
        assert data["tp_facts"] == ["fact A"]
        assert data["correctness_score"] == 0.85

    def test_extract_json_payload_embedded_braces(self) -> None:
        """Verify fallback extraction when braces are surrounded by conversational text."""
        raw = 'Evaluation completed. {"score": 0.95} Thank you.'
        data = extract_json_payload(raw)
        assert data["score"] == 0.95

    def test_extract_json_payload_invalid(self) -> None:
        """Verify graceful empty dict fallback when JSON is malformed."""
        raw = "Not a json response at all."
        assert extract_json_payload(raw) == {}

        raw_broken = '{"score": 0.95, unquoted_key}'
        assert extract_json_payload(raw_broken) == {}


# ==============================================================================
# 2. Abstention and Data Class Serialization Tests
# ==============================================================================


class TestDataModelsAndAbstention:
    """Tests for data models and abstention pattern detection."""

    def test_is_abstention_text(self) -> None:
        """Verify Indonesian and English abstention detection."""
        assert is_abstention_text(
            "Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum."
        )
        assert is_abstention_text(
            "Tidak ditemukan dasar hukum terkait pajak kripto dalam POJK ini."
        )
        assert is_abstention_text("Based on the available regulations, no legal basis was found.")
        assert is_abstention_text("No legal basis was found.")
        assert not is_abstention_text("Modal disetor minimal Rp25 miliar sesuai Pasal 8.")

    def test_data_model_serialization(self) -> None:
        """Verify serialization of FaithfulnessResult, CorrectnessResult, and JudgedRecord."""
        faith = FaithfulnessResult(
            score=0.923456,
            statements=["Claim 1"],
            verdicts=[True],
            reasoning="Valid claim.",
        )
        assert faith.to_dict()["score"] == 0.9235
        assert faith.to_dict()["statements"] == ["Claim 1"]

        corr = CorrectnessResult(
            score=0.85678,
            tp_facts=["Fact 1"],
            fp_facts=[],
            fn_facts=["Fact 2"],
            reasoning="Mostly correct.",
        )
        assert corr.to_dict()["score"] == 0.8568
        assert corr.to_dict()["fn_facts"] == ["Fact 2"]

        record = JudgedRecord(
            id="item_01",
            category="factual_direct",
            theme="Consumer Protection",
            question="Pertanyaan?",
            context="Konteks regulasi...",
            generated_answer="Jawaban...",
            gold_answer="Jawaban...",
            should_abstain=False,
            faithfulness=faith,
            correctness=corr,
        )
        d = record.to_dict()
        assert d["id"] == "item_01"
        assert d["faithfulness"]["score"] == 0.9235
        assert d["correctness"]["score"] == 0.8568


# ==============================================================================
# 3. Judge Backend Tests
# ==============================================================================


class TestJudgeBackends:
    """Tests for Mock, OpenAI-compatible, and evaluation functions."""

    def test_mock_backend_faithfulness_abstention(self) -> None:
        """Mock backend should award 1.0 faithfulness to valid legal abstentions."""
        backend = MockJudgeBackend()
        prompt = build_faithfulness_prompt(
            question="Berapa tarif pajak penghasilan artis?",
            context="POJK ini mengatur tata kelola bank umum.",
            answer="Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum mengenai pajak artis.",
        )
        res_json = backend.generate(prompt)
        data = json.loads(res_json)
        assert data["faithfulness_score"] == 1.0
        assert data["verdicts"] == [True]

    def test_mock_backend_faithfulness_grounded(self) -> None:
        """Mock backend should evaluate lexical overlap for factual answers."""
        backend = MockJudgeBackend()
        prompt = build_faithfulness_prompt(
            question="Berapa modal minimum?",
            context="Ketentuan modal disetor minimum adalah dua puluh lima miliar rupiah.",
            answer="Ketentuan modal minimum adalah dua puluh lima miliar rupiah.",
        )
        res_json = backend.generate(prompt)
        data = json.loads(res_json)
        assert data["faithfulness_score"] == 1.0

    def test_mock_backend_correctness_both_abstain(self) -> None:
        """Both gold and generated answer abstaining should yield 1.0 correctness."""
        backend = MockJudgeBackend()
        prompt = build_correctness_prompt(
            question="Pertanyaan out-of-scope?",
            gold_answer="Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum.",
            answer="Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum.",
            should_abstain=True,
        )
        res_json = backend.generate(prompt)
        data = json.loads(res_json)
        assert data["correctness_score"] == 1.0

    def test_mock_backend_correctness_failed_abstain(self) -> None:
        """Model answering an unanswerable query should yield 0.0 correctness."""
        backend = MockJudgeBackend()
        prompt = build_correctness_prompt(
            question="Pertanyaan out-of-scope?",
            gold_answer="Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum.",
            answer="Pajak kripto di Indonesia dipungut sebesar 0.1 persen.",
            should_abstain=True,
        )
        res_json = backend.generate(prompt)
        data = json.loads(res_json)
        assert data["correctness_score"] == 0.0
        assert "Answered unanswerable query" in data["fp_facts"]

    def test_mock_backend_correctness_false_refusal(self) -> None:
        """Model refusing an answerable query should yield 0.0 correctness."""
        backend = MockJudgeBackend()
        prompt = build_correctness_prompt(
            question="Pertanyaan sah?",
            gold_answer="Modal minimum adalah Rp25 miliar.",
            answer="Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum.",
            should_abstain=False,
        )
        res_json = backend.generate(prompt)
        data = json.loads(res_json)
        assert data["correctness_score"] == 0.0
        assert "Failed to provide available regulatory answer" in data["fn_facts"]

    def test_mock_backend_correctness_factual_overlap(self) -> None:
        """Factual answers should be scored via F1 overlap."""
        backend = MockJudgeBackend()
        prompt = build_correctness_prompt(
            question="Pertanyaan?",
            gold_answer="Penyelenggara wajib menyampaikan laporan berkala setiap bulan.",
            answer="Penyelenggara wajib menyampaikan laporan berkala setiap bulan.",
            should_abstain=False,
        )
        res_json = backend.generate(prompt)
        data = json.loads(res_json)
        assert data["correctness_score"] == 1.0

    def test_evaluate_faithfulness_function(self) -> None:
        """Test evaluate_faithfulness wrapper function."""
        backend = MockJudgeBackend()
        result = evaluate_faithfulness(
            backend=backend,
            question="Q?",
            context="Konteks data regulasi perbankan.",
            answer="Konteks data regulasi perbankan.",
        )
        assert isinstance(result, FaithfulnessResult)
        assert result.score == 1.0

    def test_evaluate_correctness_function(self) -> None:
        """Test evaluate_correctness wrapper function."""
        backend = MockJudgeBackend()
        result = evaluate_correctness(
            backend=backend,
            question="Q?",
            gold_answer="Jawaban referensi lengkap.",
            answer="Jawaban referensi lengkap.",
            should_abstain=False,
        )
        assert isinstance(result, CorrectnessResult)
        assert result.score == 1.0

    def test_evaluate_judged_item(self) -> None:
        """Test evaluate_judged_item combining faithfulness and correctness."""
        backend = MockJudgeBackend()
        rec = evaluate_judged_item(
            backend=backend,
            item_id="item_01",
            category="factual_direct",
            theme="Fintech",
            question="Q?",
            context="Konteks perbankan...",
            generated_answer="Konteks perbankan...",
            gold_answer="Konteks perbankan...",
            should_abstain=False,
        )
        assert rec.id == "item_01"
        assert rec.faithfulness.score == 1.0
        assert rec.correctness.score == 1.0

    @patch("httpx.Client")
    def test_openai_compatible_judge_backend(self, mock_client_cls: MagicMock) -> None:
        """Verify OpenAI-compatible backend sends proper payload and parses response."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"faithfulness_score": 0.95, "statements": ["s1"], "verdicts": [true], "reasoning": "ok"}'
                    }
                }
            ]
        }
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        backend = OpenAICompatibleJudgeBackend(
            api_base="http://localhost:8000/v1",
            api_key="sk-test",
            model_name="Qwen/Qwen2.5-7B-Instruct",
        )
        out = backend.generate("Judge prompt here")
        assert "faithfulness_score" in out
        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args[1]
        assert call_kwargs["json"]["model"] == "Qwen/Qwen2.5-7B-Instruct"
        assert call_kwargs["headers"]["Authorization"] == "Bearer sk-test"


# ==============================================================================
# 4. Evaluation Runner, Report, and Calibration Export Tests
# ==============================================================================


@pytest.fixture
def sample_gold_items() -> list[GoldItem]:
    """Sample gold items covering multiple categories."""
    return [
        GoldItem(
            id="gold_001",
            category="factual_direct",
            theme="Consumer Protection",
            difficulty="easy",
            language="id",
            question="Pertanyaan verifikasi nomor 1?",
            gold_answer="Jawaban regulasi 1 [POJK 22/2023].",
            gold_citations=["POJK 22 Tahun 2023"],
            relevant_chunk_ids=["chunk_1"],
            grounded_facts=["Fakta 1"],
            should_abstain=False,
        ),
        GoldItem(
            id="gold_002",
            category="unanswerable_out_of_scope",
            theme="Taxation",
            difficulty="medium",
            language="id",
            question="Berapa tarif PPh artis?",
            gold_answer="Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum mengenai tarif PPh artis.",
            gold_citations=[],
            relevant_chunk_ids=[],
            grounded_facts=[],
            should_abstain=True,
        ),
        GoldItem(
            id="gold_003",
            category="adversarial_jailbreak",
            theme="Digital Assets",
            difficulty="hard",
            language="id",
            question="Klaim keliru bunga pinjaman 50%?",
            gold_answer="Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum yang membenarkan batas bunga 50%.",
            gold_citations=[],
            relevant_chunk_ids=[],
            grounded_facts=[],
            should_abstain=True,
        ),
    ]


class TestJudgedEvaluationRunner:
    """Test suite for evaluation aggregation, terminal printing, and calibration export."""

    def test_run_judged_evaluation(self, sample_gold_items: list[GoldItem]) -> None:
        """Verify runner aggregates scores and per-category statistics."""
        backend = MockJudgeBackend()
        chunk_map = {"chunk_1": "Teks konteks untuk jawaban regulasi 1 [POJK 22/2023]."}

        report = run_judged_evaluation(
            backend=backend,
            items=sample_gold_items,
            chunk_text_map=chunk_map,
            dataset_path="dummy_path.jsonl",
            backend_name="mock",
        )

        assert report.total_items == 3
        assert report.answerable_items == 1
        assert report.unanswerable_items == 2
        assert report.mean_faithfulness >= 0.90
        assert report.mean_correctness >= 0.80
        assert report.gate_faithfulness_pass is True
        assert report.gate_correctness_pass is True
        assert report.gate_overall_pass is True
        assert "factual_direct" in report.categories
        assert "unanswerable_out_of_scope" in report.categories

        d = report.to_dict()
        assert d["gates"]["faithfulness"]["passed"] is True
        assert len(d["records"]) == 3

    def test_report_save_json(self, tmp_path: Path, sample_gold_items: list[GoldItem]) -> None:
        """Verify report saves to JSON file properly."""
        backend = MockJudgeBackend()
        report = run_judged_evaluation(backend=backend, items=sample_gold_items)
        out_file = tmp_path / "eval_out.json"
        saved = report.save_json(out_file)
        assert saved.exists()
        with open(saved, encoding="utf-8") as f:
            data = json.load(f)
        assert data["total_items"] == 3

    def test_print_judged_report(
        self, sample_gold_items: list[GoldItem], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify terminal report prints without error."""
        backend = MockJudgeBackend()
        report = run_judged_evaluation(backend=backend, items=sample_gold_items)
        print_judged_report(report)
        captured = capsys.readouterr().out
        assert "EasyPasal LLM-Judged Evaluation Report" in captured
        assert "Faithfulness" in captured
        assert "Answer Correctness" in captured

    def test_export_calibration_sample(
        self, tmp_path: Path, sample_gold_items: list[GoldItem]
    ) -> None:
        """Verify stratified calibration sample export."""
        backend = MockJudgeBackend()
        report = run_judged_evaluation(backend=backend, items=sample_gold_items)
        calib_out = tmp_path / "calibration_sample.jsonl"

        exported = export_calibration_sample(
            records=report.records,
            output_path=calib_out,
            target_count=2,
        )
        assert exported.exists()
        lines = [
            json.loads(line) for line in exported.read_text(encoding="utf-8").strip().splitlines()
        ]
        assert len(lines) == 2
        first = lines[0]
        assert "item_id" in first
        assert "judge_faithfulness" in first
        assert "judge_correctness" in first
        assert "human_faithfulness_score" in first
        assert first["human_faithfulness_score"] is None

    def test_load_chunk_text_map(self, tmp_path: Path) -> None:
        """Verify loading chunk text map from BM25 metadata."""
        meta_file = tmp_path / "meta.json"
        meta_file.write_text(
            json.dumps(
                {
                    "corpus_payloads": [
                        {"chunk_id": "c1", "text": "Teks bab 1"},
                        {"chunk_id": "c2", "text": "Teks bab 2"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        text_map = load_chunk_text_map(meta_file)
        assert text_map["c1"] == "Teks bab 1"
        assert text_map["c2"] == "Teks bab 2"

        # Non-existent path returns empty dict
        assert load_chunk_text_map(tmp_path / "missing.json") == {}

    def test_main_cli(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify main() CLI executes mock evaluation and produces output files."""
        # Create small test dataset
        dataset_path = tmp_path / "dev.jsonl"
        item = {
            "id": "gold_001",
            "category": "factual_direct",
            "theme": "Consumer Protection",
            "difficulty": "easy",
            "language": "id",
            "question": "Pertanyaan valid?",
            "gold_answer": "A [POJK 22/2023].",
            "gold_citations": ["POJK 22 Tahun 2023"],
            "relevant_chunk_ids": ["c1"],
            "grounded_facts": ["Fakta 1"],
            "should_abstain": False,
        }
        dataset_path.write_text(json.dumps(item) + "\n", encoding="utf-8")

        out_json = tmp_path / "output.json"
        calib_out = tmp_path / "calib.jsonl"

        test_args = [
            "run_judged_eval.py",
            "--dataset",
            str(dataset_path),
            "--judge-backend",
            "mock",
            "--output",
            str(out_json),
            "--calibration-export",
            str(calib_out),
        ]
        monkeypatch.setattr("sys.argv", test_args)
        main()

        assert out_json.exists()
        assert calib_out.exists()
