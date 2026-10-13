"""LLM-as-a-judge metrics: Faithfulness and Answer Correctness using RAGAS/DeepEval-style prompts."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("eval.metrics.judge")

STANDARD_ABSTENTION_PATTERNS = [
    r"tidak ditemukan dasar hukum",
    r"no legal basis was found",
    r"berdasarkan regulasi yang tersedia, tidak ditemukan",
    r"based on the available regulations, no legal basis",
]


def is_abstention_text(text: str) -> bool:
    """Check if generated text expresses an explicit standardized legal abstention."""
    lower = text.lower()
    return any(re.search(pat, lower) for pat in STANDARD_ABSTENTION_PATTERNS)


@dataclass
class FaithfulnessResult:
    """Result of LLM-judged faithfulness evaluation."""

    score: float
    statements: list[str]
    verdicts: list[bool]
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": round(self.score, 4),
            "statements": self.statements,
            "verdicts": self.verdicts,
            "reasoning": self.reasoning,
        }


@dataclass
class CorrectnessResult:
    """Result of LLM-judged answer correctness evaluation."""

    score: float
    tp_facts: list[str]
    fp_facts: list[str]
    fn_facts: list[str]
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": round(self.score, 4),
            "tp_facts": self.tp_facts,
            "fp_facts": self.fp_facts,
            "fn_facts": self.fn_facts,
            "reasoning": self.reasoning,
        }


@dataclass
class JudgedRecord:
    """Combined record of judged evaluation for a single inquiry."""

    id: str
    category: str
    theme: str
    question: str
    context: str
    generated_answer: str
    gold_answer: str
    should_abstain: bool
    faithfulness: FaithfulnessResult
    correctness: CorrectnessResult

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "category": self.category,
            "theme": self.theme,
            "question": self.question,
            "context": self.context,
            "generated_answer": self.generated_answer,
            "gold_answer": self.gold_answer,
            "should_abstain": self.should_abstain,
            "faithfulness": self.faithfulness.to_dict(),
            "correctness": self.correctness.to_dict(),
        }


def extract_json_payload(raw_text: str) -> dict[str, Any]:
    """Extract and parse JSON object from LLM response text."""
    clean = raw_text.strip()
    # Match markdown json block if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
    if match:
        clean = match.group(1).strip()
    else:
        # Fallback: find outer curly brackets
        first_brace = clean.find("{")
        last_brace = clean.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            clean = clean[first_brace : last_brace + 1]

    try:
        data = json.loads(clean)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError as err:
        logger.warning("Failed to decode JSON from judge output: %s (raw=%r)", err, raw_text[:200])

    return {}


def build_faithfulness_prompt(question: str, context: str, answer: str) -> str:
    """Construct RAGAS-style decomposed claim faithfulness evaluation prompt."""
    return f"""You are an expert Indonesian financial regulatory compliance evaluator.
Your task is to judge whether the GENERATED ANSWER is strictly FAITHFUL to the provided RETRIEVED CONTEXT.
A faithful answer contains ONLY claims that are directly supported by or logically entailed by the context.

Special Rule for Abstentions:
If the question is unanswerable or the answer states that no regulatory basis was found in the provided regulations, that response is 100% FAITHFUL because it makes zero ungrounded factual claims.

EVALUATION PROCESS:
1. Decompose the GENERATED ANSWER into individual atomic factual statements/claims.
2. For each statement, verify if it can be directly inferred from the RETRIEVED CONTEXT (verdict: true or false).
3. Compute faithfulness_score = (number of true statements) / (total statements).
   If there are 0 factual statements (e.g. an explicit abstention refusal), faithfulness_score is 1.0.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Respond ONLY with a valid JSON object in this exact schema:
{{
  "statements": ["claim 1", "claim 2"],
  "verdicts": [true, true],
  "faithfulness_score": 1.0,
  "reasoning": "Step-by-step evaluation explanation"
}}
"""


def build_correctness_prompt(
    question: str, gold_answer: str, answer: str, should_abstain: bool
) -> str:
    """Construct DeepEval-style factual correctness evaluation prompt."""
    abstain_instruction = (
        "This question is OUT OF SCOPE / ADVERSARIAL. The system MUST refuse/abstain. "
        "If the answer properly abstains, correctness_score is 1.0. If the answer fabricates or attempts to answer, correctness_score is 0.0."
        if should_abstain
        else "This question is ANSWERABLE. Compare the generated answer against the reference gold answer."
    )

    return f"""You are an expert Indonesian legal and regulatory compliance evaluator.
Your task is to judge the FACTUAL CORRECTNESS of the GENERATED ANSWER against the authoritative REFERENCE GOLD ANSWER.

SPECIAL CONTEXT:
{abstain_instruction}

EVALUATION PROCESS:
1. Identify atomic verifiable legal facts in the REFERENCE GOLD ANSWER.
2. Identify verifiable facts asserted in the GENERATED ANSWER.
3. Classify facts:
   - tp_facts (True Positives): Gold facts correctly stated in the generated answer.
   - fp_facts (False Positives): Claims in the generated answer that contradict or add unverified details not in the reference answer.
   - fn_facts (False Negatives): Required facts from the reference answer missing from the generated answer.
4. If should_abstain is True:
   - If the answer refuses correctly, tp_facts=["proper_refusal"], fp_facts=[], fn_facts=[], correctness_score=1.0.
   - If the answer attempts to answer, tp_facts=[], fp_facts=["failed_to_abstain"], fn_facts=["required_refusal"], correctness_score=0.0.
5. For answerable queries:
   - Compute F1 score: 2 * len(tp) / (2 * len(tp) + len(fp) + len(fn)) as correctness_score.

QUESTION:
{question}

REFERENCE GOLD ANSWER:
{gold_answer}

GENERATED ANSWER:
{answer}

Respond ONLY with a valid JSON object in this exact schema:
{{
  "tp_facts": ["fact 1"],
  "fp_facts": [],
  "fn_facts": [],
  "correctness_score": 1.0,
  "reasoning": "Step-by-step comparison explanation"
}}
"""


class BaseJudgeBackend(ABC):
    """Abstract interface for LLM judge generation backends."""

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate response text for a given judge prompt."""
        pass


class MockJudgeBackend(BaseJudgeBackend):
    """Deterministic heuristic mock judge for testing and offline CI without GPU or API keys."""

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        _ = temperature
        # Check prompt type: faithfulness or correctness
        if "faithfulness_score" in prompt:
            # Check if answer is an abstention
            match_ans = re.search(r"GENERATED ANSWER:\s*\n([\s\S]*?)\n\s*Respond ONLY", prompt)
            ans = match_ans.group(1).strip() if match_ans else ""

            if is_abstention_text(ans):
                return json.dumps(
                    {
                        "statements": ["No regulatory basis found in corpus"],
                        "verdicts": [True],
                        "faithfulness_score": 1.0,
                        "reasoning": "Standard refusal makes zero ungrounded claims beyond context.",
                    }
                )

            # For factual answers: verify keyword presence in context
            match_ctx = re.search(
                r"RETRIEVED CONTEXT:\s*\n([\s\S]*?)\n\s*GENERATED ANSWER:", prompt
            )
            ctx = match_ctx.group(1).lower() if match_ctx else ""
            ans_tokens = list(re.findall(r"\w{4,}", ans.lower()))
            if not ans_tokens:
                return json.dumps(
                    {
                        "statements": [ans],
                        "verdicts": [True],
                        "faithfulness_score": 1.0,
                        "reasoning": "No verifiable factual tokens.",
                    }
                )

            hits = sum(1 for w in ans_tokens if w in ctx)
            ratio = hits / len(ans_tokens)
            score = 1.0 if ratio >= 0.25 else min(1.0, max(0.0, round(ratio * 2.5, 2)))
            return json.dumps(
                {
                    "statements": [ans],
                    "verdicts": [score >= 0.70],
                    "faithfulness_score": score,
                    "reasoning": f"Mock lexical grounding check: {hits}/{len(ans_tokens)} tokens matched context.",
                }
            )

        # Correctness prompt
        match_gold = re.search(
            r"REFERENCE GOLD ANSWER:\s*\n([\s\S]*?)\n\s*GENERATED ANSWER:", prompt
        )
        match_ans = re.search(r"GENERATED ANSWER:\s*\n([\s\S]*?)\n\s*Respond ONLY", prompt)
        gold = match_gold.group(1).strip() if match_gold else ""
        ans = match_ans.group(1).strip() if match_ans else ""

        is_gold_abstain = is_abstention_text(gold)
        is_ans_abstain = is_abstention_text(ans)

        if is_gold_abstain:
            if is_ans_abstain:
                return json.dumps(
                    {
                        "tp_facts": ["Standard legal abstention appropriately issued"],
                        "fp_facts": [],
                        "fn_facts": [],
                        "correctness_score": 1.0,
                        "reasoning": "Correct refusal on unanswerable/adversarial inquiry.",
                    }
                )
            else:
                return json.dumps(
                    {
                        "tp_facts": [],
                        "fp_facts": ["Answered unanswerable query"],
                        "fn_facts": ["Expected standardized refusal"],
                        "correctness_score": 0.0,
                        "reasoning": "Failed to abstain on out-of-scope/adversarial query.",
                    }
                )

        if is_ans_abstain:
            return json.dumps(
                {
                    "tp_facts": [],
                    "fp_facts": [],
                    "fn_facts": ["Failed to provide available regulatory answer"],
                    "correctness_score": 0.0,
                    "reasoning": "False refusal on answerable query.",
                }
            )

        # Both factual
        gold_words = set(re.findall(r"\w{4,}", gold.lower()))
        ans_words = set(re.findall(r"\w{4,}", ans.lower()))
        tp = gold_words & ans_words
        fn = gold_words - ans_words
        fp = ans_words - gold_words

        f1 = (
            2.0 * len(tp) / (2.0 * len(tp) + len(fp) + len(fn))
            if (2.0 * len(tp) + len(fp) + len(fn)) > 0
            else 0.0
        )
        score = 1.0 if f1 >= 0.50 else round(f1, 2)

        return json.dumps(
            {
                "tp_facts": list(tp)[:3],
                "fp_facts": list(fp)[:2],
                "fn_facts": list(fn)[:2],
                "correctness_score": score,
                "reasoning": f"Mock factual overlap F1: {f1:.2f}.",
            }
        )


class OpenAICompatibleJudgeBackend(BaseJudgeBackend):
    """Judge backend targeting OpenAI-compatible endpoints (vLLM, Ollama, OpenRouter, Together)."""

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "dummy",
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        timeout_seconds: float = 60.0,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        import httpx

        url = f"{self.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1024,
        }

        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"])


class TransformersJudgeBackend(BaseJudgeBackend):
    """Judge backend for local PyTorch / Hugging Face pipeline (Colab/Kaggle free GPU)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        load_in_4bit: bool = True,
        device_map: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self._pipeline: Any = None

    def _init_pipeline(self) -> Any:
        if self._pipeline is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            logger.info("Initializing open-weight judge model %r...", self.model_name)
            model_kwargs: dict[str, Any] = {
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            }
            if self.load_in_4bit and torch.cuda.is_available():
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                **model_kwargs,
            )
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return self._pipeline

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        pipe = self._init_pipeline()
        messages = [{"role": "user", "content": prompt}]
        formatted = pipe.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": 1024,
            "return_full_text": False,
        }
        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        outputs = pipe(formatted, **gen_kwargs)
        return str(outputs[0]["generated_text"])


def evaluate_faithfulness(
    backend: BaseJudgeBackend,
    question: str,
    context: str,
    answer: str,
) -> FaithfulnessResult:
    """Evaluate factual faithfulness of generated answer against retrieved passages."""
    prompt = build_faithfulness_prompt(question=question, context=context, answer=answer)
    raw_response = backend.generate(prompt=prompt, temperature=0.0)
    data = extract_json_payload(raw_response)

    score = float(data.get("faithfulness_score", 0.0))
    statements = [str(s) for s in data.get("statements", [])]
    verdicts = [bool(v) for v in data.get("verdicts", [])]
    reasoning = str(data.get("reasoning", "No explanation provided."))

    # Boundary clamp
    score = max(0.0, min(1.0, score))

    return FaithfulnessResult(
        score=score,
        statements=statements,
        verdicts=verdicts,
        reasoning=reasoning,
    )


def evaluate_correctness(
    backend: BaseJudgeBackend,
    question: str,
    gold_answer: str,
    answer: str,
    should_abstain: bool,
) -> CorrectnessResult:
    """Evaluate factual correctness of generated answer against reference answer."""
    prompt = build_correctness_prompt(
        question=question,
        gold_answer=gold_answer,
        answer=answer,
        should_abstain=should_abstain,
    )
    raw_response = backend.generate(prompt=prompt, temperature=0.0)
    data = extract_json_payload(raw_response)

    score = float(data.get("correctness_score", 0.0))
    tp = [str(x) for x in data.get("tp_facts", [])]
    fp = [str(x) for x in data.get("fp_facts", [])]
    fn = [str(x) for x in data.get("fn_facts", [])]
    reasoning = str(data.get("reasoning", "No explanation provided."))

    score = max(0.0, min(1.0, score))

    return CorrectnessResult(
        score=score,
        tp_facts=tp,
        fp_facts=fp,
        fn_facts=fn,
        reasoning=reasoning,
    )


def evaluate_judged_item(
    backend: BaseJudgeBackend,
    item_id: str,
    category: str,
    theme: str,
    question: str,
    context: str,
    generated_answer: str,
    gold_answer: str,
    should_abstain: bool,
) -> JudgedRecord:
    """Evaluate both faithfulness and correctness for a single item."""
    faith_res = evaluate_faithfulness(
        backend=backend,
        question=question,
        context=context,
        answer=generated_answer,
    )
    corr_res = evaluate_correctness(
        backend=backend,
        question=question,
        gold_answer=gold_answer,
        answer=generated_answer,
        should_abstain=should_abstain,
    )
    return JudgedRecord(
        id=item_id,
        category=category,
        theme=theme,
        question=question,
        context=context,
        generated_answer=generated_answer,
        gold_answer=gold_answer,
        should_abstain=should_abstain,
        faithfulness=faith_res,
        correctness=corr_res,
    )
