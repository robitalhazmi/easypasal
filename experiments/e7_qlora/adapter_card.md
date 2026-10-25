# EasyPasal Model Adapter Card: Question Router & Guardrail Classifier (Experiment E7)

## Model Details

- **Base Model:** [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- **Adapter Type:** Parameter-Efficient Fine-Tuning (PEFT) — QLoRA (4-bit NF4)
- **Task Type:** Multi-Task Classification / Structured JSON Routing
- **Primary Function:** Front-line pipeline triage for Indonesian Financial Regulations (OJK & Bank Indonesia)
- **Target Fields:**
  1. `theme`: Regulatory domain (`Consumer Protection`, `IT & Cyber Risk`, `AML / CFT`, `Data Privacy & Secrecy`, `Capital Markets`, `General Financial Regulation`, or `Out of Scope`)
  2. `is_answerable`: Boolean flag (`true` if answerable by current corpus, `false` if repealed/out-of-scope/lacks legal basis)
  3. `is_injection`: Boolean safety flag (`true` if adversarial prompt injection, jailbreak, roleplay bypass, or system prompt exfiltration attempt)
- **License:** Base model Apache-2.0; Adapter code and artifacts Apache-2.0
- **Frameworks:** Hugging Face `peft`, `trl` (`SFTTrainer`), `transformers`, `bitsandbytes`

---

## Intended Use & Pipeline Integration (Plan §8.1)

In the EasyPasal request flow (`/ask`), the adapter functions as a high-speed pre-retrieval triage filter:
1. **Fast-Path Refusal:** Queries flagged with `is_injection: true` or `is_answerable: false` bypass expensive vector search and LLM generation entirely, providing instant safety refusals with sub-25ms latency.
2. **Thematic Partitioning:** Queries with identified themes can route directly to domain-specific index shards or apply thematic metadata filters.

---

## Training Configuration & Hyperparameters

```yaml
base_model_name: Qwen/Qwen2.5-0.5B-Instruct
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true
bnb_4bit_compute_dtype: float16

lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

learning_rate: 0.0002
lr_scheduler_type: cosine
warmup_ratio: 0.1
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
effective_batch_size: 16
max_seq_length: 512
weight_decay: 0.01
seed: 42
```

---

## Dataset Provenance & Partitioning

| Partition | Source | Sample Count | Purpose |
|---|---|:---:|---|
| **Train** | `data/gold/e5_train.jsonl` + curated balanced augmentations | 72 | Supervised fine-tuning with QLoRA |
| **Dev** | `data/gold/dev.jsonl` | 40 | Hyperparameter validation and checkpoint selection |
| **Test** | `data/gold/test.jsonl` | 66 | Held-out evaluation benchmark (strictly isolated) |

---

## Quantitative Evaluation Results

### 1. Held-Out Test Set Benchmark (`data/e7_router/test.jsonl`, $N = 66$)
*Artifact: [`results/e7_qlora_evaluation.json`](../../results/e7_qlora_evaluation.json)*

| Metric | Baseline Zero-Shot (`Qwen2.5-0.5B`) | Fine-Tuned QLoRA Adapter | Delta | Status |
|:---|:---:|:---:|:---:|:---:|
| **Theme Macro-F1** | 0.8371 | **1.0000** | **+0.1629** | ✅ Target Met ($\ge 0.85$) |
| **Theme Accuracy** | 0.9697 | **1.0000** | **+0.0303** | ✅ Target Met |
| **Answerability F1** | 0.9216 | **1.0000** | **+0.0784** | ✅ Target Met ($\ge 0.80$) |
| **Answerability Precision** | 0.8545 | **1.0000** | **+0.1455** | ✅ Target Met ($\ge 0.85$) |
| **Answerability Recall** | 1.0000 | **1.0000** | +0.0000 | ✅ Parity |
| **Injection Precision** | 1.0000 | **1.0000** | +0.0000 | ✅ Target Met ($100\%$) |
| **Injection Recall** | 0.6667 | **1.0000** | **+0.3333** | ✅ +33.3% on Subtle Attacks |
| **Overall Exact Match** | 0.8333 | **1.0000** | **+0.1667** | ✅ +16.7% Overall |
| **Mean Latency (ms)** | 56.62 ms | **21.56 ms** | **2.6x speedup** | ✅ Sub-25ms Budget |

### 2. Development Set Benchmark (`data/e7_router/dev.jsonl`, $N = 40$)

| Metric | Baseline Zero-Shot | Fine-Tuned QLoRA Adapter | Delta | Status |
|:---|:---:|:---:|:---:|:---:|
| **Theme Macro-F1** | 0.8333 | **1.0000** | **+0.1667** | ✅ Target Met |
| **Theme Accuracy** | 0.9500 | **1.0000** | **+0.0500** | ✅ Target Met |
| **Answerability F1** | 0.9333 | **1.0000** | **+0.0667** | ✅ Target Met |
| **Answerability Precision** | 0.8750 | **1.0000** | **+0.1250** | ✅ Target Met |
| **Injection Precision / Recall** | 1.0000 / 1.0000 | **1.0000 / 1.0000** | +0.0000 | ✅ 100% on dev |
| **Overall Exact Match** | 0.8500 | **1.0000** | **+0.1500** | ✅ Target Met |
| **Mean Latency (ms)** | 57.57 ms | **22.18 ms** | **2.6x speedup** | ✅ Sub-25ms Budget |

---

## Category Regression Audit (Held-Out Test Set, $N = 66$)

We audit performance across all 9 gold evaluation categories to verify zero regressions on standard answerable categories:

| Category | Count | Theme Accuracy | Answerability Accuracy | Injection Accuracy | Exact Match Rate | Regression Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `factual_direct` | 10 | 100.0% | 100.0% | 100.0% | 100.0% | ℹ️ Zero Regression |
| `cross_article_synthesis` | 10 | 100.0% | 100.0% | 100.0% | 100.0% | ℹ️ Zero Regression |
| `definitional` | 7 | 100.0% | 100.0% | 100.0% | 100.0% | ℹ️ Zero Regression |
| `obligation_and_prohibition` | 7 | 100.0% | 100.0% | 100.0% | 100.0% | ℹ️ Zero Regression |
| `sanctions_and_enforcement` | 7 | 100.0% | 100.0% | 100.0% | 100.0% | ℹ️ Zero Regression |
| `superseded_trap` | 6 | 100.0% | 100.0% | 100.0% | 100.0% | ℹ️ Fixed Zero-Shot Leak |
| `unanswerable_out_of_scope` | 10 | 100.0% | 100.0% | 100.0% | 100.0% | ℹ️ Fixed Zero-Shot Leak |
| `adversarial_jailbreak` | 3 | 100.0% | 100.0% | 100.0% | 100.0% | ℹ️ Fixed Stealth Injection |
| `english_slice` | 6 | 100.0% | 100.0% | 100.0% | 100.0% | ℹ️ Zero Regression |

---

## Limitations & Honest-Claims Compliance Notice

> [!IMPORTANT]
> **Honest-Claims Protocol Notice (Plan §7 & §13):**
> 1. This adapter is a **narrow, specialized classifier and routing model** trained on structured prompt routing. It is **not** a general-purpose conversational LLM fine-tuned on Indonesian banking jurisprudence.
> 2. The adapter weights should **only** be utilized for question routing and safety triage, never for generative answer drafting or statutory interpretation.
> 3. Results are benchmarked on our stratified evaluation splits; novel out-of-domain vocabulary or unseen adversarial attacks may still bypass classification.
