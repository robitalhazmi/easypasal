"""QLoRA fine-tuning training script using PEFT and TRL (Experiment E7)."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from experiments.e7_qlora.config import QLoRAConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_mock_training(config: QLoRAConfig) -> dict[str, Any]:
    """Execute simulated mock QLoRA training for local CI and CPU verification.

    Validates dataset integrity, simulates step-by-step SFT loss progression,
    and exports adapter configuration artifacts.
    """
    logger.info("Starting mock QLoRA training run (CPU / CI fallback mode)...")
    train_path = Path(config.train_data_path)
    dev_path = Path(config.dev_data_path)

    if not train_path.exists() or not dev_path.exists():
        from experiments.e7_qlora.dataset import build_router_dataset

        logger.info("Router datasets missing; generating them automatically...")
        build_router_dataset()

    with open(train_path, encoding="utf-8") as f:
        train_samples = [json.loads(line) for line in f if line.strip()]
    with open(dev_path, encoding="utf-8") as f:
        dev_samples = [json.loads(line) for line in f if line.strip()]

    logger.info(
        "Loaded %d training samples and %d validation samples.",
        len(train_samples),
        len(dev_samples),
    )

    # Simulate training loss curve
    total_steps = (len(train_samples) * config.num_train_epochs) // (
        config.per_device_train_batch_size * config.gradient_accumulation_steps
    )
    total_steps = max(total_steps, 15)

    loss_history: list[dict[str, float]] = []
    initial_loss = 2.450
    final_loss = 0.285
    start_time = time.time()

    for step in range(1, total_steps + 1):
        progress = step / total_steps
        current_loss = initial_loss - (initial_loss - final_loss) * (progress**0.7)
        if step % config.logging_steps == 0 or step == total_steps:
            val_loss = current_loss * 1.08
            loss_history.append(
                {"step": step, "train_loss": round(current_loss, 4), "val_loss": round(val_loss, 4)}
            )
            logger.info(
                "Step %d/%d - Train Loss: %.4f - Val Loss: %.4f",
                step,
                total_steps,
                current_loss,
                val_loss,
            )

    train_duration = time.time() - start_time

    # Export mock adapter files
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter_config = {
        "base_model_name_or_path": config.base_model_name,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "peft_type": "LORA",
        "r": config.lora_r,
        "target_modules": config.target_modules,
        "task_type": "CAUSAL_LM",
    }

    with open(out_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2)

    training_meta = {
        "status": "completed",
        "mode": "mock_simulation",
        "base_model": config.base_model_name,
        "train_samples": len(train_samples),
        "dev_samples": len(dev_samples),
        "total_steps": total_steps,
        "train_duration_sec": round(train_duration, 2),
        "final_train_loss": round(final_loss, 4),
        "final_val_loss": round(final_loss * 1.08, 4),
        "loss_history": loss_history,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
    }

    with open(out_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(training_meta, f, indent=2)

    logger.info("Mock QLoRA training completed successfully. Artifacts saved to %s", out_dir)
    return training_meta


def run_gpu_training(config: QLoRAConfig) -> dict[str, Any]:
    """Execute live GPU QLoRA fine-tuning via Hugging Face PEFT and TRL."""
    try:
        import torch
        from datasets import Dataset  # type: ignore[import-not-found]
        from peft import (  # type: ignore[import-not-found]
            LoraConfig,
            PeftModel,
            prepare_model_for_kbit_training,
        )
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer  # type: ignore[import-not-found]
    except ImportError as err:
        logger.warning(
            "Live GPU dependencies (torch, peft, trl, transformers) not found: %s. "
            "Falling back to mock training simulation.",
            err,
        )
        return run_mock_training(config)

    logger.info("Initializing GPU QLoRA training on %s...", torch.cuda.get_device_name(0))

    # 0. Hardware Capability Check (Tesla T4 is CC 7.5; BFloat16 GEMM requires CC >= 8.0)
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        if (major, minor) < (8, 0) and config.bnb_4bit_compute_dtype == "bfloat16":
            logger.warning(
                "CUDA device %s has compute capability %d.%d (< 8.0) which lacks native BFloat16 GEMM. "
                "Switching bnb_4bit_compute_dtype and training flags to float16.",
                torch.cuda.get_device_name(0),
                major,
                minor,
            )
            config.bnb_4bit_compute_dtype = "float16"

    # 1. 4-bit Quantization Config
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype, torch.float16)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # 2. Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Config (passed directly to SFTTrainer to avoid double-wrapping)
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.target_modules,
    )

    # 4. Load datasets
    with open(config.train_data_path, encoding="utf-8") as f:
        train_raw = [json.loads(line) for line in f if line.strip()]
    with open(config.dev_data_path, encoding="utf-8") as f:
        dev_raw = [json.loads(line) for line in f if line.strip()]

    train_ds = Dataset.from_list([{"messages": it["messages"]} for it in train_raw])
    dev_ds = Dataset.from_list([{"messages": it["messages"]} for it in dev_raw])

    # 5. Training Arguments / SFTConfig (compatible with both transformers 4.x and 5.x+, TRL 0.x)
    import inspect

    args_class = TrainingArguments
    try:
        from trl import SFTConfig

        args_class = SFTConfig
    except (ImportError, AttributeError):
        pass

    sig_params = inspect.signature(args_class.__init__).parameters

    training_kwargs: dict[str, Any] = {
        "output_dir": config.output_dir,
        "learning_rate": config.learning_rate,
        "lr_scheduler_type": config.lr_scheduler_type,
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "logging_steps": config.logging_steps,
        "eval_steps": config.eval_steps,
        "save_strategy": config.save_strategy,
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "weight_decay": config.weight_decay,
        "seed": config.seed,
        "fp16": (config.bnb_4bit_compute_dtype == "float16"),
        "bf16": (config.bnb_4bit_compute_dtype == "bfloat16"),
        "report_to": "none",
    }

    # Handle warmup_ratio (transformers <5) vs warmup_steps (transformers >=5)
    if "warmup_ratio" in sig_params:
        training_kwargs["warmup_ratio"] = config.warmup_ratio
    elif "warmup_steps" in sig_params:
        training_kwargs["warmup_steps"] = config.warmup_ratio

    # Handle eval_strategy (transformers >=4.41) vs evaluation_strategy (legacy)
    if "eval_strategy" in sig_params:
        training_kwargs["eval_strategy"] = config.eval_strategy
    elif "evaluation_strategy" in sig_params:
        training_kwargs["evaluation_strategy"] = config.eval_strategy

    # Handle max sequence length in SFTConfig
    if "max_seq_length" in sig_params:
        training_kwargs["max_seq_length"] = config.max_seq_length
    elif "max_length" in sig_params:
        training_kwargs["max_length"] = config.max_seq_length

    training_args = args_class(**training_kwargs)

    # 6. SFTTrainer initialization with backward/forward compatibility
    trainer_sig = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": dev_ds,
        "args": training_args,
    }

    # Pass peft_config only if model has not already been wrapped as a PeftModel
    if not isinstance(model, PeftModel):
        trainer_kwargs["peft_config"] = peft_config

    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer

    if "max_seq_length" in trainer_sig and "max_seq_length" not in training_kwargs:
        trainer_kwargs["max_seq_length"] = config.max_seq_length

    trainer = SFTTrainer(**trainer_kwargs)

    logger.info("Starting training loop...")
    train_result = trainer.train()

    # Save final model
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    metrics = train_result.metrics
    logger.info("Training complete. Metrics: %s", metrics)
    return dict(metrics) if isinstance(metrics, dict) else {"status": "completed"}


def main() -> None:
    """CLI entrypoint for QLoRA fine-tuning."""
    parser = argparse.ArgumentParser(description="Experiment E7: QLoRA Fine-Tuning.")
    parser.add_argument(
        "--config", default="experiments/e7_qlora/training_config.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--mock", action="store_true", help="Force mock training execution (for CPU / CI)"
    )
    parser.add_argument("--output-dir", default=None, help="Output directory override")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count override")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    config = QLoRAConfig.from_yaml(cfg_path) if cfg_path.exists() else QLoRAConfig()

    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.num_train_epochs = args.epochs

    meta = run_mock_training(config) if args.mock else run_gpu_training(config)

    print("\n=== Training Execution Summary ===")
    print(f"Status: {meta.get('status', 'done')}")
    print(f"Output Directory: {config.output_dir}")


if __name__ == "__main__":
    main()
