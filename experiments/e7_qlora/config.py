"""Configuration parameters for Experiment E7 (QLoRA Fine-Tuning)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class QLoRAConfig:
    """Hyperparameter and hardware configuration for QLoRA fine-tuning."""

    # Model and paths
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str = "experiments/e7_qlora/adapter"
    train_data_path: str = "data/e7_router/train.jsonl"
    dev_data_path: str = "data/e7_router/dev.jsonl"
    test_data_path: str = "data/e7_router/test.jsonl"

    # Quantization (4-bit NF4)
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"

    # LoRA Adapter parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # Training Hyperparameters
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 512
    weight_decay: float = 0.01
    logging_steps: int = 5
    eval_strategy: str = "steps"
    eval_steps: int = 25
    save_strategy: str = "steps"
    save_steps: int = 25
    save_total_limit: int = 2
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> QLoRAConfig:
        """Load configuration from a YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
