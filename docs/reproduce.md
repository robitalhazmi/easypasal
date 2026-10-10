# EasyPasal — Reproduction Guide

This guide walks through reproducing every baseline, evaluation score, and benchmark reported in the EasyPasal project.

## Step 1: Environment Setup
```bash
git clone https://github.com/robit/easypasal.git
cd easypasal
make setup
make check
```

## Step 2: Ingest and Index Corpus
```bash
make ingest
make index
```

## Step 3: Run Baseline Evaluation
```bash
make eval
```

## Step 4: Run Experiments & Benchmarks
- See `experiments/` for scripts to reproduce E1–E5.
- See `benchmarks/vllm/` for vLLM evaluation notebooks on free GPU.
