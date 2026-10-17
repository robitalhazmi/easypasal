---
title: EasyPasal - Indonesian Financial Regulatory Compliance AI
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: AI assistant for OJK & Bank Indonesia compliance
---

# ⚖️ EasyPasal — Live Demo on Hugging Face Spaces

**EasyPasal** is an open-source, citation-grounded regulatory compliance assistant designed for the Indonesian financial services sector (Otoritas Jasa Keuangan & Bank Indonesia).

## Features
- **Strict Statutory Grounding:** Answers cite exact regulation numbers, articles (*Pasal*), and clauses (*Ayat*).
- **Refusal on Out-of-Scope / Ungrounded Queries:** Explicitly abstains when statutory evidence is absent or query is outside the corpus.
- **Bilingual Interface:** Supports both Bahasa Indonesia and English.
- **Prebuilt Embedded Index:** Qdrant vector store and BM25 index packaged directly within the container for sub-second retrieval.

## Disclaimer
> ⚠️ **Pemberitahuan / Legal Disclaimer:** EasyPasal adalah purwarupa riset kepatuhan berbasis AI dan **tidak memberikan nasihat hukum mengikat (*no legal advice*)**. Selalu periksa dan rujuk dokumen naskah peraturan resmi yang diterbitkan oleh Otoritas Jasa Keuangan (OJK) atau Bank Indonesia (BI).
