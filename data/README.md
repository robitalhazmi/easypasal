# EasyPasal Data Directory

This directory contains metadata manifests and evaluation datasets for the EasyPasal project.

## Structure

- `corpus_manifest.csv`: Catalog of all official public regulatory documents included in the corpus (OJK, Bank Indonesia, UU PDP), tracking official numbers, publication years, issuers, source URLs, retrieval dates, amendment/superseded statuses, and SHA-256 hashes.
- `gold/`: Hand-labeled gold evaluation items partitioned into `train`, `dev`, and `test` splits.
- `raw/` *(git-ignored)*: Local storage directory for downloaded raw PDF files fetched via `make ingest`.

## Corpus Composition (21 Regulations across 5 Themes)

| Theme | Key In-Force Regulations | Superseded / Trap Cases |
|---|---|---|
| **1. Consumer Protection** | POJK 22/2023, SEOJK 2/2024, POJK 61/2020 | POJK 6/2022 (*dicabut*) |
| **2. IT & Cyber Risk in Banking** | POJK 11/2022, SEOJK 29/2022, POJK 21/2023 | POJK 38/2016 (*dicabut*) |
| **3. AML / CFT & PPPSPM** | POJK 8/2023, UU 8/2010 (TPPU), UU 9/2013 (TPPT) | POJK 12/2017 (*dicabut*) |
| **4. Personal Data & Banking Secrecy** | UU 27/2022 (UU PDP), UU 4/2023 (P2SK), UU 10/1998, PP 71/2019 | — |
| **5. Capital Markets & Bursa Efek** | POJK 31/2015 (Keterbukaan Informasi), POJK 15/2020 (RUPS), POJK 3/2021, UU 8/1995 | POJK 32/2014 (*dicabut*) |

## Data Governance & Licensing

All regulatory documents referenced in `corpus_manifest.csv` are public regulations issued by the Republic of Indonesia authorities (OJK, Bank Indonesia, JDIH). Raw PDF binaries are not tracked in git; the ingestion pipeline (`make ingest`) downloads and validates files from official source URLs.
