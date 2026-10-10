# EasyPasal Data Directory

This directory contains metadata manifests and evaluation datasets for the EasyPasal project.

## Structure

- `corpus_manifest.csv`: Catalog of all official public regulatory documents included in the corpus (OJK, Bank Indonesia, UU PDP), tracking official numbers, publication years, issuers, source URLs, retrieval dates, amendment/superseded statuses, and SHA-256 hashes.
- `gold/`: Hand-labeled gold evaluation items partitioned into `train`, `dev`, and `test` splits.
- `raw/` *(git-ignored)*: Local storage directory for downloaded raw PDF files fetched via `make ingest`.

## Data Governance & Licensing

All regulatory documents referenced in `corpus_manifest.csv` are public regulations issued by the Republic of Indonesia authorities (OJK, Bank Indonesia, JDIH). Raw PDF binaries are not tracked in git; the ingestion pipeline downloads and validates files from official sources.
