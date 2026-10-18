"""Privacy preservation and PII redaction utilities (Plan §9 / Task T7.1).

Detects and redacts sensitive personal data from user queries and feedback:
- Indonesian National Identity Numbers (NIK - 16 digits)
- Indonesian Tax Identification Numbers (NPWP - 15 or 16 digits)
- Indonesian and international phone numbers
- Email addresses
- Credit / Debit Card numbers (13 to 19 digits)
- Bank account numbers

Preserves statutory citation anchors (e.g. 'UU Nomor 4 Tahun 2023',
'POJK 22 Tahun 2023, Pasal 71') without corruption.
"""

from __future__ import annotations

import re

# Indonesian NIK: exactly 16 consecutive digits
_NIK_PATTERN = re.compile(r"\b[1-9]\d{15}\b")

# Indonesian Tax ID (NPWP): 15-16 digits formatted (XX.XXX.XXX.X-XXX.XXX) or raw digits
_NPWP_FORMATTED_PATTERN = re.compile(r"\b\d{2}\.\d{3}\.\d{3}\.\d{1}-\d{3}\.\d{3}\b")
_NPWP_RAW_PATTERN = re.compile(r"\b(?:npwp\s*[:=]?\s*)(\d{15,16})\b", re.IGNORECASE)

# Email address
_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# Phone number: Indonesian mobile / landline (+62 or 08 followed by 8-12 digits)
_PHONE_PATTERN = re.compile(r"(?:\+62|62|08)[0-9\s\-]{8,14}\b")

# Credit / Debit Card Numbers:
# 1. Spaced or hyphenated 16 digits (e.g. 4111-2222-3333-4444 or 4111 2222 3333 4444)
_CARD_FORMATTED_PATTERN = re.compile(r"\b(?:\d{4}[ -]){3}\d{4}\b")
# 2. Raw card numbers explicitly preceded by card keywords
_CARD_KEYWORD_PATTERN = re.compile(
    r"(?i)\b(?:kartu\s*(?:kredit|debit)?|card|cc|nomor\s*kartu)\s*[:=]?\s*(\d{13,19})\b"
)

# Bank Account Numbers: 10 to 18 digits preceded by banking keywords
_ACCOUNT_PATTERN = re.compile(
    r"(?i)\b(?:rekening|no\.?\s*rek(?:ening)?|rek(?:ening)?|acc(?:ount)?(?:\s*no)?)\s*[:=]?\s*(\d{8,18})\b"
)


def redact_pii(text: str | None) -> str:
    """Redact personal identifiable information from text, preserving statutory citations.

    Args:
        text: Input string potentially containing sensitive personal data.

    Returns:
        Sanitized string with PII replaced by standardized redaction tokens.
    """
    if not text:
        return ""

    sanitized = text

    # 1. Redact Email addresses
    sanitized = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", sanitized)

    # 2. Redact Card numbers (explicit keyword or spaced/hyphenated)
    sanitized = _CARD_KEYWORD_PATTERN.sub(r"kartu: [REDACTED_CARD]", sanitized)
    sanitized = _CARD_FORMATTED_PATTERN.sub("[REDACTED_CARD]", sanitized)

    # 3. Redact Formatted NPWP
    sanitized = _NPWP_FORMATTED_PATTERN.sub("[REDACTED_NPWP]", sanitized)

    # 4. Redact Keyword-preceded NPWP
    sanitized = _NPWP_RAW_PATTERN.sub("NPWP: [REDACTED_NPWP]", sanitized)

    # 5. Redact Bank Account Numbers
    sanitized = _ACCOUNT_PATTERN.sub(r"rekening: [REDACTED_ACCOUNT]", sanitized)

    # 6. Redact NIK (16 consecutive digits)
    sanitized = _NIK_PATTERN.sub("[REDACTED_NIK]", sanitized)

    # 7. Redact Phone numbers
    sanitized = _PHONE_PATTERN.sub("[REDACTED_PHONE]", sanitized)

    return sanitized


def has_pii(text: str | None) -> bool:
    """Check whether text contains any detected personal identifiable information.

    Args:
        text: Input text to inspect.

    Returns:
        True if any PII pattern matches.
    """
    if not text:
        return False

    return bool(
        _EMAIL_PATTERN.search(text)
        or _CARD_KEYWORD_PATTERN.search(text)
        or _CARD_FORMATTED_PATTERN.search(text)
        or _NPWP_FORMATTED_PATTERN.search(text)
        or _NPWP_RAW_PATTERN.search(text)
        or _ACCOUNT_PATTERN.search(text)
        or _NIK_PATTERN.search(text)
        or _PHONE_PATTERN.search(text)
    )
