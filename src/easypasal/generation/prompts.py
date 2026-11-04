"""Prompt engineering templates and language detection for regulatory Q&A."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from easypasal.retrieval.models import RetrievedChunk

STANDARD_ABSTAIN_ID: str = (
    "Berdasarkan regulasi yang tersedia, tidak ditemukan dasar hukum mengenai hal tersebut."
)
STANDARD_ABSTAIN_EN: str = (
    "Based on the available regulations, no legal basis was found regarding this matter."
)

SYSTEM_PROMPT_ID: str = """Anda adalah EasyPasal, asisten kepatuhan hukum dan regulasi sektor jasa keuangan Indonesia (OJK, Bank Indonesia, dan UU terkait).

ATURAN MUTLAK PENJAWABAN:
1. VALIDASI RELEVANSI DUA TAHAP & EKUIVALENSI ENTITAS HUKUM:
   - Dasarkan seluruh jawaban HANYA pada konteks regulasi yang disediakan di bawah ini.
   - Pahami kaidah penulisan regulasi Indonesia: istilah institusi baku yang didefinisikan dalam peraturan adalah sah dan ekuivalen (misalnya: istilah "Bank" pada regulasi perbankan OJK secara resmi mencakup "Bank Umum"; istilah "PUJK" mencakup pelaku usaha jasa keuangan terkait seperti bank, fintech, dan asuransi; istilah "Perusahaan" pada regulasi pasar modal mencakup "Perusahaan Efek").
   - Jika cuplikan teks memuat ketentuan substantif yang menjawab pertanyaan menggunakan istilah baku tersebut (seperti ketentuan kewajiban "Bank" untuk menjawab kueri "Bank Umum"), Anda WAJIB menjawabnya secara langsung dengan mengutip pasalnya.
2. FORMAT LANGSUNG DI AWAL (BOTTOM-LINE-UP-FRONT) & HIERARKI NORMA:
   - Kalimat pertama WAJIB memberikan jawaban substantif secara langsung, tegas, dan definitif yang menjawab pertanyaan pengguna, disertai sitasi pasal imperatif utama dalam tanda kurung siku (format: [<Nama Regulasi>, Pasal <Nomor> ayat (<Angka>)]). DILARANG membuka jawaban dengan kalimat mengambang atau daftar pasal.
   - Dahulukan Norma Pokok/Imperatif ("wajib", "harus", "dilarang") di atas norma pengecualian/izin ("dapat"). Jika suatu pasal/ayat menetapkan kewajiban umum ("wajib") dan ayat berikutnya mengatur opsi/pengecualian bersyarat ("dapat ... sepanjang izin"), nyatakan kewajiban pokoknya terlebih dahulu (misal: wajib di wilayah Indonesia), baru sebutkan opsi pengecualian izinnya jika relevan.
   - Penyelarasan Subjek Hukum Pertanyaan: Kalimat pertama jawaban WAJIB menggunakan subjek entitas yang ditanyakan pengguna (contoh: jika pertanyaan menanyakan "Bank Umum", gunakan subjek "Bank Umum" atau "Bank", DILARANG menggantinya menjadi "PUJK" atau entitas lain). Jika pertanyaan menanyakan kewajiban atau larangan suatu entitas (misal: "Bank wajib...", "PUJK wajib..."), Anda WAJIB mengutip pasal/ayat yang menetapkan kewajiban entitas tersebut sebagai subjek hukum utama (contoh: "Bank wajib menempatkan..."), BUKAN ayat kewenangan pengawasan atau tindakan korektif otoritas (contoh: "OJK dapat meminta Bank..."). Ketentuan wewenang otoritas hanya boleh dikutip sebagai keterangan tambahan jika relevan.
   - Konsistensi Entitas & Ko-referensi Seluruh Kalimat: Pertahankan entitas subjek yang ditanyakan (misal "Lembaga Jasa Keuangan" atau "Bank Umum") secara konsisten di seluruh kalimat dan butir penjelasan. JANGAN mencampuradukkan sebutan entitas dalam satu kalimat (misalnya membuka dengan "Lembaga Jasa Keuangan wajib..." namun menutup dengan "...diterima PUJK"). Gunakan sebutan entitas yang selaras di seluruh kalimat.
   - LARANGAN MERANGKAI KAUSALITAS ANTAR-PASAL (ANTI-CAUSAL STITCHING): DILARANG menggabungkan atau merangkai klausa dari pasal atau regulasi yang berbeda menjadi hubungan sebab-akibat, syarat-kondisi, atau kronologi yang tidak tertulis secara eksplisit dalam teks (contoh pelanggaran: mengarang 'Dalam hal tidak terdapat kesepakatan, Lembaga Jasa Keuangan wajib menangani pengaduan secara tertulis...' atau mengarang 'Jika dokumen pengaduan belum lengkap, Lembaga Jasa Keuangan wajib menyelesaikan pengaduan secara tertulis paling lama 10 hari kerja...'). Gunakan kata hubung aditif netral ('Selain itu,' atau 'Sementara itu,') saat memaparkan pasal yang berbeda.
   - FOKUS KEWAJIBAN ENTITAS (BUKAN HAK SENGKETA PIHAK KETIGA): Jika pertanyaan menanyakan kewajiban suatu entitas dalam penanganan atau penyelesaian suatu proses, fokuskan jawaban pada tindakan, prosedur, dan batas waktu yang wajib dilaksanakan oleh entitas tersebut, BUKAN pada hak atau opsi sengketa konsumen mengajukan pengaduan ke pihak ketiga/otoritas di luar entitas (kecuali ditanyakan secara spesifik).
3. WAJIB mencantumkan sitasi regulasi lengkap dan spesifik dalam tanda kurung siku di akhir SETIAP kalimat pernyataan hukum (format: [<Nama Regulasi>, Pasal <Nomor> ayat (<Angka>)]). Jika jawaban terdiri atas beberapa kalimat yang merujuk pada pasal berbeda (misalnya kalimat pertama dari Pasal X ayat (1) dan kalimat kedua dari Pasal Y ayat (1)), SETIAP kalimat WAJIB ditutup dengan kurung siku sitasinya masing-masing secara akurat. DILARANG menuliskan sitasi dalam bentuk frasa naratif (seperti ', seperti yang ditetapkan dalam...'), menyatukan sitasi pasal yang berbeda ke satu kurung siku tunggal di akhir paragraf, atau menyalin tag metadata header seperti '[Tentang: ...]', '[Peran: ...]', dan '(Status: ...)' ke dalam jawaban.
4. HANYA JIKA konteks regulasi yang diberikan benar-benar TIDAK memuat ketentuan atau dasar hukum yang menjawab pertanyaan (atau jika subjek pertanyaan di luar sektor jasa keuangan seperti kedai kopi atau restoran), nyatakan secara singkat bahwa dasar hukum mengenai hal tersebut tidak ditemukan dalam regulasi yang tersedia. DILARANG merangkum pasal-pasal yang tidak relevan (passage dumping) atau memaksakan ketentuan dari sub-sektor keuangan lain.
5. Jawab dalam bahasa yang sama dengan pertanyaan pengguna (Bahasa Indonesia).
6. Jangan memberikan nasihat hukum formal atau spekulasi non-yuridis.
7. STRUKTUR JAWABAN & SINTESIS KEWAJIBAN BERCABANG: Berikan penjelasan jawaban dalam kalimat prosa yang utuh, padat, dan terstruktur tanpa pengulangan poin yang sama. DILARANG membuat daftar nomor pasal/romawi kosong tanpa isi kalimat ketentuannya. Jika pertanyaan menanyakan kewajiban umum suatu entitas yang dalam regulasi terbagi ke dalam beberapa bentuk/modalitas atau komponen, Anda WAJIB menyajikan seluruh cabang/komponen kewajiban tersebut secara komprehensif dan terstruktur dengan menyertakan sitasi pasal/ayat masing-masing dalam kurung siku pada setiap kalimat/cabang, serta menyederhanakan rujukan silang internal (seperti 'dokumen sebagaimana dimaksud dalam Pasal ...' menjadi 'dokumen yang dipersyaratkan')."""

SYSTEM_PROMPT_EN: str = """You are EasyPasal, an Indonesian financial regulatory compliance assistant (covering OJK, Bank Indonesia, and relevant statutes).

STRICT ANSWERING RULES:
1. TWO-STEP RELEVANCE VALIDATION & INSTITUTIONAL ENTITY EQUIVALENCE:
   - Ground your entire answer ONLY in the provided regulatory context passages below.
   - Understand standard Indonesian statutory drafting conventions: statutory shorthand designations are canonical and equivalent (e.g., "Bank" in banking regulations officially covers "Bank Umum"; "PUJK" covers regulated financial service providers including banks, fintech, and insurers; "Perusahaan" in capital market rules covers "Perusahaan Efek").
   - If the context passages contain substantive rules answering the query using these statutory designations (such as rules for "Bank" answering a question about "Bank Umum"), you MUST answer directly and cite the article.
2. BOTTOM-LINE-UP-FRONT (BLUF) & STATUTORY NORM HIERARCHY:
   - Your very first sentence MUST state the direct, definitive answer to the user's question, citing the primary governing article in square brackets (format: [<Regulation Name>, Pasal <Number> ayat (<Clause>)]). NEVER begin with passive hedging or passage listing.
   - Prioritize Primary Imperative Norms ("wajib", "harus", "dilarang") over permissive exceptions ("dapat"). If an article/clause establishes a general obligation ("wajib") and subsequent clauses provide conditional permissions/exceptions ("dapat ... sepanjang izin"), state the primary general obligation first, then mention the exception if applicable.
   - Legal Subject & Grammatical Agent Alignment: The opening sentence MUST retain the exact institutional entity queried by the user as its grammatical subject (e.g., if asked about "Bank Umum", use "Bank Umum" or "Bank", NEVER replace it with "PUJK" or unrelated sectors). If the question inquires about an entity's obligation or prohibition (e.g. "Bank wajib...", "PUJK must..."), you MUST cite the clause where that entity is the primary grammatical subject of the obligation (e.g. "Bank must place..."), NOT provisions where the supervisory authority is the subject exercising discretionary or corrective enforcement powers (e.g. "OJK may request/order..."). Authority enforcement powers may only be cited as secondary qualifications.
   - Sentence-Wide Entity Consistency & Co-Reference: Maintain the queried institutional entity consistently across all sentences and explanatory clauses. NEVER mix different nomenclature within the same sentence (e.g. opening with "Lembaga Jasa Keuangan must..." but ending with "...received by PUJK"). Use consistent entity naming throughout the response.
   - ANTI-INTER-PASSAGE CAUSAL STITCHING: NEVER combine clauses from different articles or regulations into a conditional, cause-and-effect, or chronological link that is not explicitly stated in the statutory text (e.g., fabricating 'In the absence of agreement, the financial institution must handle complaints in writing...' or 'If complaint documents are incomplete, the institution must resolve written complaints within 10 working days...'). Use neutral additive transitions ('Furthermore,' or 'Additionally,') when presenting distinct articles.
   - FOCUS ON ENTITY OBLIGATIONS (NOT THIRD-PARTY REMEDIES): When answering an entity's operational duties, procedures, or deadlines, focus strictly on the actions the entity itself is mandated to perform, NOT on external dispute escalation options available to consumers.
3. You MUST cite the exact regulation, article, and clause in square brackets at the end of EVERY normative sentence (format: [<Regulation Name>, Pasal <Number> ayat (<Clause>)]). When an answer spans multiple sentences covering distinct articles (e.g. Sentence 1 from Pasal X ayat (1) and Sentence 2 from Pasal Y ayat (1)), EVERY sentence MUST end with its own specific bracketed citation. NEVER collapse distinct articles into a single bracketed citation at the end of the paragraph, NEVER use parenthetical citations like '(POJK ...)', NEVER invent translated English statute names (such as 'the Financial Institutions Act' or 'the Consumer Protection Law'), NEVER place citations as the grammatical subject of a sentence (e.g. '[POJK ...] mandates that...'), NEVER cite general elucidation preambles ('I. Umum') when operative Pasal articles exist, and NEVER copy document header metadata tags such as '[Tentang: ...]', '[Peran: ...]', or '(Status: ...)' into your answer.
4. ONLY IF the provided regulatory passages genuinely do NOT contain rules answering the question (or if the question is outside financial services regulation, e.g., coffee shops, restaurants, retail shops, non-financial enterprises, general criminal/labor law), state briefly that no legal basis was found in the available regulations. STRICTLY FORBIDDEN from summarizing tangentially related passages (passage dumping) or forcing rules across disjoint financial sub-sectors.
5. Answer in the same language as the user's question (English). Retain official Indonesian regulatory names and article designations in your citations.
6. Do not provide formal legal advice or non-legal speculation.
7. STRUCTURE & MULTI-BRANCH OBLIGATION SYNTHESIS: Provide concise, complete prose sentences without repeating identical points or listing bare article numbers without their substantive rules. When a question asks generally about an obligation of an entity, and the governing regulation bifurcates that obligation across multiple modalities, channels, or components, you MUST present all applicable branches comprehensively in a structured manner, accompanied by their respective official bracketed citations on each sentence/branch, and simplify raw internal cross-references (such as 'dokumen sebagaimana dimaksud dalam Pasal ...' into 'dokumen yang dipersyaratkan')."""

# -------------------------------------------------------------------------
# GEPA-Optimized System Prompts (Experiment E5 / T5.1)
# Derived from evolutionary Pareto optimization addressing citation formatting,
# anti-hallucination of pasal numbers, and temporal regulation revocation precedence.
# -------------------------------------------------------------------------
GEPA_SYSTEM_PROMPT_ID: str = """Anda adalah EasyPasal, asisten kepatuhan hukum dan regulasi sektor jasa keuangan Indonesia (OJK, Bank Indonesia, dan UU terkait).

ATURAN MUTLAK PENJAWABAN:
1. VALIDASI RELEVANSI DUA TAHAP & EKUIVALENSI ENTITAS:
   - Dasarkan seluruh jawaban HANYA pada konteks regulasi yang disediakan di bawah ini.
   - Pahami ekuivalensi istilah hukum Indonesia: kata "Bank" dalam peraturan perbankan OJK secara resmi mencakup "Bank Umum"; "PUJK" mencakup lembaga jasa keuangan terkait; dan "Perusahaan" mencakup "Perusahaan Efek".
   - Jika cuplikan teks memuat ketentuan yang menjawab pertanyaan (misalnya aturan "Bank" untuk pertanyaan "Bank Umum"), Anda WAJIB menjawabnya secara langsung dengan mengutip pasalnya.
2. STRUKTUR LANGSUNG DI AWAL (BLUF) & HIERARKI NORMA:
   - Kalimat pertama WAJIB memberikan kesimpulan jawaban substantif secara langsung dan tegas, disertai sitasi nomor pasal/ayat rujukan intinya dalam tanda kurung siku (format: [<Nama Regulasi>, Pasal <Nomor> ayat (<Angka>)]). DILARANG membuka dengan rangkuman daftar dokumen atau kalimat mengambang.
   - Dahulukan Norma Pokok/Imperatif ("wajib", "harus", "dilarang") di atas norma pengecualian/izin ("dapat"). Jika suatu pasal/ayat menetapkan kewajiban umum ("wajib") dan ayat berikutnya mengatur opsi/pengecualian bersyarat ("dapat ... sepanjang izin"), nyatakan kewajiban pokoknya terlebih dahulu (misal: wajib di wilayah Indonesia), baru sebutkan opsi pengecualian izinnya jika relevan.
   - Penyelarasan Subjek Hukum Pertanyaan: Kalimat pertama jawaban WAJIB menggunakan subjek entitas yang ditanyakan pengguna (contoh: jika pertanyaan menanyakan "Bank Umum", gunakan subjek "Bank Umum" atau "Bank", DILARANG menggantinya menjadi "PUJK" atau entitas lain). Jika pertanyaan menanyakan kewajiban atau larangan suatu entitas (misal: "Bank wajib...", "PUJK wajib..."), Anda WAJIB mengutip pasal/ayat yang menetapkan kewajiban entitas tersebut sebagai subjek hukum utama (contoh: "Bank wajib menempatkan..."), BUKAN ayat kewenangan pengawasan atau tindakan korektif otoritas (contoh: "OJK dapat meminta Bank..."). Ketentuan wewenang otoritas hanya menjadi pelengkap jika relevan.
   - Konsistensi Entitas & Ko-referensi Seluruh Kalimat: Pertahankan entitas subjek yang ditanyakan (misal "Lembaga Jasa Keuangan" atau "Bank Umum") secara konsisten di seluruh kalimat dan butir penjelasan. JANGAN mencampuradukkan sebutan entitas dalam satu kalimat (misalnya membuka dengan "Lembaga Jasa Keuangan wajib..." namun menutup dengan "...diterima PUJK"). Gunakan sebutan entitas yang selaras di seluruh kalimat.
   - LARANGAN MERANGKAI KAUSALITAS ANTAR-PASAL (ANTI-CAUSAL STITCHING): DILARANG menggabungkan atau merangkai klausa dari pasal atau regulasi yang berbeda menjadi hubungan sebab-akibat, syarat-kondisi, atau kronologi yang tidak tertulis secara eksplisit dalam teks (contoh pelanggaran: mengarang 'Dalam hal tidak terdapat kesepakatan, Lembaga Jasa Keuangan wajib menangani pengaduan secara tertulis...' atau mengarang 'Jika dokumen pengaduan belum lengkap, Lembaga Jasa Keuangan wajib menyelesaikan pengaduan secara tertulis paling lama 10 hari kerja...'). Gunakan kata hubung aditif netral ('Selain itu,' atau 'Sementara itu,') saat memaparkan pasal yang berbeda.
   - FOKUS KEWAJIBAN ENTITAS (BUKAN HAK SENGKETA PIHAK KETIGA): Jika pertanyaan menanyakan kewajiban suatu entitas dalam penanganan atau penyelesaian suatu proses, fokuskan jawaban pada tindakan, prosedur, dan batas waktu yang wajib dilaksanakan oleh entitas tersebut, BUKAN pada hak atau opsi sengketa konsumen mengajukan pengaduan ke pihak ketiga/otoritas di luar entitas (kecuali ditanyakan secara spesifik).
3. WAJIB mencantumkan sitasi regulasi lengkap dan spesifik dalam tanda kurung siku di akhir SETIAP kalimat pernyataan hukum dengan format baku: [<Nama Regulasi Resmi/Singkat>, Pasal <Nomor> ayat (<Angka>)]. Jika jawaban terdiri atas beberapa kalimat yang merujuk pada pasal berbeda (misalnya kalimat pertama dari Pasal X ayat (1) dan kalimat kedua dari Pasal Y ayat (1)), SETIAP kalimat WAJIB ditutup dengan kurung siku sitasinya masing-masing secara akurat. DILARANG menulis sitasi tanpa kurung siku, menulis frasa pengantar naratif (seperti ', seperti yang ditetapkan dalam...'), menyatukan sitasi pasal berbeda ke satu kurung siku tunggal di akhir paragraf, atau menyalin tag metadata header seperti '[Tentang: ...]', '[Peran: ...]', dan '(Status: ...)' ke dalam jawaban.
4. HANYA JIKA konteks regulasi yang diberikan benar-benar TIDAK memuat ketentuan atau dasar hukum yang menjawab pertanyaan, nyatakan bahwa dasar hukum mengenai hal tersebut tidak ditemukan dalam regulasi yang tersedia. DILARANG merangkum pasal-pasal yang tidak relevan (passage dumping) atau memaksakan ketentuan dari sub-sektor keuangan lain.
5. LARANGAN HALUSINASI PASAL & REGULASI: HANYA gunakan nama regulasi resmi dan nomor pasal yang tertulis secara eksplisit pada header atau isi cuplikan teks konteks di atas. JANGAN PERNAH mengarang nama POJK fiktif, tahun fiktif, atau menebak nomor pasal yang tidak tertulis.
6. STATUS REGULASI & KETENTUAN PENGGANTI: Perhatikan catatan status regulasi pada header dokumen (Status: berlaku vs dicabut/diubah). Selalu utamakan dan dasarkan jawaban pada regulasi yang berstatus BERLAKU. Jika regulasi terdahulu telah dicabut oleh regulasi baru yang berlaku, nyatakan secara tegas bahwa ketentuan tersebut telah digantikan.
7. Jawab dalam bahasa yang sama dengan pertanyaan pengguna (Bahasa Indonesia).
8. Jangan memberikan nasihat hukum formal atau spekulasi non-yuridis.
9. STRUKTUR JAWABAN, KEJELASAN LENGKAP & SINTESIS MULTIKONDISI: Tuliskan kalimat jawaban substantif dalam prosa lengkap dan jelas terlebih dahulu yang menjawab pertanyaan, disusul penjelasan ketentuan terkait. DILARANG membuat daftar nomor pasal/romawi kosong tanpa isi kalimat ketentuannya, dan dilarang HANYA menuliskan kurung siku sitasi tanpa kalimat penjelasan jawaban yang utuh. Jika pertanyaan menanyakan kewajiban umum yang memiliki cabang modalitas atau komponen, sajikan seluruh cabang/komponen tersebut secara lengkap disertai sitasi masing-masing pada setiap kalimat/cabang, serta sederhanakan rujukan silang internal (seperti 'dokumen sebagaimana dimaksud dalam Pasal ...' menjadi 'dokumen yang dipersyaratkan')."""

GEPA_SYSTEM_PROMPT_EN: str = """You are EasyPasal, an Indonesian financial regulatory compliance assistant (covering OJK, Bank Indonesia, and relevant statutes).

STRICT ANSWERING RULES:
1. TWO-STEP RELEVANCE VALIDATION & ENTITY EQUIVALENCE:
   - Ground your entire answer ONLY in the provided regulatory context passages below.
   - Understand standard Indonesian legal equivalence: "Bank" in banking regulations officially covers "Bank Umum"; "PUJK" covers relevant financial institutions; and "Perusahaan" covers "Perusahaan Efek".
   - If the passages contain rules answering the query using these designations (such as rules for "Bank" answering a "Bank Umum" question), you MUST answer directly and cite the article.
2. BOTTOM-LINE-UP-FRONT (BLUF) & STATUTORY NORM HIERARCHY:
   - Your very first sentence MUST state the direct substantive conclusion answering the user's question, citing the primary governing article in square brackets (format: [<Official/Short Regulation Name>, Pasal <Number> ayat (<Clause>)]). NEVER begin with passive hedging or document listing.
   - Prioritize Primary Imperative Norms ("wajib", "harus", "dilarang") over permissive exceptions ("dapat"). If an article/clause establishes a general obligation ("wajib") and subsequent clauses provide conditional permissions/exceptions ("dapat ... sepanjang izin"), state the primary general obligation first, then mention the exception if applicable.
   - Legal Subject & Grammatical Agent Alignment: The opening sentence MUST retain the exact institutional entity queried by the user as its grammatical subject (e.g., if asked about "Bank Umum", use "Bank Umum" or "Bank", NEVER replace it with "PUJK" or unrelated sectors). If the question inquires about an entity's obligation or prohibition (e.g. "Bank wajib...", "PUJK must..."), you MUST cite the clause where that entity is the primary grammatical subject of the obligation (e.g. "Bank must place..."), NOT provisions where the supervisory authority is the subject exercising discretionary or corrective enforcement powers (e.g. "OJK may request/order..."). Authority enforcement powers may only be cited as secondary qualifications.
   - Sentence-Wide Entity Consistency & Co-Reference: Maintain the queried institutional entity consistently across all sentences and explanatory clauses. NEVER mix different nomenclature within the same sentence (e.g. opening with "Lembaga Jasa Keuangan must..." but ending with "...received by PUJK"). Use consistent entity naming throughout the response.
   - ANTI-INTER-PASSAGE CAUSAL STITCHING: NEVER combine clauses from different articles or regulations into a conditional, cause-and-effect, or chronological link that is not explicitly stated in the statutory text (e.g., fabricating 'In the absence of agreement, the financial institution must handle complaints in writing...' or 'If complaint documents are incomplete, the institution must resolve written complaints within 10 working days...'). Use neutral additive transitions ('Furthermore,' or 'Additionally,') when presenting distinct articles.
   - FOCUS ON ENTITY OBLIGATIONS (NOT THIRD-PARTY REMEDIES): When answering an entity's operational duties, procedures, or deadlines, focus strictly on the actions the entity itself is mandated to perform, NOT on external dispute escalation options available to consumers.
3. You MUST cite the exact regulation, article, and clause in square brackets at the end of EVERY normative sentence with standard format: [<Official/Short Regulation Name>, Pasal <Number> ayat (<Clause>)]. When an answer spans multiple sentences covering distinct articles (e.g. Sentence 1 from Pasal X ayat (1) and Sentence 2 from Pasal Y ayat (1)), EVERY sentence MUST end with its own specific bracketed citation. Do not write citations in parentheses '(POJK ...)', NEVER invent English act titles (such as 'the Financial Institutions Act' or 'the Consumer Protection Law'), NEVER use citations as the grammatical subject of a sentence (e.g. '[POJK ...] mandates that...'), NEVER cite general elucidation preambles ('I. Umum') or unretrieved cross-referenced statutes mentioned inside elucidations (e.g. '[Pasal ... UU P2SK]'), and NEVER copy document header metadata tags such as '[Tentang: ...]', '[Peran: ...]', or '(Status: ...)' into your answer.
4. ONLY IF the provided regulatory passages genuinely do NOT contain rules answering the question, state that no legal basis was found in the available regulations. STRICTLY FORBIDDEN from misapplying or analogizing requirements across disjoint financial sectors.
5. STRICT ANTI-HALLUCINATION CLAUSE: ONLY cite official regulations and articles that are explicitly written in the headers or body of the provided context passages. NEVER fabricate fictitious regulation names, old/fake years, or imaginary article/clause numbers.
6. REGULATION STATUS & PRECEDENCE: Observe document status headers (Status: berlaku vs dicabut/diubah). Always prioritize regulations marked as in-force (berlaku). If an earlier regulation was revoked by a newer regulation in force, state explicitly that it has been superseded.
7. Answer in the same language as the user's question (English). Retain official Indonesian regulatory names and article designations in your square-bracket citations. Never omit square brackets when answering in English.
8. Do not provide formal legal advice or non-legal speculation.
9. STRUCTURE, COMPLETENESS & MULTI-BRANCH SYNTHESIS: Write a clear and complete substantive prose sentence answering the question first, accompanied by the primary article/clause citation in square brackets. NEVER output bare article/section outlines without full sentences, and NEVER output a raw citation bracket alone without an accompanying explanatory answer."""


_EN_INDICATOR_WORDS: set[str] = {
    "what",
    "is",
    "are",
    "how",
    "why",
    "when",
    "where",
    "which",
    "who",
    "the",
    "of",
    "according",
    "regulation",
    "bank",
    "capital",
    "consumer",
    "protection",
    "minimum",
    "definition",
    "requirement",
    "complaint",
    "data",
}

_ID_INDICATOR_WORDS: set[str] = {
    "apa",
    "apakah",
    "bagaimana",
    "berapa",
    "kapan",
    "dimana",
    "di",
    "mana",
    "siapa",
    "mengapa",
    "kenapa",
    "yang",
    "dan",
    "dari",
    "pada",
    "untuk",
    "menurut",
    "berdasarkan",
    "peraturan",
    "kewajiban",
    "syarat",
    "konsumen",
    "pelindungan",
    "pengaduan",
    "modal",
    "pujk",
}


def detect_language(text: str) -> str:
    """Detect whether text is primarily Indonesian ('id') or English ('en')."""
    words = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    if not words:
        return "id"

    en_matches = len(words & _EN_INDICATOR_WORDS)
    id_matches = len(words & _ID_INDICATOR_WORDS)

    if en_matches > id_matches:
        return "en"
    return "id"


_EN_TO_ID_REGULATORY_LEXICON: list[tuple[str, str]] = [
    (r"\bconsumer\s+protection\b", "pelindungan konsumen"),
    (
        r"\bdigital\s+financial\s+services\b",
        "layanan digital penyelenggara layanan digital sektor jasa keuangan pujk",
    ),
    (
        r"\bdigital\s+(?:banking|services)\b",
        "layanan digital bank penyelenggara layanan digital",
    ),
    (
        r"\b(?:data\s+protection|personal\s+data|privacy|confidentiality)\b",
        "pelindungan data pribadi kerahasiaan keamanan data informasi",
    ),
    (r"\b(?:consumers?|customers?)\b", "konsumen nasabah"),
    (r"\b(?:principal|principles?|core|fundamental)\b", "prinsip pokok"),
    (
        r"\b(?:obligations?|mandatory|must|shall|required|duties|duty|requirements?)\b",
        "wajib harus kewajiban menerapkan memenuhi",
    ),
    (
        r"\b(?:complaints?|grievances?|disputes?|redress)\b",
        "pengaduan sengketa tindak lanjut penyelesaian",
    ),
    (r"\b(?:authentication|education|literacy)\b", "edukasi autentikasi literasi"),
    (
        r"\b(?:transparency|disclosure|fair\s+treatment)\b",
        "keterbukaan transparansi informasi perlakuan adil",
    ),
    (r"\b(?:commercial\s+banks?|banks?|banking)\b", "bank umum perbankan"),
    (r"\b(?:minimum\s+capital|core\s+capital)\b", "modal inti minimum"),
    (
        r"\b(?:data\s+centers?|disaster\s+recovery)\b",
        "pusat data pusat pemulihan bencana wilayah indonesia",
    ),
    (
        r"\b(?:cyber(?:security)?|cyber\s+resilience)\b",
        "ketahanan keamanan siber insiden siber",
    ),
    (r"\b(?:governance|risk\s+management)\b", "tata kelola manajemen risiko"),
]


def expand_cross_lingual_query_terms(text: str) -> str:
    """Append canonical Indonesian regulatory terms to English or mixed-language text."""
    if not text:
        return ""
    expansions: list[str] = []
    for pattern, id_terms in _EN_TO_ID_REGULATORY_LEXICON:
        if re.search(pattern, text, re.IGNORECASE):
            expansions.append(id_terms)
    if not expansions:
        return text
    return f"{text} {' '.join(expansions)}"


def is_enumeration_query(query: str) -> bool:
    """Detect whether the query asks for a taxonomic list, pillars, components, aspects, principles, or obligations."""
    if not query:
        return False
    return bool(
        re.search(
            r"\b(?:apa\s+saja|sebutkan|meliputi\s+apa|mencakup\s+apa|terdiri\s+(?:dari|atas)\s+apa|"
            r"pilar|tahapan|komponen|faktor|aspek|prinsip|unsur|kriteria|jenis-jenis|"
            r"what\s+are\s+the\s+(?:[\w\-]+\s+){0,4}?(?:pillars|components|factors|aspects|stages|principles|elements|processes|obligations|requirements|duties|provisions|rules)|"
            r"(?:principal|main|key|core|primary)\s+(?:[\w\-]+\s+){0,3}?(?:obligations|principles|requirements|duties|pillars|components))\b",
            query,
            re.IGNORECASE,
        )
    )


def is_taxonomic_enumeration_clause(text: str) -> bool:
    """Detect whether a regulatory clause defines a foundational taxonomic enumeration
    (pillars, principles, aspects, stages, or required core processes) rather than a narrow
    contractual/documentary item list (e.g., fee disclosures in agreements or application attachments).
    """
    if not text:
        return False
    stripped = text.strip()
    header_match = re.match(r"^\[([^\]]+)\]\s*", stripped)
    body = stripped[header_match.end() :] if header_match else stripped
    norm = re.sub(r"\s+", " ", body.lower()).strip()

    # Exclude narrow contractual, fee/commission disclosure, or licensing attachment sub-lists
    intro = norm.split(":", maxsplit=1)[0] if ":" in norm else norm[:160]
    if re.search(
        r"\b(?:dalam\s+perjanjian|mencantumkan|melampirkan|surat\s+permohonan|komisi/imbalan|agen\s+pemasar|dokumen\s+persyaratan)\b",
        intro,
    ):
        return False

    # 1. Explicit taxonomic category or core process enumeration introducers
    if re.search(
        r"(?:menerapkan\s+prinsip|melakukan\s+proses(?:\s+paling\s+sedikit)?|"
        r"(?:mencakup|meliputi|terdiri\s+(?:atas|dari))\s+(?:\d+\s*\([^)]+\)\s+)?(?:aspek|faktor|pilar|proses|tahap|tahapan|komponen|prinsip|unsur|kegiatan))\s*[^:]{0,40}:\s*[a-z1-9][\.\)]",
        norm,
    ):
        return True

    # 2. General multi-item enumeration ("paling sedikit:", "meliputi:", "mencakup:") with at least 3 items (a., b., c.)
    return bool(
        re.search(
            r"(?:paling\s+sedikit|mencakup|meliputi|terdiri\s+atas|yaitu)\s*:\s*a[\.\)]",
            norm,
        )
        and re.search(r"(?:^|[:;\s])c[\.\)]\s+", norm)
    )


def _is_appendix_or_closing_fragment(text: str, citation_label: str = "") -> bool:
    """Detect closing-section ('PENUTUP'), general elucidation preamble ('I. UMUM'), or multi-part appendix fragments."""
    if not text:
        return False
    stripped = text.strip()
    header_match = re.match(r"^\[([^\]]+)\]\s*", stripped)
    header_low = header_match.group(1).lower() if header_match else ""
    body = stripped[header_match.end() :].strip() if header_match else stripped
    label_low = (citation_label or "").lower()

    # 1. Closing / transitional headers ("Romawi X PENUTUP", "BAB XIV KETENTUAN PENUTUP") or general elucidation preambles ("I. UMUM")
    if re.search(
        r"\b(?:penutup|ketentuan\s+penutup|ketentuan\s+peralihan|romawi\s+x\b|i\.\s*umum\b|bagian\s+umum\b)",
        f"{header_low} {label_low}",
    ):
        return True

    # 2. Multi-part split continuation chunks ("(bagian 2)" .. "(bagian 19)")
    if re.search(r"\(bagian\s+(?:[2-9]|\d{2,})\)", label_low):
        return True

    # 3. Mid-sentence continuation fragment (starts with lowercase conjunction/word or deep worksheet numbering "3.1.g.")
    return bool(
        re.match(r"^(?:maupun|serta|atau|dan|yang|dengan)\b", body, re.IGNORECASE)
        or (body and body[0].islower() and not re.match(r"^[a-z][\.\)]\s", body))
        or re.match(r"^\d+\.\d+\.[a-z][\.\)]\s", body, re.IGNORECASE)
    )


def _detect_statutory_role(text: str) -> str:
    """Classify the primary normative role of a regulatory passage chunk.

    Tags help smaller LLMs prioritize primary substantive duties of regulated entities
    over discretionary supervisory actions, conditional exceptions, or elucidations.
    """
    if not text:
        return ""

    stripped = text.strip()
    header_match = re.match(r"^\[([^\]]+)\]\s*", stripped)
    header_low = header_match.group(1).lower() if header_match else ""
    body = stripped[header_match.end() :] if header_match else stripped
    norm = re.sub(r"\s+", " ", body.lower()).strip()

    if not norm:
        return ""

    # 0. Elucidation (Penjelasan) or Closing/Appendix matrix fragments are not primary norms
    if header_low.startswith("penjelasan") or re.match(
        r"^(?:ayat\s+\(\d+\)|pasal\s+\d+|huruf\s+[a-z])\s+cukup\s+jelas\b", norm
    ):
        return "[Peran: Penjelasan / Keterangan Tambahan]"

    if _is_appendix_or_closing_fragment(text):
        return "[Peran: Penjelasan / Keterangan Tambahan]"

    # 1. Authority corrective / discretionary supervisory / licensing / sanction powers
    # e.g., "OJK dapat meminta", "Otoritas Jasa Keuangan memberikan izin", "dikenai sanksi"
    auth_pattern = (
        r"\b(?:ojk|otoritas\s+jasa\s+keuangan|bank\s+indonesia|bi|ppatk)\s+"
        r"(?:dapat|berwenang|meminta|memerintahkan|mengenakan|menjatuhkan|"
        r"memberikan\s+izin|menolak\s+permohonan|menentukan|menetapkan|melakukan\s+pemeriksaan)\b"
        r"|\bdalam\s+hal\s+berdasarkan\s+penilaian\s+(?:ojk|otoritas\s+jasa\s+keuangan|bank\s+indonesia)\b"
        r"|\b(?:dikenai|pengenaan)\s+sanksi\b"
    )
    auth_match = re.search(auth_pattern, norm)
    has_authority = auth_match is not None

    # 2. Conditional permission / licensing exceptions: "Bank dapat menempatkan ... sepanjang izin"
    perm_pattern = (
        r"\b(?:bank\s+umum|bank|bpr|pujk|pjk|perusahaan\s+efek|perusahaan|penyelenggara|ljk|emiten)\s+"
        r"dapat\s+(?:menempatkan|mengajukan\s+permohonan|melakukan|menggunakan|menunjuk|memilih)\b"
        r"|\bdapat\b[^.]{0,140}\b(?:sepanjang|setelah\s+memperoleh|dengan\s+persetujuan|izin)\b"
        r"|\bmengajukan\s+permohonan\s+izin\b"
    )
    perm_match = re.search(perm_pattern, norm)
    has_permission = perm_match is not None

    # 2b. Consumer dispute filing & external remedy options:
    # e.g., "Konsumen dapat: a. menyampaikan pengaduan...", "mengajukan sengketa kepada lembaga atau badan..."
    remedy_pattern = (
        r"\b(?:konsumen|nasabah|pengguna)\s+dapat\s*:\s*(?:[a-z]\.|\d+\.)"
        r"|\b(?:konsumen|nasabah|pengguna)\s+dapat\s+(?:menyampaikan\s+pengaduan|mengajukan\s+sengketa|memilih\s+upaya|mengajukan\s+keberatan)\b"
        r"|\b(?:mengajukan\s+sengketa\s+kepada\s+lembaga|penyelesaian\s+sengketa\s+(?:di\s+luar\s+pengadilan|melalui\s+pengadilan|ke\s+laps))\b"
        r"|\bdalam\s+hal\s+tidak\s+terdapat\s+kesepakatan\b[^.]{0,100}\bkonsumen\s+dapat\b"
    )
    remedy_match = re.search(remedy_pattern, norm)
    has_consumer_remedy = remedy_match is not None

    # 3. Primary imperative duties of entities: "Bank wajib", "PUJK wajib", "Perusahaan dilarang"
    # Exclude passive relative clauses like "yang diterapkan Bank harus disesuaikan"
    norm_for_duty = re.sub(
        r"\byang\s+(?:diterapkan|dilakukan|dilaksanakan|disusun|ditetapkan)\s+(?:oleh\s+)?(?:bank\s+umum|bank|bpr|pujk|ljk|perusahaan|penyelenggara)\s+(?:wajib|harus)\b",
        " ",
        norm,
    )
    duty_pattern = (
        r"\b(?:bank\s+umum|bank|bpr|pujk|pjk|perusahaan\s+efek|perusahaan|penyelenggara|ljk|emiten|"
        r"pihak|direksi|dewan\s+komisaris|pelaku\s+usaha)\s+(?:wajib|harus|dilarang)\b"
        r"|\bwajib\s+(?:menempatkan|memiliki|menerapkan|menyediakan|menyampaikan|memenuhi|"
        r"menggunakan|menjaga|melakukan|memastikan|melaksanakan|menyusun|menindaklanjuti)\b"
        r"|\b(?:bank\s+penyelenggara\s+layanan\s+digital|pujk|bank(?:\s+umum)?)\s+"
        r"(?:menerapkan\s+prinsip\s+pelindungan|menyediakan\s+fungsi\s+penanganan|memberikan\s+edukasi\s+kepada\s+nasabah)\b"
    )
    duty_match = re.search(duty_pattern, norm_for_duty)
    has_entity_duty = duty_match is not None

    # 3b. Statutory Enumeration / Pillar Elaboration Clauses:
    # e.g., "(2) Untuk menjaga ketahanan siber sebagaimana dimaksud pada ayat (1), Bank melakukan proses paling sedikit: a. ..."
    # or "2. Penerapan manajemen risiko terkait keamanan siber mencakup 4 (empat) aspek, yaitu: a. ..."
    enum_duty_pattern = (
        r"(?:untuk\s+(?:menjaga|melaksanakan|menerapkan|memastikan|memenuhi)|dalam\s+(?:menerapkan|melaksanakan|menyelenggarakan))\s+"
        r"[^:]{3,140}?\b(?:bank\s+umum|bank|bpr|pujk|ljk|perusahaan|penyelenggara)\s+"
        r"(?:melakukan|mempertimbangkan|menerapkan|menetapkan|menjaga)\s+[^:]{0,60}:\s*[a-z1-9][\.\)]"
        r"|\b(?:penerapan|pelaksanaan|penyelenggaraan)\s+[^:]{3,120}?\b(?:mencakup|meliputi|terdiri\s+atas|paling\s+sedikit)\s+"
        r"(?:\d+\s*\([^)]+\)\s+)?(?:aspek|faktor|pilar|proses|tahap|komponen|prinsip)[^:]*:\s*[a-z1-9][\.\)]"
        r"|\b(?:pelindungan\s+konsumen|pujk|bank(?:\s+umum)?|ljk|penyelenggara|perusahaan)[^:]{0,60}?"
        r"\b(?:menerapkan|melaksanakan|memenuhi)\s+prinsip\s*:\s*[a-z1-9][\.\)]"
    )
    has_enum_duty = re.search(enum_duty_pattern, norm) is not None

    # Leading conditional supervisory assessment ("Dalam hal berdasarkan penilaian OJK...")
    if (
        re.match(
            r"^(?:\(\d+\)\s*)?dalam\s+hal\s+(?:berdasarkan\s+penilaian|terdapat\s+kondisi)\b",
            norm,
        )
        and has_authority
    ):
        return "[Peran: Wewenang Korektif Otoritas / Sanksi]"

    # Leading permissive / licensing application clause ("(1) Bank dapat mengajukan permohonan izin...")
    if (
        perm_match
        and perm_match.start() < 35
        and (not duty_match or perm_match.start() < duty_match.start())
    ):
        return "[Peran: Opsi Pengecualian / Izin Bersyarat]"

    if has_entity_duty and has_authority:
        pos_duty = duty_match.start() if duty_match else 999999
        pos_auth = auth_match.start() if auth_match else 999999
        if pos_duty < pos_auth:
            return "[Peran: Kewajiban Pokok Entitas]"
        return "[Peran: Wewenang Korektif Otoritas / Sanksi]"
    if has_entity_duty or has_enum_duty:
        return "[Peran: Kewajiban Pokok Entitas]"
    if has_consumer_remedy:
        return "[Peran: Opsi Sengketa / Pemulihan Konsumen]"
    if has_authority:
        return "[Peran: Wewenang Korektif Otoritas / Sanksi]"
    if has_permission:
        return "[Peran: Opsi Pengecualian / Izin Bersyarat]"

    return ""


def _legal_stem(word: str) -> str:
    """Lightweight morphological stemmer for Indonesian legal terms."""
    w = word.lower().strip()
    if len(w) <= 3:
        return w
    for suf in ("kan", "an", "i", "nya"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    for pref in ("peny", "meny"):
        if w.startswith(pref) and len(w) - len(pref) >= 2:
            return "s" + w[len(pref) :]
    for pref in ("peng", "meng", "pen", "men", "per", "ber", "ter", "di", "ke", "se"):
        if w.startswith(pref) and len(w) - len(pref) >= 3:
            return w[len(pref) :]
    for pref in ("pem", "mem"):
        if w.startswith(pref) and len(w) - len(pref) >= 3:
            return "p" + w[len(pref) :]
    return w


def score_primary_obligation_alignment(chunk: RetrievedChunk, query: str) -> float:
    """Score how strongly a chunk serves as the primary imperative obligation answering an obligation query.

    Distinguishes standalone root obligations (e.g., Pasal 35 ayat (1) 'Bank wajib menempatkan...')
    from derivative/cross-referencing clauses (e.g., Pasal 36 ayat (3) or Pasal 37 'Bank wajib memastikan
    ... sebagaimana dimaksud dalam Pasal 35') and non-obligation clauses.
    """
    if not chunk or not chunk.text:
        return 0.0

    if _is_appendix_or_closing_fragment(chunk.text, chunk.citation_label or ""):
        return 0.0

    role = _detect_statutory_role(chunk.text)
    if role != "[Peran: Kewajiban Pokok Entitas]":
        return 0.0

    stripped = chunk.text.strip()
    header_match = re.match(r"^\[([^\]]+)\]\s*", stripped)
    body = stripped[header_match.end() :] if header_match else stripped
    norm_body = re.sub(r"\s+", " ", body.lower()).strip()
    norm_query = re.sub(r"\s+", " ", (query or "").lower()).strip()
    expanded_query = expand_cross_lingual_query_terms(norm_query)

    # 0. Institutional Subject Scope Concordance:
    # Exclude obligations binding specialized dispute bodies (e.g. LAPS) when the query asks about regulated financial institutions
    if (
        "laps" not in norm_query
        and "alternatif penyelesaian sengketa" not in norm_query
        and re.search(
            r"^(?:\(\d+\)\s*)?laps\b|\blaps\s+(?:sektor\s+jasa\s+keuangan\s+)?(?:wajib|harus|dilarang)\b",
            norm_body,
        )
    ):
        return 0.0

    score = 1.0
    is_enum_q = is_enumeration_query(norm_query)

    # 1. Action Predicate Concordance ("wajib <verb>", "kewajiban ... dalam <verb>", or stem matching)
    generic_meta_verbs = {
        "wajib",
        "harus",
        "dilarang",
        "kewajiban",
        "keharusan",
        "diterapkan",
        "menerapkan",
        "dilakukan",
        "melakukan",
        "dipenuhi",
        "memenuhi",
        "dilaksanakan",
        "melaksanakan",
        "dijalankan",
        "menjalankan",
        "dipatuhi",
        "mematuhi",
        "ditaati",
    }
    q_action_verbs = set()
    m_action = re.search(r"\b(?:wajib|harus|dilarang)\s+(\w+)", norm_query)
    if m_action and m_action.group(1) not in generic_meta_verbs:
        q_action_verbs.add(m_action.group(1))
    for m in re.finditer(
        r"\b(?:kewajiban|ketentuan)\s+(?:[\w\s]{1,40}\s+)?(?:dalam|untuk)\s+(\w+)",
        norm_query,
    ):
        if m.group(1) not in generic_meta_verbs:
            q_action_verbs.add(m.group(1))
    for m in re.finditer(r"\bdalam\s+(\w+)(?:\s+dan\s+(\w+))?", norm_query):
        if m.group(1) and m.group(1) not in generic_meta_verbs:
            q_action_verbs.add(m.group(1))
        if m.group(2) and m.group(2) not in generic_meta_verbs:
            q_action_verbs.add(m.group(2))

    action_matched = False
    if q_action_verbs:
        q_stems = {_legal_stem(v) for v in q_action_verbs if len(v) >= 4}
        body_words = re.findall(r"\b[a-z]{4,}\b", norm_body)
        body_stems = {_legal_stem(w) for w in body_words}
        if q_stems & body_stems and re.search(r"\b(?:wajib|harus|dilarang)\b", norm_body):
            score += 2.5
            action_matched = True

    if not action_matched and m_action:
        action_verb = m_action.group(1)
        if len(action_verb) >= 4 and re.search(
            rf"\b(?:wajib|harus|dilarang)\s+(?:\w+\s+)?{re.escape(_legal_stem(action_verb))}\w*\b",
            norm_body,
        ):
            score += 1.5 if action_verb in generic_meta_verbs else 2.5

    # 2. Standalone Root Norm vs. Dependent Cross-Referencing Clause
    is_dependent_ref = bool(
        re.search(
            r"^(?:\(\d+\)\s*)?(?:kewajiban|ketentuan|sanksi|persyaratan)\s+sebagaimana\s+dimaksud\s+dalam\s+pasal\b",
            norm_body,
        )
        or re.search(
            r"\bwajib\s+(?:memastikan|memenuhi|melaksanakan)\s+[^.]{1,80}\bsebagaimana\s+dimaksud\s+dalam\s+pasal\b",
            norm_body,
        )
    )
    has_structured_enum = is_taxonomic_enumeration_clause(chunk.text)
    has_explicit_substance = bool(
        re.search(r"\bpaling\s+(?:lama|lambat)\s+\d+", norm_body)
        or ("di wilayah indonesia" in norm_body and "di luar wilayah" not in norm_body)
        or (is_enum_q and has_structured_enum)
    )

    if not is_dependent_ref or has_explicit_substance:
        score += 1.2
    else:
        score -= 0.6

    # 3. Root Clause Structural Priority (ayat (1) or un-numbered single-clause Pasal, or enumeration on enum query)
    if str(chunk.ayat) == "1" or chunk.ayat is None or (is_enum_q and has_structured_enum):
        score += 0.5
    elif (
        is_enum_q
        and chunk.ayat is not None
        and str(chunk.ayat).isdigit()
        and int(str(chunk.ayat)) >= 4
    ):
        # Penalize deep procedural sub-clauses (ayat (4)+) on high-level enumeration/principle queries
        score -= 0.8

    # 4. Interrogative Target Concordance ("di mana" -> domestic placement vs overseas exception)
    if ("di mana" in norm_query or "dimana" in norm_query or "where" in norm_query) and (
        "di wilayah indonesia" in norm_body or "di indonesia" in norm_body
    ):
        if "di luar wilayah indonesia" not in norm_body:
            score += 0.8
        else:
            score += 0.2

    # 5. Concrete Numerical Statutory Deadlines / Time Limits
    if re.search(r"\bpaling\s+(?:lama|lambat)\s+\d+\s*(?:\([a-z\s]+\)\s*)?hari\b", norm_body):
        score += 1.5

    # 6. Binding Regulatory Precedence (UU, POJK, PBI, PP over SEOJK circulars)
    doc_id_low = (chunk.doc_id or "").lower()
    if any(p in doc_id_low for p in ("pojk", "pbi", "uu_", "pp_")):
        score += 0.5
    elif any(p in doc_id_low for p in ("seojk", "sebi")) and not (
        is_enum_q and has_structured_enum
    ):
        score -= 0.3

    # 7. Topical Content Keyword Overlap & Enumeration Completeness Boost
    stop_tokens = {
        "apa",
        "apakah",
        "saja",
        "bagaimana",
        "berapa",
        "kapan",
        "dimana",
        "di",
        "mana",
        "siapa",
        "mengapa",
        "yang",
        "dan",
        "atau",
        "dari",
        "pada",
        "oleh",
        "untuk",
        "dalam",
        "menurut",
        "berdasarkan",
        "regulasi",
        "peraturan",
        "ketentuan",
        "ojk",
        "otoritas",
        "jasa",
        "keuangan",
        "wajib",
        "harus",
        "dilarang",
        "bank",
        "umum",
        "perbankan",
        "pujk",
        "pilar",
        "diterapkan",
        "dilakukan",
        # English query tokens (after cross-lingual expansion into Indonesian terms)
        "what",
        "are",
        "the",
        "for",
        "under",
        "rules",
        "regulation",
        "regulations",
        "according",
        "with",
        "from",
        "that",
        "this",
        "which",
        "when",
        "where",
        "principal",
        "principles",
        "principle",
        "obligations",
        "obligation",
        "requirements",
        "requirement",
        "duties",
        "duty",
        "mandatory",
        "must",
        "shall",
        "required",
        "consumer",
        "consumers",
        "customer",
        "customers",
        "protection",
        "financial",
        "services",
        "banking",
        "commercial",
        "minimum",
        "capital",
        "core",
    }
    q_tokens = {t for t in re.findall(r"\b[a-z0-9]{4,}\b", expanded_query) if t not in stop_tokens}
    if q_tokens:
        b_tokens = set(re.findall(r"\b[a-z0-9]{4,}\b", norm_body))
        q_stemmed = {_legal_stem(t) for t in q_tokens}
        b_stemmed = {_legal_stem(t) for t in b_tokens}
        overlap = len(q_stemmed & b_stemmed) / len(q_stemmed)
        score += overlap * 2.5
        if is_enum_q and has_structured_enum and overlap > 0:
            score += 2.5

    return round(score, 4)


def format_passages(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved regulatory chunks into a structured context string."""
    if not chunks:
        return "TIDAK ADA KONTEKS REGULASI YANG DITEMUKAN."

    formatted_blocks: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        raw_label = chunk.citation_label or chunk.chunk_id
        label = re.sub(r"\s*\(bagian\s+\d+\)", "", raw_label, flags=re.IGNORECASE).strip()
        title_note = f"[Tentang: {chunk.doc_title}]" if chunk.doc_title else ""
        role_note = _detect_statutory_role(chunk.text)
        status_note = f"(Status: {chunk.status})" if chunk.status else ""
        header_parts = [f"--- DOKUMEN {idx}: {label}"]
        if title_note:
            header_parts.append(title_note)
        if role_note:
            header_parts.append(role_note)
        if status_note:
            header_parts.append(status_note)
        header = f"{' '.join(header_parts)} ---".strip()
        body = chunk.text.strip()
        # Strip redundant leading pipe-delimited breadcrumb bracket "[DOC | BAB | Pasal]" from body
        body = re.sub(r"^\[[^\]]*\|[^\]]*\]\s*", "", body).strip()
        formatted_blocks.append(f"{header}\n{body}")

    return "\n\n".join(formatted_blocks)


_KNOWN_INSTITUTIONAL_SUBJECTS: list[str] = [
    "Bank Umum Syariah",
    "Bank Umum",
    "Bank Syariah",
    "BPR Syariah",
    "BPR",
    "Bank",
    "PUJK",
    "Perusahaan Efek",
    "Manajer Investasi",
    "Penyelenggara LPBBTI",
    "Penyelenggara Fintech",
    "Penyelenggara P2P",
    "Penyelenggara",
    "Perusahaan Asuransi",
    "Perusahaan Pembiayaan",
    "Lembaga Jasa Keuangan",
    "LJK",
    "Dana Pensiun",
    "Emiten",
]

_SECTORAL_SUBJECT_MAP: list[tuple[str, str]] = [
    (r"\bperbankan\s+syariah\b", "Bank Syariah"),
    (r"\bperbankan\b", "Bank"),
    (r"\bperasuransian\b", "Perusahaan Asuransi"),
    (r"\bdana\s+pensiun\b", "Dana Pensiun"),
    (r"\blemba(?:ga)?\s+pembiayaan\b", "Perusahaan Pembiayaan"),
    (r"\b(?:pinjaman\s+daring|pendanaan\s+bersama)\b", "Penyelenggara LPBBTI"),
]


def extract_query_subject(question: str) -> str | None:
    """Extract the primary institutional financial entity referenced in a query."""
    if not question:
        return None
    for subj in _KNOWN_INSTITUTIONAL_SUBJECTS:
        if re.search(rf"\b{re.escape(subj)}\b", question, re.IGNORECASE):
            return subj
    for pat, canonical_subj in _SECTORAL_SUBJECT_MAP:
        if re.search(pat, question, re.IGNORECASE):
            return canonical_subj
    return None


def build_rag_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    language: str | None = None,
    system_prompt_override: str | None = None,
    prompt_variant: str = "gepa",
) -> str:
    """Assemble system rules, context passages, and user question into a complete prompt."""
    lang = language or detect_language(question)
    if system_prompt_override:
        system_prompt = system_prompt_override
    elif prompt_variant == "baseline":
        system_prompt = SYSTEM_PROMPT_EN if lang == "en" else SYSTEM_PROMPT_ID
    else:
        system_prompt = GEPA_SYSTEM_PROMPT_EN if lang == "en" else GEPA_SYSTEM_PROMPT_ID

    passages_text = format_passages(chunks)
    query_subj = extract_query_subject(question)
    subj_id = query_subj or "Entitas Jasa Keuangan"
    subj_en = query_subj or "Financial institutions"
    example_cite = (
        re.sub(
            r"\s*\(bagian\s+\d+\)",
            "",
            chunks[0].citation_label or chunks[0].chunk_id,
            flags=re.IGNORECASE,
        ).strip()
        if chunks
        else "POJK, Pasal 1"
    )

    if lang == "en":
        return f"""<system>
{system_prompt}
</system>

<context>
{passages_text}
</context>

<question>
{question}
</question>

Provide a direct, comprehensive answer in complete prose sentences citing official regulations in square brackets (NEVER copy header metadata tags like [Tentang: ...], [Peran: ...], or (Status: ...)). If the obligation covers multiple modalities or components, present each clearly with its citation.
Format example:
{subj_en} must comply with regulatory requirements [{example_cite}].

Answer:"""

    return f"""<system>
{system_prompt}
</system>

<context>
{passages_text}
</context>

<question>
{question}
</question>

Jawab secara langsung dan komprehensif dalam kalimat prosa utuh dengan mencantumkan sitasi pasal utama dalam tanda kurung siku (DILARANG menyalin tag header seperti [Tentang: ...], [Peran: ...], (Status: ...), atau membuat daftar nomor pasal kosong). Jika ketentuan mencakup beberapa bentuk/modalitas atau komponen, sebutkan secara terstruktur disertai sitasi masing-masing.
Contoh format:
{subj_id} wajib memenuhi ketentuan terkait [{example_cite}].

Jawaban:"""
