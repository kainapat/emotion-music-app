import re
from pythainlp.tokenize import word_tokenize as thai_tokenize
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

_SECTION_PATTERNS = [
    r'^\s*(intro|อินโทร)\s*:?\s*$', 
    r'^\s*(verse\s*\d*|ท่อน\s*\d*)\s*:?\s*$',
    r'^\s*(chorus|hook|คอรัส|ท่อนฮุก)\s*:?\s*$',
    r'^\s*(bridge|บริดจ์)\s*:?\s*$',
    r'^\s*(outro|เอาท์โทร)\s*:?\s*$',
]

def auto_tokenize(text: str) -> str:
    """
    Automatically tokenize mixed Thai-English text
    Returns formatted text with proper word boundaries
    """
    if not text:
        return ""
    
    # แยกบรรทัด
    lines = text.split('\n')
    tokenized_lines = []
    
    for line in lines:
        if not line.strip():
            tokenized_lines.append('')
            continue
            
        # แยกส่วนไทย-อังกฤษ
        parts = re.split(r'([A-Za-z]+(?:\s+[A-Za-z]+)*)', line)
        tokenized_parts = []
        
        for part in parts:
            if not part.strip():
                continue
            # ถ้าเป็นภาษาอังกฤษ
            if re.match(r'^[A-Za-z\s]+$', part):
                tokens = nltk.word_tokenize(part)
                tokenized_parts.append(' '.join(tokens))
            # ถ้าเป็นภาษาไทย
            else:
                tokens = thai_tokenize(part)
                tokenized_parts.append(' '.join(tokens))
                
        tokenized_lines.append(' '.join(tokenized_parts))
    
    return '\n'.join(tokenized_lines)

def _clean_text(s: str) -> str:
    s = re.sub(r'https?://\S+', ' ', s)     # remove URL
    s = re.sub(r'#[\wก-๙]+', ' ', s)        # remove hashtags
    s = re.sub(r'[^\S\r\n]+', ' ', s)       # collapse whitespace
    s = re.sub(r'[^\x00-\x7Fก-๙\r\n ]', ' ', s)  # remove emoji/special symbols
    return s.strip()

def preprocess_lyrics(lyrics: str):
    """
    Split lyrics into sections based on song structure. If no headers, split by paragraph or long sentences.
    Returns: list[str] (non-empty)
    """
    if not lyrics:
        return []

    text = _clean_text(lyrics)
    lines = [l.strip() for l in text.splitlines()]

    sections = []
    cur = []
    def flush():
        if cur:
            joined = " ".join(cur).strip()
            if len(joined) > 0:
                sections.append(joined)
        cur.clear()

    section_regex = re.compile("|".join(_SECTION_PATTERNS), flags=re.IGNORECASE)

    found_header = False
    for ln in lines:
        if not ln:
            # blank line = new section if accumulating
            if cur:
                flush()
            continue
        if section_regex.match(ln):
            found_header = True
            flush()
        else:
            cur.append(ln)
    flush()

    if not sections:
        # Fallback: split by paragraph
        paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        if paras:
            sections = paras
        else:
            # Final fallback: split by long sentences every ~120 chars
            chunk, tmp = [], []
            cnt = 0
            for tok in re.split(r'([.!?…]|[。！？])', text):
                if not tok: 
                    continue
                tmp.append(tok)
                cnt += len(tok)
                if cnt >= 120:
                    chunk.append("".join(tmp).strip())
                    tmp, cnt = [], 0
            if tmp:
                chunk.append("".join(tmp).strip())
            sections = [c for c in chunk if c]

    # Limit length and number (avoid too long)
    out = []
    for seg in sections:
        seg = re.sub(r'\s+', ' ', seg).strip()
        if len(seg) > 400:
            # split into smaller parts every ~200 chars
            for i in range(0, len(seg), 200):
                part = seg[i:i+200].strip()
                if part:
                    out.append(part)
        elif seg:
            out.append(seg)

    # Limit to 20 sections for speed
    return out[:20]
