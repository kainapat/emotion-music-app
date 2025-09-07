from transformers import pipeline
from pythainlp import word_tokenize

# Lexicon fallback
THAI_LEXICON = {
    "เศร้า": "sad", "เสียใจ": "sad",
    "เหงา": "lonely", "เดียวดาย": "lonely",
    "หวัง": "hope", "ความหวัง": "hope",
    "สุข": "happy", "ยินดี": "happy",
    "เร้าใจ": "excited", "ตื่นเต้น": "excited",
    "สงบ": "calm", "เยือกเย็น": "calm",
    "โกรธ": "angry", "โมโห": "angry",
}

CANDIDATE_LABELS = ["sad","lonely","hope","happy","excited","calm","angry","neutral"]

# ใช้โมเดลที่ stable
ZS_MODEL = "facebook/bart-large-mnli"
_zs = pipeline("zero-shot-classification", model=ZS_MODEL)

def _lexicon_fallback(text: str) -> str:
    for w in word_tokenize(text):
        if w in THAI_LEXICON:
            return THAI_LEXICON[w]
    return "neutral"

def detect_emotion(text: str, threshold: float = 0.55, multi_label: bool = False) -> str:
    if not text.strip():
        return "neutral"
    try:
        res = _zs(text, candidate_labels=CANDIDATE_LABELS, multi_label=multi_label)
        if multi_label:
            picked = [lbl for lbl, sc in zip(res["labels"], res["scores"]) if sc >= threshold]
            return picked[0] if picked else _lexicon_fallback(text)
        else:
            lbl, sc = res["labels"][0], res["scores"][0]
            return lbl if sc >= threshold else _lexicon_fallback(text)
    except Exception:
        return _lexicon_fallback(text)
