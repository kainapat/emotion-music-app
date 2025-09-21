from transformers import pipeline
from pythainlp import word_tokenize

# Lexicon mapping ระหว่างไทย-อังกฤษ
THAI_TO_ENG = {
    # เศร้า
    "เศร้า": "sad", "เสียใจ": "sad", "หม่น": "sad", "หมอง": "sad", 
    "หดหู่": "sad", "ซึม": "sad", "ร้องไห้": "sad", "น้ำตา": "sad",
    "ทุกข์": "sad", "น้อยใจ": "sad", "ผิดหวัง": "sad",
    
    # เหงา
    "เหงา": "lonely", "เดียวดาย": "lonely", "ว้าเหว่": "lonely",
    
    # หวัง
    "หวัง": "hope", "ความหวัง": "hope", "มีหวัง": "hope", 
    "กำลังใจ": "hope", "สู้": "hope", "พยายาม": "hope",
    
    # สุข
    "สุข": "happy", "ยินดี": "happy", "ดีใจ": "happy", 
    "ร่าเริง": "happy", "สดใส": "happy", "สนุก": "happy",
    "แฮปปี้": "happy", "ยิ้ม": "happy", "เบิกบาน": "happy",
    
    # ตื่นเต้น
    "เร้าใจ": "excited", "ตื่นเต้น": "excited", "พีค": "excited",
    "มัน": "excited", "เปรี้ยว": "excited", "ฮึกเหิม": "excited",
    
    # สงบ
    "สงบ": "calm", "เยือกเย็น": "calm", "นิ่ง": "calm",
    "ใจเย็น": "calm", "ผ่อนคลาย": "calm", "ชิล": "calm",
    
    # โกรธ
    "โกรธ": "angry", "โมโห": "angry", "เดือด": "angry",
    "แค้น": "angry", "โกรธา": "angry", "เคือง": "angry",
    
    # เป็นกลาง
    "ปกติ": "neutral", "ธรรมดา": "neutral", "เฉย": "neutral"
}

# reverse mapping อังกฤษ-ไทย
ENG_TO_THAI = {
    "sad": "เศร้า",
    "lonely": "เหงา",
    "hope": "หวัง",
    "happy": "สุข",
    "excited": "ตื่นเต้น",
    "calm": "สงบ",
    "angry": "โกรธ",
    "neutral": "เฉย"
}

CANDIDATE_LABELS = ["sad", "lonely", "hope", "happy", "excited", "calm", "angry", "neutral"]

# ใช้โมเดลที่ stable
ZS_MODEL = "facebook/bart-large-mnli"
_zs = pipeline("zero-shot-classification", model=ZS_MODEL)

def _lexicon_fallback(text: str) -> str:
    """ค้นหาอารมณ์จาก lexicon ถ้าไม่เจอใช้ neutral"""
    # แยกคำด้วย PyThaiNLP
    for w in word_tokenize(text):
        # ลองหาใน lexicon ไทย-อังกฤษ
        if w in THAI_TO_ENG:
            eng_emotion = THAI_TO_ENG[w]
            return ENG_TO_THAI.get(eng_emotion, "เฉย")  # แปลงกลับเป็นไทย
    return "เฉย"  # default เป็นเฉย

def detect_emotion(text: str, threshold: float = 0.55, multi_label: bool = False) -> str:
    """วิเคราะห์อารมณ์จากข้อความ ถ้าไม่มั่นใจจะใช้ lexicon fallback"""
    if not text.strip():
        return "เฉย"
    try:
        res = _zs(text, candidate_labels=CANDIDATE_LABELS, multi_label=multi_label)
        if multi_label:
            picked = [lbl for lbl, sc in zip(res["labels"], res["scores"]) if sc >= threshold]
            if picked:
                return ENG_TO_THAI.get(picked[0], "เฉย")
        else:
            lbl, sc = res["labels"][0], res["scores"][0]
            if sc >= threshold:
                return ENG_TO_THAI.get(lbl, "เฉย")
        return _lexicon_fallback(text)
    except Exception:
        return _lexicon_fallback(text)
