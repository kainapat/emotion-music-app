from transformers import pipeline
from pythainlp import word_tokenize

# Lexicon mapping ระหว่างไทย-อังกฤษ
# Lexicon mapping ระหว่างไทย-อังกฤษ (Expanded to 80+ words)
THAI_TO_ENG = {
    # เศร้า (Sad) - 15 words
    "เศร้า": "sad", "เสียใจ": "sad", "หม่น": "sad", "หมอง": "sad", 
    "หดหู่": "sad", "ซึม": "sad", "ร้องไห้": "sad", "น้ำตา": "sad",
    "ทุกข์": "sad", "น้อยใจ": "sad", "ผิดหวัง": "sad", "เจ็บปวด": "sad",
    "ทรมาน": "sad", "ช้ำ": "sad", "อาลัย": "sad",
    
    # เหงา (Lonely) - 12 words
    "เหงา": "lonely", "เดียวดาย": "lonely", "ว้าเหว่": "lonely",
    "โดดเดี่ยว": "lonely", "อ้างว้าง": "lonely", "เปล่าเปลี่ยว": "lonely",
    "ขาด": "lonely", "ห่างไกล": "lonely", "คิดถึง": "lonely", 
    "ลำพัง": "lonely", "คนเดียว": "lonely", "ว่างเปล่า": "lonely",
    
    # หวัง (Hope) - 15 words
    "หวัง": "hope", "ความหวัง": "hope", "มีหวัง": "hope", 
    "กำลังใจ": "hope", "สู้": "hope", "พยายาม": "hope",
    "ศรัทธา": "hope", "ฝัน": "hope", "เชื่อมั่น": "hope", "แสงสว่าง": "hope",
    "พรุ่งนี้": "hope", "สักวัน": "hope", "ภาวนา": "hope", "ขอให้": "hope", "อธิษฐาน": "hope",
    
    # สุข (Happy) - 12 words
    "สุข": "happy", "ยินดี": "happy", "ดีใจ": "happy", 
    "ร่าเริง": "happy", "สดใส": "happy", "สนุก": "happy",
    "แฮปปี้": "happy", "ยิ้ม": "happy", "เบิกบาน": "happy",
    "หัวเราะ": "happy", "สำราญ": "happy", "ปลื้ม": "happy",
    
    # ตื่นเต้น (Excited) - 10 words
    "เร้าใจ": "excited", "ตื่นเต้น": "excited", "พีค": "excited",
    "มัน": "excited", "เปรี้ยว": "excited", "ฮึกเหิม": "excited",
    "สุดยอด": "excited", "ระทึก": "excited", "ร้อนแรง": "excited", "กระหาย": "excited",
    
    # สงบ (Calm) - 10 words
    "สงบ": "calm", "เยือกเย็น": "calm", "นิ่ง": "calm",
    "ใจเย็น": "calm", "ผ่อนคลาย": "calm", "ชิล": "calm",
    "ร่มรื่น": "calm", "สบาย": "calm", "พักผ่อน": "calm", "เรียบง่าย": "calm",
    
    # โกรธ (Angry) - 8 words
    "โกรธ": "angry", "โมโห": "angry", "เดือด": "angry",
    "แค้น": "angry", "โกรธา": "angry", "เคือง": "angry",
    "เกลียด": "angry", "ด่า": "angry",
    
    # เป็นกลาง (Neutral) - 6 words
    "ปกติ": "neutral", "ธรรมดา": "neutral", "เฉย": "neutral",
    "เรื่อยๆ": "neutral", "ทั่วไป": "neutral", "กลางๆ": "neutral"
}

# คำบ่งชี้บริบทเชิงบวก/ลบ (Contextual Indicators)
POSITIVE_MARKERS = {"ดี", "สวย", "งาม", "รัก", "ชอบ", "ใช่", "เลิศ", "สุด"}
NEGATIVE_MARKERS = {"แย่", "เลว", "ไม่", "เกลียด", "เบื่อ", "เซ็ง", "เจ็บ", "ตาย"}

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
    tokens = word_tokenize(text)
    
    # 1. Direct Lexicon Match
    for w in tokens:
        if w in THAI_TO_ENG:
            return THAI_TO_ENG[w]
            
    # 2. Contextual Analysis Fallback
    pos_count = sum(1 for w in tokens if w in POSITIVE_MARKERS)
    neg_count = sum(1 for w in tokens if w in NEGATIVE_MARKERS)
    
    if pos_count > neg_count:
        return "happy" # Tend towards positive
    elif neg_count > pos_count:
        return "sad"   # Tend towards negative
        
    return "neutral"

def detect_emotion(text: str, threshold: float = 0.55, multi_label: bool = False) -> str:
    """วิเคราะห์อารมณ์จากข้อความ ถ้าไม่มั่นใจจะใช้ lexicon fallback"""
    if not text.strip():
        return "neutral" 
    
    try:
        # 1. Zero-shot Classification
        res = _zs(text, candidate_labels=CANDIDATE_LABELS, multi_label=multi_label)
        
        if multi_label:
            picked = [lbl for lbl, sc in zip(res["labels"], res["scores"]) if sc >= threshold]
            if picked:
                return picked[0]
        else:
            lbl, sc = res["labels"][0], res["scores"][0]
            if sc >= threshold:
                return lbl
        
        # 2. Smart Fallback (Lexicon + Context)
        # กรณีคะแนนไม่ถึง threshold หรือโมเดลไม่มั่นใจ ให้ใช้ Lexicon/Context
        lexicon_result = _lexicon_fallback(text)
        
        # ถ้า Lexicon เจออารมณ์ชัดเจน (ไม่ใช่ neutral) ให้ใช้อันนั้น
        if lexicon_result != "neutral":
            return lexicon_result
            
        # ถ้า Lexicon ก็ไม่เจอ ให้ return ผลจากโมเดลแม้คะแนนจะต่ำ (Best Effort)
        # แต่ถ้าคะแนนต่ำมากๆ (<0.3) ให้ fallback เป็น neutral จริงๆ เพื่อลด noise
        if not multi_label and res["scores"][0] < 0.35: # Hard floor for very low confidence
            return "neutral"
            
        return res["labels"][0] # Return best guess from model
        
    except Exception:
        return _lexicon_fallback(text)
