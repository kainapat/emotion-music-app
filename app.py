import re  # ⬅ เพิ่ม
from collections import Counter
from flask import Flask, render_template, request, jsonify
import sqlite3
from pythainlp.tokenize import word_tokenize  # เพิ่ม import
from youtube_utils import fetch_youtube_metadata, extract_video_id
from nlp_utils import preprocess_lyrics
from emotion_model import detect_emotion
from analysis import build_trajectory, plot_interactive_trajectory

# === Thai Emotion Aliases → Canonical Labels ===
TH_EMO_ALIASES = {
    "เศร้า": {"เศร้า", "เสียใจ", "หม่น", "หมอง", "หดหู่", "ซึม", "ร้องไห้", "เหงา", "ทุกข์", "น้อยใจ", "ผิดหวัง"},
    "หวัง": {"หวัง", "ความหวัง", "มีความหวัง", "เริ่มหวัง", "ฝัน", "กำลังใจ", "สู้", "พยายาม"},
    "สุข": {"สุข", "มีความสุข", "ร่าเริง", "สดใส", "สนุก", "แฮปปี้", "ยิ้ม", "ดีใจ", "เบิกบาน", "ชื่นใจ"},
    "สงบ": {"สงบ", "นิ่ง", "ใจเย็น", "เย็น", "ผ่อนคลาย", "ชิล", "สบาย", "พักผ่อน", "สงัด"},
    "โกรธ": {"โกรธ", "โมโห", "เดือด", "เกรี้ยวกราด", "แค้น", "เคือง", "ฉุน", "เดือดดาล"},
    "ตื่นเต้น": {"ตื่นเต้น", "เร้าใจ", "พีค", "เข้มข้น", "ฮึกเหิม", "เร่งเร้า", "มัน", "สะใจ", "เปรี้ยว"},
    "กังวล": {"กังวล", "เครียด", "กลัว", "หวาดกลัว", "ประหม่า", "ลังเล", "ไม่แน่ใจ", "กระวนกระวาย"},
}

# reverse map: alias -> canonical
ALIAS2CANON = {}
for canon, aliases in TH_EMO_ALIASES.items():
    for a in aliases:
        ALIAS2CANON[a] = canon
    ALIAS2CANON[canon] = canon  # รวมตัวมันเอง

def _canonize(label: str) -> str:
    """แปะ label ให้เป็น canonical (ไทย) จากผลโมเดล/ข้อความ"""
    if not label:
        return ""
    t = re.sub(r"\s+", "", label.strip())
    
    # ตรวจสอบกรณีที่ label เป็นภาษาอังกฤษตรงๆ
    from emotion_model import ENG_TO_THAI
    if t.lower() in ENG_TO_THAI:
        return ENG_TO_THAI[t.lower()]
    
    # หาแบบ contains เพื่อครอบคลุมคำขยาย เช่น 'มีความสุขมาก'
    for alias, canon in ALIAS2CANON.items():
        if alias in t:
            return canon
    
    return t if t else ""

def _extract_emotion_keywords(text: str) -> list:
    """
    แยกคำสำคัญที่เกี่ยวกับอารมณ์จากข้อความ
    """
    # คำที่บ่งบอกการเปลี่ยนแปลง
    transition_words = {
        "เริ่ม": "start",
        "ค่อยๆ": "gradual",
        "พุ่ง": "sudden",
        "เปลี่ยน": "change",
        "กลาย": "change",
        "แล้ว": "then",
        "จาก": "from",
        "เป็น": "to",
        "ก่อน": "before",
    }
    
    # คำที่บ่งบอกระดับความเข้ม
    intensity_words = {
        "มาก": "high",
        "เบาๆ": "low",
        "ค่อยๆ": "gradual",
        "พุ่ง": "spike",
        "ขึ้น": "up",
        "ลง": "down"
    }
    
    keywords = []
    tokens = word_tokenize(text)  # ต้อง import word_tokenize จาก pythainlp
    
    for i, token in enumerate(tokens):
        # ตรวจหาคำเกี่ยวกับอารมณ์
        emotion = _canonize(token)
        if emotion:
            # ดูคำรอบๆ เพื่อหาความเข้มและการเปลี่ยนแปลง
            context = tokens[max(0, i-2):i+3]
            intensity = next((intensity_words[w] for w in context if w in intensity_words), "normal")
            transition = next((transition_words[w] for w in context if w in transition_words), None)
            
            keywords.append({
                "emotion": emotion,
                "intensity": intensity,
                "transition": transition,
                "position": i
            })
    
    return keywords

def parse_thai_emotion_query(q: str):
    """
    รับ query ภาษาธรรมชาติ → ลำดับอารมณ์พร้อมข้อมูลเพิ่มเติม
    รองรับ: 
    1. แบบลูกศร "neutral → excited → sad"
    2. แบบคงที่ "เพลงที่อารมณ์ neutral ตลอดทั้งเพลง"
    3. แบบธรรมชาติ "เพลงที่เริ่มเศร้าแล้วค่อยๆเปลี่ยนเป็นหวัง"
    4. แบบซับซ้อน "เพลงที่โทนใจเย็นก่อนแล้วพุ่งขึ้นมาเปล่งประกาย"
    """
    if not q:
        return []

    # ตรวจสอบรูปแบบลูกศร
    if "→" in q or "->" in q:
        q = q.replace("->", "→")
        parts = [p.strip() for p in q.split("→") if p.strip()]
        return [_canonize(p) for p in parts]

    # ตรวจสอบรูปแบบอารมณ์คงที่ก่อน
    constant_patterns = [
        r"(คงที่|ไม่เปลี่ยนแปลง|ตลอดทั้งเพลง|throughout|consistent|stable)",
        r"(เหมือนเดิม|ทั้งเพลง|all the way|same emotion)"
    ]
    
    for pattern in constant_patterns:
        if re.search(pattern, q, re.IGNORECASE):
            # ค้นหาอารมณ์เดียวที่กล่าวถึง
            tokens = word_tokenize(q)
            for token in tokens:
                emotion = _canonize(token)
                # ตรวจสอบว่าเป็นอารมณ์จริงๆ ไม่ใช่คำทั่วไป
                if emotion and emotion != " " and emotion != "" and emotion not in ["เพลง", "ที่", "อารมณ์", "หา", "มี", "ตลอด", "ทั้ง", "ไม่", "เปลี่ยนแปลง", "คงที่"]:
                    # ส่งคืนอารมณ์เดียวกัน 3 ครั้งเพื่อแสดงความต่อเนื่อง
                    return [emotion] * 3
    
    # ตรวจสอบกรณีที่ query เป็นแค่ชื่ออารมณ์ภาษาอังกฤษ
    if q.strip().lower() in ["neutral", "sad", "happy", "excited", "calm", "angry", "lonely", "hope"]:
        return [q.strip().lower()]
    
    # ตรวจสอบกรณีที่ query มีชื่ออารมณ์ภาษาอังกฤษอยู่ (ก่อนลบคำทั่วไป)
    for emotion in ["neutral", "sad", "happy", "excited", "calm", "angry", "lonely", "hope"]:
        if emotion in q.lower():
            return [emotion]
    
    # Enhanced natural language processing for complex queries
    return _parse_complex_emotion_query(q)

def _parse_complex_emotion_query(q: str):
    """
    ประมวลผล query ที่ซับซ้อนด้วยการวิเคราะห์ภาษาธรรมชาติแบบลึก
    ตัวอย่าง: "ขอเพลงที่เริ่มเศร้าแล้วค่อยๆ เปลี่ยนเป็นหวัง"
             "เพลงที่โทนใจเย็นก่อนแล้วพุ่งขึ้นมาเปล่งประกาย"
    """
    # ลบคำทั่วไปที่ไม่จำเป็น
    q = re.sub(r"เพลงที่|ขอเพลง|แนว|โทน|อารมณ์|ช่วง|looking for|a|consistently|song|หา|มี", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()
    
    # ตรวจสอบกรณีที่ query เป็นแค่ชื่ออารมณ์เดียว
    single_emotion = _canonize(q.strip())
    if single_emotion and single_emotion != "" and single_emotion not in ["เพลง", "ที่", "อารมณ์"]:
        from emotion_model import THAI_TO_ENG
        if single_emotion in THAI_TO_ENG:
            return [THAI_TO_ENG[single_emotion]]
        return [single_emotion]
    
    # แยกคำด้วย PyThaiNLP
    tokens = word_tokenize(q)
    
    # Enhanced pattern dictionary for complex emotional progressions
    transition_patterns = {
        "start": ["เริ่ม", "ตอนแรก", "แรกๆ", "ช่วงแรก", "ก่อน", "starts", "begins", "initially"],
        "gradual_change": ["ค่อยๆ", "ค่อย", "ค่อยเป็นค่อยไป", "ช้าๆ", "gradually", "slowly", "gently"],
        "sudden_change": ["พุ่ง", "กระแส", "ฉับพลัน", "ทันที", "เดี๋ยวเดียว", "suddenly", "quickly", "spikes", "bursts"],
        "transition": ["แล้ว", "จากนั้น", "ต่อมา", "เปลี่ยน", "กลาย", "then", "becomes", "changes", "transforms"],
        "uplifting": ["ขึ้น", "สู่", "เปล่งประกาย", "สดใส", "โปร่ง", "bright", "uplifting", "rising", "soaring"],
        "end": ["สุดท้าย", "ตอนจบ", "ท้ายเพลง", "จบ", "ends", "finally", "eventually"]
    }
    
    # Advanced synonym mapping for emotions
    emotion_synonyms = {
        "เศร้า": ["เศร้า", "เศร้าโศก", "โศกเศร้า", "หม่น", "หมอง", "หดหู่"],
        "หวัง": ["หวัง", "ความหวัง", "มีความหวัง", "เริ่มหวัง", "หวังใจ"],
        "สงบ": ["สงบ", "ใจเย็น", "เย็น", "โทนใจเย็น", "ผ่อนคลาย", "ชิล", "สบาย"],
        "ตื่นเต้น": ["ตื่นเต้น", "เร้าใจ", "เปล่งประกาย", "ประกาย", "พีค", "เข้มข้น", "มัน"],
        "สุข": ["สุข", "มีความสุข", "ร่าเริง", "สดใส", "สนุก", "ยิ้ม", "ดีใจ"],
        "โกรธ": ["โกรธ", "โมโห", "เดือด", "แค้น", "เคือง"],
        "เหงา": ["เหงา", "หงอย", "เศร้าเหงา", "โดดเดี่ยว"],
        "กลาง": ["กลาง", "เฉย", "ปกติ", "ธรรมดา"]
    }
    
    emotions_found = []
    transition_type = None
    
    # สแกนหาอารมณ์และรูปแบบการเปลี่ยนแปลง
    for i, token in enumerate(tokens):
        # ตรวจหารูปแบบการเปลี่ยนแปลง
        for pattern_type, words in transition_patterns.items():
            if any(word in token.lower() or token.lower() in word for word in words):
                transition_type = pattern_type
                break
        
        # ตรวจหาอารมณ์จากคำและ synonyms
        emotion = _canonize(token)
        if emotion and emotion != " ":
            emotions_found.append(emotion)
            continue
            
        # ตรวจหา synonyms
        for canonical_emotion, synonyms in emotion_synonyms.items():
            if any(syn in token.lower() or token.lower() in syn for syn in synonyms):
                emotions_found.append(canonical_emotion)
                break
    
    # สร้างลำดับอารมณ์ตามรูปแบบที่พบ
    if len(emotions_found) >= 2:
        # มีอารมณ์อย่างน้อย 2 อารมณ์ → ส่งคืนลำดับ
        return emotions_found[:3]  # จำกัดไว้ 3 อารมณ์
    elif len(emotions_found) == 1:
        # มีอารมณ์เดียว แต่มีการบ่งบอกการเปลี่ยนแปลง
        base_emotion = emotions_found[0]
        
        if transition_type in ["gradual_change", "sudden_change", "uplifting"]:
            # ถ้ามีการบ่งบอกการเปลี่ยนแปลง ให้เดาอารมณ์ที่เป็นไปได้
            if base_emotion == "เศร้า":
                return ["เศร้า", "หวัง"]  # เศร้า → หวัง
            elif base_emotion == "สงบ":
                return ["สงบ", "ตื่นเต้น"]  # สงบ → ตื่นเต้น
            elif base_emotion == "ตื่นเต้น":
                return ["สงบ", "ตื่นเต้น"]  # สงบ → ตื่นเต้น
        
        # กรณีอื่นๆ ส่งคืนอารมณ์เดียว
        from emotion_model import THAI_TO_ENG
        if base_emotion in THAI_TO_ENG:
            return [THAI_TO_ENG[base_emotion]]
        return [base_emotion]
    
    # ถ้าไม่พบอารมณ์ชัดเจน ลองหาจากคำทั้งหมด
    all_emotions = [_canonize(t) for t in tokens if _canonize(t) and _canonize(t) != " "]
    return [e for e in all_emotions if e and e != " "][:2]  # จำกัดไว้ 2 อารมณ์

def soft_subseq_match(target, seq):
    """
    soft subsequence matching with support for:
    1. Regular sequence matching: ['sad','happy'] matches ['sad','neutral','happy']
    2. Constant emotion: ['neutral','neutral','neutral'] matches ['neutral'] * N
    3. Mixed language support: 'sad' matches 'เศร้า'
    """
    if not target:
        return False
        
    # ทำให้เป็นภาษาเดียวกัน (อังกฤษ) เพื่อให้ตรงกับฐานข้อมูล
    from emotion_model import THAI_TO_ENG, ENG_TO_THAI
    
    def normalize_emotion(e):
        # ถ้าเป็นภาษาไทย แปลงเป็นอังกฤษ
        if e in THAI_TO_ENG:
            return THAI_TO_ENG[e]
        # ถ้าเป็นภาษาอังกฤษอยู่แล้ว ให้เป็น lowercase
        return e.lower() if e else e
    
    # แปลงทั้งสองฝั่งให้เป็นภาษาอังกฤษ (ตรงกับฐานข้อมูล)
    target = [normalize_emotion(t) for t in target]
    seq = [normalize_emotion(s) for s in seq]
    
    # กรณีอารมณ์คงที่: ถ้า target มีอารมณ์เดียวซ้ำกัน
    if len(set(target)) == 1:
        target_emotion = target[0]
        # ต้องพบอารมณ์นั้นอย่างน้อย 1 ครั้ง (สำหรับการค้นหาอารมณ์เดียว)
        emotion_count = sum(1 for s in seq if s == target_emotion)
        return emotion_count >= 1
    
    # กรณีปกติ: ค้นหาลำดับอารมณ์
    i = 0
    for s in seq:
        if i < len(target) and s == target[i]:
            i += 1
    return i == len(target)

def calculate_overall_emotion(emotions):
    """
    คำนวณอารมณ์โดยรวมของเพลงจากรายการอารมณ์
    """
    if not emotions:
        return "unknown"
    
    # นับความถี่ของแต่ละอารมณ์
    emotion_counts = {}
    for emotion in emotions:
        emotion = emotion.lower() if emotion else "unknown"
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    # หาอารมณ์ที่มีความถี่สูงสุด
    most_common_emotion = max(emotion_counts.items(), key=lambda x: x[1])
    
    # ถ้าอารมณ์ที่พบบ่อยที่สุดมีสัดส่วนมากกว่า 50% ให้ใช้อารมณ์นั้น
    total_emotions = len(emotions)
    if most_common_emotion[1] / total_emotions > 0.5:
        return most_common_emotion[0]
    
    # ถ้าไม่มีความชัดเจน ให้วิเคราะห์จากลำดับอารมณ์
    # หาอารมณ์ที่ปรากฏในส่วนท้ายของเพลง (มีน้ำหนักมากกว่า)
    if len(emotions) >= 3:
        # ใช้ 30% สุดท้ายของเพลง
        end_portion = emotions[-max(1, len(emotions)//3):]
        end_emotion_counts = {}
        for emotion in end_portion:
            emotion = emotion.lower() if emotion else "unknown"
            end_emotion_counts[emotion] = end_emotion_counts.get(emotion, 0) + 1
        
        if end_emotion_counts:
            return max(end_emotion_counts.items(), key=lambda x: x[1])[0]
    
    return most_common_emotion[0]

def get_emotion_color(emotion):
    """
    กำหนดสีสำหรับแต่ละอารมณ์
    """
    emotion_colors = {
        'sad': 'bg-blue-100 text-blue-800 border-blue-200',
        'lonely': 'bg-purple-100 text-purple-800 border-purple-200',
        'hope': 'bg-green-100 text-green-800 border-green-200',
        'happy': 'bg-yellow-100 text-yellow-800 border-yellow-200',
        'excited': 'bg-red-100 text-red-800 border-red-200',
        'calm': 'bg-indigo-100 text-indigo-800 border-indigo-200',
        'angry': 'bg-orange-100 text-orange-800 border-orange-200',
        'neutral': 'bg-gray-100 text-gray-600 border-gray-300',
        'unknown': 'bg-gray-100 text-gray-500 border-gray-300'
    }
    return emotion_colors.get(emotion.lower(), 'bg-gray-100 text-gray-500 border-gray-300')

def get_emotion_icon(emotion):
    """
    กำหนดไอคอนสำหรับแต่ละอารมณ์ (ไม่ใช้อีโมจิ)
    """
    emotion_icons = {
        'sad': '💙',
        'lonely': '💜',
        'hope': '💚',
        'happy': '💛',
        'excited': '❤️',
        'calm': '🔵',
        'angry': '🧡',
        'neutral': '⚪',
        'unknown': '❓'
    }
    return emotion_icons.get(emotion.lower(), '❓')

def get_emotion_explanation(emotion, emotions_list):
    """
    อธิบายว่าทำไมเพลงถึงมีอารมณ์โดยรวมแบบนั้น
    """
    if not emotions_list:
        return "ไม่สามารถวิเคราะห์อารมณ์ได้"
    
    emotion_counts = {}
    for e in emotions_list:
        e = e.lower() if e else "unknown"
        emotion_counts[e] = emotion_counts.get(e, 0) + 1
    
    total_segments = len(emotions_list)
    main_emotion = emotion.lower()
    
    # คำนวณเปอร์เซ็นต์ของอารมณ์หลัก
    main_percentage = (emotion_counts.get(main_emotion, 0) / total_segments) * 100
    
    # หาอารมณ์รอง
    sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
    secondary_emotions = [e for e, count in sorted_emotions[1:3] if count > 0]
    
    # สร้างคำอธิบาย
    explanations = {
        'sad': f"เพลงนี้มีอารมณ์เศร้าเป็นหลัก ({main_percentage:.1f}% ของเพลง) เนื่องจากเนื้อเพลงส่วนใหญ่แสดงถึงความเศร้า ความเสียใจ หรือความหดหู่",
        'lonely': f"เพลงนี้มีอารมณ์เหงาเป็นหลัก ({main_percentage:.1f}% ของเพลง) เนื่องจากเนื้อเพลงส่วนใหญ่แสดงถึงความโดดเดี่ยว ความว้าเหว่ หรือความรู้สึกเหงา",
        'hope': f"เพลงนี้มีอารมณ์หวังเป็นหลัก ({main_percentage:.1f}% ของเพลง) เนื่องจากเนื้อเพลงส่วนใหญ่แสดงถึงความหวัง การให้กำลังใจ หรือการมองโลกในแง่ดี",
        'happy': f"เพลงนี้มีอารมณ์สุขเป็นหลัก ({main_percentage:.1f}% ของเพลง) เนื่องจากเนื้อเพลงส่วนใหญ่แสดงถึงความสุข ความร่าเริง หรือความสนุกสนาน",
        'excited': f"เพลงนี้มีอารมณ์ตื่นเต้นเป็นหลัก ({main_percentage:.1f}% ของเพลง) เนื่องจากเนื้อเพลงส่วนใหญ่แสดงถึงความตื่นเต้น ความเร้าใจ หรือความเข้มข้น",
        'calm': f"เพลงนี้มีอารมณ์สงบเป็นหลัก ({main_percentage:.1f}% ของเพลง) เนื่องจากเนื้อเพลงส่วนใหญ่แสดงถึงความสงบ ความเยือกเย็น หรือความผ่อนคลาย",
        'angry': f"เพลงนี้มีอารมณ์โกรธเป็นหลัก ({main_percentage:.1f}% ของเพลง) เนื่องจากเนื้อเพลงส่วนใหญ่แสดงถึงความโกรธ ความโมโห หรือความไม่พอใจ",
        'neutral': f"เพลงนี้มีอารมณ์เฉยเป็นหลัก ({main_percentage:.1f}% ของเพลง) เนื่องจากเนื้อเพลงส่วนใหญ่แสดงถึงอารมณ์ที่เป็นกลาง ไม่เด่นชัดไปทางใดทางหนึ่ง"
    }
    
    base_explanation = explanations.get(main_emotion, f"เพลงนี้มีอารมณ์{main_emotion}เป็นหลัก ({main_percentage:.1f}% ของเพลง)")
    
    # เพิ่มข้อมูลอารมณ์รองถ้ามี
    if secondary_emotions:
        secondary_text = " และ ".join(secondary_emotions[:2])
        base_explanation += f" นอกจากนี้ยังมีอารมณ์{secondary_text}ผสมอยู่ด้วย"
    
    return base_explanation

app = Flask(__name__)

def db_query(query, args=(), fetch=False):
    conn = None
    try:
        conn = sqlite3.connect("songs.db")
        cur = conn.cursor()
        cur.execute(query, args)
        rows = cur.fetchall() if fetch else None
        conn.commit()
        return rows
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

# ----------------------
# Index (เพิ่มเพลง / วิเคราะห์)
# ----------------------
@app.route("/", methods=["GET","POST"])
def index():
    error = None
    if request.method == "POST":
        yt_link = request.form["youtube"]
        lyrics = request.form["lyrics"]

        video_id = extract_video_id(yt_link)
        meta = fetch_youtube_metadata(video_id)

        if meta:
            # ----------------------------
            # เช็คว่ามีเพลงนี้แล้วหรือยัง (ใช้ youtube_link)
            # ----------------------------
            existing = db_query("SELECT id FROM songs WHERE youtube_link=?", (yt_link,), fetch=True)
            if existing:
                songs = db_query("SELECT id,title,view_count,like_count,upload_date,graph_html FROM songs", fetch=True)
                return render_template("index.html", songs=songs, error="⚠️ เพลงนี้ถูกเพิ่มแล้ว ไม่สามารถเพิ่มซ้ำได้")

            try:
                # สร้าง connection ใหม่สำหรับ transaction
                conn = sqlite3.connect("songs.db")
                cur = conn.cursor()
                
                try:
                    # เริ่ม transaction
                    cur.execute("BEGIN TRANSACTION")
                    
                    # ----------------------------
                    # ถ้าไม่ซ้ำ → insert
                    # ----------------------------
                    cur.execute("""
                        INSERT INTO songs (title,youtube_link,description,tags,upload_date,view_count,like_count,lyrics)
                        VALUES (?,?,?,?,?,?,?,?)
                    """, (
                        meta.get("title"), yt_link, meta.get("description",""),
                        ",".join(meta.get("tags",[])), meta.get("upload_date",""),
                        meta.get("view_count",0), meta.get("like_count",0), lyrics
                    ))
                    
                    song_id = cur.lastrowid

                    # ตัด segment + emotion detection
                    segments = preprocess_lyrics(lyrics)
                    emotions = []
                    for i, seg in enumerate(segments):
                        e = detect_emotion(seg)
                        emotions.append(e)
                        cur.execute("""
                            INSERT INTO segments (song_id,segment_order,text,emotion) 
                            VALUES (?,?,?,?)""", (song_id, i, seg, e))

                    # สร้าง interactive graph
                    trajectory_html = plot_interactive_trajectory(emotions, meta.get("title"))
                    cur.execute("UPDATE songs SET graph_html=? WHERE id=?", 
                              (trajectory_html, song_id))
                    
                    # commit transaction
                    conn.commit()
                    
                except Exception as e:
                    # ถ้าเกิดข้อผิดพลาด rollback
                    conn.rollback()
                    raise e
                finally:
                    # ปิด connection
                    conn.close()
            except Exception as e:
                # ถ้าเกิดข้อผิดพลาดระหว่างการวิเคราะห์ ให้ลบข้อมูลเพลงทิ้ง
                db_query("DELETE FROM segments WHERE song_id=?", (song_id,))
                db_query("DELETE FROM songs WHERE id=?", (song_id,))
                return render_template("index.html", songs=songs, error=f"⚠️ เกิดข้อผิดพลาดในการวิเคราะห์: {str(e)}")

    songs_data = db_query("SELECT id,title,view_count,like_count,upload_date,graph_html FROM songs", fetch=True)
    
    # เพิ่มข้อมูลอารมณ์โดยรวมให้กับแต่ละเพลง
    enhanced_songs = []
    for song in songs_data:
        song_id = song[0]
        # ดึงอารมณ์ของเพลงนี้
        emotions = db_query("SELECT emotion FROM segments WHERE song_id=? ORDER BY segment_order", (song_id,), fetch=True)
        emotion_list = [emotion[0] for emotion in emotions if emotion[0]]
        overall_emotion = calculate_overall_emotion(emotion_list)
        
        enhanced_songs.append({
            "id": song[0],
            "title": song[1],
            "views": song[2],
            "likes": song[3],
            "upload": song[4],
            "graph": song[5],
            "overall_emotion": overall_emotion,
            "emotion_color": get_emotion_color(overall_emotion),
            "emotion_icon": get_emotion_icon(overall_emotion)
        })
    
    return render_template("index.html", songs=enhanced_songs, error=error)

# ----------------------
# Search (ค้นหาเพลง)
# ----------------------
@app.route("/search", methods=["GET","POST"])
def search():
    songs = []
    q_tokens = []
    if request.method == "POST":
        raw = request.form.get("query", "")
        q_tokens = parse_thai_emotion_query(raw)  # ⬅ แปลงข้อความไทยเป็นลิสต์อารมณ์

        # ดึงเพลงทั้งหมด + segments
        all_songs = db_query("SELECT id,title,view_count,like_count,upload_date,graph_html FROM songs", fetch=True)
        for s in all_songs:
            segs = db_query("SELECT emotion FROM segments WHERE song_id=? ORDER BY segment_order", (s[0],), fetch=True)
            # ทำ canonical เช่นเดียวกับ query
            song_seq = [_canonize(x[0]) for x in segs]

            # ตรงตามลำดับแบบ soft-subsequence ก็ถือว่า match
            if soft_subseq_match(q_tokens, song_seq):
                songs.append(s)

    return render_template("search.html", songs=songs, q_tokens=" → ".join(q_tokens))

# ----------------------
# Song Detail (ดูรายละเอียดเพลง)
# ----------------------
@app.route("/song/<int:song_id>")
def song_detail(song_id):
    # id,title,youtube_link,upload_date,view_count,like_count,graph_html,lyrics
    song = db_query("""SELECT id,title,youtube_link,upload_date,view_count,like_count,graph_html,lyrics
                       FROM songs WHERE id=?""", (song_id,), fetch=True)
    if not song:
        return "ไม่พบเพลงนี้", 404
    song = song[0]

    # segments: (order, text, emotion)
    segments = db_query("""SELECT segment_order,text,emotion
                           FROM segments WHERE song_id=? ORDER BY segment_order""",
                        (song_id,), fetch=True)

    # คำนวณอารมณ์โดยรวม
    emotions = [seg[2] for seg in segments if seg[2]]  # ดึงอารมณ์จาก segments
    overall_emotion = calculate_overall_emotion(emotions)
    emotion_explanation = get_emotion_explanation(overall_emotion, emotions)
    
    # เพิ่มข้อมูลสีและไอคอนให้กับ segments
    enhanced_segments = []
    for seg in segments:
        emotion = seg[2] if seg[2] else "unknown"
        enhanced_segments.append({
            "order": seg[0],
            "text": seg[1],
            "emotion": emotion,
            "color_class": get_emotion_color(emotion),
            "icon": get_emotion_icon(emotion)
        })

    # ❌ ไม่ auto-refresh แล้ว → ใช้ค่าที่ DB มีเท่านั้น
    graph_html = song[6]

    return render_template(
        "song_detail.html",
        song={
            "id": song[0],
            "title": song[1],
            "youtube": song[2],
            "upload": song[3],
            "views": song[4],
            "likes": song[5],
            "graph": graph_html
        },
        segments=enhanced_segments,
        overall_emotion=overall_emotion,
        overall_emotion_color=get_emotion_color(overall_emotion),
        overall_emotion_icon=get_emotion_icon(overall_emotion),
        emotion_explanation=emotion_explanation
    )

# ----------------------
# เพิ่ม route refresh (รีเฟรชข้อมูลเพลง)
# ----------------------
@app.route("/song/<int:song_id>/refresh")
def refresh_song(song_id):
    # ดึงข้อมูลเพลง
    song = db_query("SELECT id, title, lyrics FROM songs WHERE id=?", (song_id,), fetch=True)
    if not song:
        return "ไม่พบเพลงนี้", 404
    sid, title, lyrics = song[0]

    # ลบ segments เดิม
    db_query("DELETE FROM segments WHERE song_id=?", (sid,))

    # ประมวลผลใหม่
    segments = preprocess_lyrics(lyrics or "")
    emotions = []
    for i, seg in enumerate(segments):
        e = detect_emotion(seg)
        emotions.append(e)
        db_query("INSERT INTO segments (song_id,segment_order,text,emotion) VALUES (?,?,?,?)",
                 (sid, i, seg, e))

    # สร้างกราฟใหม่
    trajectory_html = plot_interactive_trajectory(emotions, title)
    db_query("UPDATE songs SET graph_html=? WHERE id=?", (trajectory_html, sid))

    # redirect กลับไปหน้า song_detail
    from flask import redirect, url_for
    return redirect(url_for("song_detail", song_id=sid))

# ----------------------
# Rebuild (บังคับสร้างใหม่)
# ----------------------
@app.route("/song/<int:song_id>/rebuild", methods=["POST"])
def rebuild_song(song_id):
    song = db_query("SELECT id,title,lyrics FROM songs WHERE id=?", (song_id,), fetch=True)
    if not song:
        return "ไม่พบเพลงนี้", 404
    song = song[0]

    # ลบของเดิม
    db_query("DELETE FROM segments WHERE song_id=?", (song_id,))
    db_query("UPDATE songs SET graph_html=NULL WHERE id=?", (song_id,))

    if song[2]:
        raw_segments = preprocess_lyrics(song[2])
        emotions = []
        for i, seg in enumerate(raw_segments):
            e = detect_emotion(seg)
            emotions.append(e)
            db_query("INSERT INTO segments (song_id,segment_order,text,emotion) VALUES (?,?,?,?)",
                     (song_id, i, seg, e))
        if emotions:
            graph_html = plot_interactive_trajectory(emotions, song[1])
            db_query("UPDATE songs SET graph_html=? WHERE id=?", (graph_html, song_id))

    return ("", 204)  # กลับไปที่หน้าเดิม (จะใช้ JS reload)

# ----------------------
# Delete Song (ลบเพลง)
# ----------------------
@app.route("/song/<int:song_id>/delete", methods=["POST"])
def delete_song(song_id):
    # ลบ segments ก่อน
    db_query("DELETE FROM segments WHERE song_id=?", (song_id,))
    # ลบเพลง
    db_query("DELETE FROM songs WHERE id=?", (song_id,))
    from flask import redirect, url_for
    return redirect(url_for("index"))

# ----------------------
# Explore (สำรวจเพลง)
# ----------------------
# ถ้ามี _canonize อยู่แล้ว (จากส่วนค้นหา) จะช่วย normalize label ให้แม่นขึ้น
try:
    _canonize  # noqa
except NameError:
    def _canonize(x): return (x or "").strip()

@app.route("/explore")
def explore():
    # อารมณ์ยอดนิยม (จริง)
    top_emotions = db_query("""
        SELECT emotion, COUNT(*) as cnt
        FROM segments
        GROUP BY emotion
        ORDER BY cnt DESC
        LIMIT 6
    """, fetch=True)  # [(emotion, count), ...]

    # เตรียมข้อมูลเพลงทั้งหมด
    songs = db_query("SELECT id, title FROM songs", fetch=True)

    transition_rows = []  # [{id,title, path, trans_cnt}]
    stable_rows = []      # [{id,title, emotion}]

    for sid, title in songs:
        segs = db_query(
            "SELECT emotion FROM segments WHERE song_id=? ORDER BY segment_order",
            (sid,), fetch=True
        )
        seq = [_canonize(e[0]) for e in segs if e and e[0]]

        if not seq:
            continue

        # บีบอัดอารมณ์ซ้ำติดกัน (เช่น 'สุข,สุข,สุข' -> 'สุข')
        compressed = []
        for e in seq:
            if not compressed or compressed[-1] != e:
                compressed.append(e)

        path = " → ".join(compressed)
        trans_cnt = max(len(compressed) - 1, 0)

        # เพลงคงที่ = มีอารมณ์เดียวในทั้งเพลง (หลังบีบอัดเหลือ 1)
        if len(set(compressed)) == 1:
            stable_rows.append({"id": sid, "title": title, "emotion": compressed[0]})

        transition_rows.append({
            "id": sid, "title": title, "path": path, "trans_cnt": trans_cnt
        })

    # จัดอันดับ transition เด่นสุด (มาก→น้อย)
    transition_rows.sort(key=lambda x: x["trans_cnt"], reverse=True)
    top_transition = transition_rows[:8]  # ปรับจำนวนได้

    # เอาเพลงคงที่ 8 รายการพอ
    stable_songs = stable_rows[:8]

    return render_template(
        "explore.html",
        top_emotions=top_emotions,
        top_transition=top_transition,
        stable_songs=stable_songs
    )

# ----------------------
# Dashboard (แดชบอร์ด)
# ----------------------
@app.route("/dashboard")
def dashboard():
    total_songs = db_query("SELECT COUNT(*) FROM songs", fetch=True)[0][0]
    total_segments = db_query("SELECT COUNT(*) FROM segments", fetch=True)[0][0]

    # อารมณ์ยอดนิยม
    popular_emotion = db_query("""
        SELECT emotion, COUNT(*) as cnt
        FROM segments
        GROUP BY emotion
        ORDER BY cnt DESC
        LIMIT 1
    """, fetch=True)
    popular_emotion = popular_emotion[0][0] if popular_emotion else "N/A"

    # ค่าเฉลี่ยอารมณ์ = segments / songs
    avg_score = round(total_segments / max(total_songs,1), 2)

    # นับจำนวนอารมณ์ทั้งหมด
    emotion_stats = db_query("""
        SELECT emotion, COUNT(*) as cnt
        FROM segments
        GROUP BY emotion
        ORDER BY cnt DESC
    """, fetch=True)

    stats = {
        "total_songs": total_songs,
        "total_segments": total_segments,
        "avg_score": avg_score,
        "popular_emotion": popular_emotion,
        "emotion_stats": emotion_stats
    }

    return render_template("dashboard.html", stats=stats)

# ----------------------
# Tokenize API
# ----------------------
@app.route("/tokenize", methods=["POST"])
def tokenize_text():
    try:
        data = request.get_json()
        lyrics = data.get("lyrics", "")
        
        from nlp_utils import auto_tokenize
        tokenized = auto_tokenize(lyrics)
        
        return jsonify({
            "success": True,
            "tokenized_text": tokenized
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

# ----------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)