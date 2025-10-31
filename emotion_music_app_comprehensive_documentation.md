# คู่มือครอบคลุมโปรแกรม Emotion Music App

## บทนำ

โปรแกรม Emotion Music App เป็นแอปพลิเคชันเว็บที่ใช้เทคโนโลยี Machine Learning และ Natural Language Processing เพื่อวิเคราะห์เส้นทางอารมณ์ของเพลงอย่างอัจฉริยะ สามารถประมวลผลเนื้อเพลงและสร้างกราฟแสดงการเปลี่ยนแปลงอารมณ์แบบอินเตอร์แอคทีฟ

## ฟังก์ชันหลัก

### 1. ระบบวิเคราะห์อารมณ์เพลง

#### การตรวจจับอารมณ์
- **โมเดลหลัก**: ใช้โมเดล BART (facebook/bart-large-mnli) สำหรับการจำแนกอารมณ์แบบ zero-shot
- **8 ประเภทอารมณ์**:
  - เศร้า (sad) - รวม เสียใจ, หม่น, หมอง, หดหู่, ซึม, ร้องไห้, ทุกข์, น้อยใจ, ผิดหวัง
  - เหงา (lonely) - รวม เดียวดาย, ว้าเหว่
  - หวัง (hope) - รวม ความหวัง, มีหวัง, กำลังใจ, สู้, พยายาม
  - สุข (happy) - รวม ยินดี, ดีใจ, ร่าเริง, สดใส, สนุก, ยิ้ม, เบิกบาน
  - ตื่นเต้น (excited) - รวม เร้าใจ, พีค, มัน, เปรี้ยว, ฮึกเหิม
  - สงบ (calm) - รวม เยือกเย็น, นิ่ง, ใจเย็น, ผ่อนคลาย, ชิล
  - โกรธ (angry) - รวม โมโห, เดือด, แค้น, เคือง
  - เฉย (neutral) - รวม ปกติ, ธรรมดา

#### ระบบสำรอง
- **พจนานุกรมภาษาไทย**: ครอบคลุมกว่า 50 คำอารมณ์พร้อมการแมปไทย-อังกฤษสองทิศทาง
- **การแปลงอัตโนมัติ**: แปลงอารมณ์ภาษาไทยเป็นอังกฤษและกลับกัน
- **เกณฑ์ความมั่นใจ**: threshold = 0.55 สำหรับการจำแนก

#### การประมวลผลสองภาษา
- **รองรับเนื้อเพลง**: ภาษาไทย อังกฤษ และผสมกัน
- **การแบ่งคำอัตโนมัติ**: ใช้ PyThaiNLP สำหรับภาษาไทย และ NLTK สำหรับภาษาอังกฤษ

### 2. การประมวลผลและแบ่งส่วนเนื้อเพลง

#### การแบ่งส่วนอัจฉริยะ
- **รูปแบบไทย**: อินโทร, ท่อน, คอรัส, บริดจ์, เอาท์โทร
- **รูปแบบอังกฤษ**: intro, verse, chorus, bridge, outro
- **ระบบสำรอง**: แบ่งตามย่อหน้าและความยาว

#### การทำความสะอาดข้อมูล
```python
def _clean_text(s: str) -> str:
    s = re.sub(r'https?://\S+', ' ', s)     # ลบ URL
    s = re.sub(r'#[\wก-๙]+', ' ', s)        # ลบ hashtags
    s = re.sub(r'[^\S\r\n]+', ' ', s)       # ย่อช่องว่าง
    s = re.sub(r'[^\x00-\x7Fก-๙\r\n ]', ' ', s)  # ลบ emoji/อักขระพิเศษ
    return s.strip()
```

#### การแบ่งคำอัตโนมัติ
```python
def auto_tokenize(text: str) -> str:
    # แยกส่วนไทย-อังกฤษ
    parts = re.split(r'([A-Za-z]+(?:\s+[A-Za-z]+)*)', line)
    
    for part in parts:
        if re.match(r'^[A-Za-z\s]+$', part):
            tokens = nltk.word_tokenize(part)  # อังกฤษ
        else:
            tokens = thai_tokenize(part)  # ไทย
```

### 3. ระบบการค้นหาขั้นสูง

#### รูปแบบการค้นหาที่รองรับ
- **รูปแบบลูกศร**: "เศร้า → หวัง" หรือ "sad → hope"
- **ภาษาธรรมชาติภาษาไทย**: "เพลงที่เริ่มเศร้าแล้วค่อยๆเปลี่ยนเป็นหวัง"
- **ภาษาอังกฤษ**: "song that starts sad and becomes happy"
- **อารมณ์เดียว**: "เศร้า", "neutral", "สุข"
- **อารมณ์คงที่**: "เพลงที่อารมณ์ neutral ตลอดทั้งเพลง"

#### อัลกอริทึมการจับคู่
```python
def calculate_match_score(query_emotions, song_emotions):
    # ใช้ Longest Common Subsequence (LCS) algorithm
    n, m = len(query_emotions), len(song_emotions)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if query_emotions[i-1] == song_emotions[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[n][m]
    return min(lcs_length / len(query_emotions), 1.0)
```

#### การจับคู่แบบยืดหยุ่น
```python
def soft_subseq_match(target, seq):
    # การจับคู่แบบ soft subsequence
    i = 0
    for s in seq:
        if i < len(target) and s == target[i]:
            i += 1
    return i == len(target)
```

#### การค้นหาเชิงความหมาย
```python
# vectorstore.py
import faiss
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
index = faiss.IndexFlatL2(384)

def search_query(query, top_k=5):
    vec = embedder.encode([query])
    D, I = index.search(np.array(vec, dtype="float32"), top_k)
    results = [metadata[i] for i in I[0]]
    return results
```

### 4. การแสดงผลแบบอินเตอร์แอคทีฟ

#### กราฟ Plotly
```python
def plot_interactive_trajectory(emotions, song_name):
    df = pd.DataFrame({"step": range(len(emotions)), "emotion": emotions})
    fig = px.line(
        df, x="step", y="emotion",
        title=f"Emotion Trajectory: {song_name}",
        markers=True,
        labels={"step": "Step", "emotion": "Emotion"}
    )
    return fig.to_html(full_html=False)
```

#### ระบบสีและไอคอน
- **เศร้า (SAD)**: พื้นหลังสีฟ้า + ไอคอน 💙
- **เหงา (LONELY)**: พื้นหลังสีม่วง + ไอคอน 💜
- **หวัง (HOPE)**: พื้นหลังสีเขียว + ไอคอน 💚
- **สุข (HAPPY)**: พื้นหลังสีเหลือง + ไอคอน 💛
- **ตื่นเต้น (EXCITED)**: พื้นหลังสีแดง + ไอคอน ❤️
- **สงบ (CALM)**: พื้นหลังสีคราม + ไอคอน 🔵
- **โกรธ (ANGRY)**: พื้นหลังสีส้ม + ไอคอน 🧡
- **เฉย (NEUTRAL)**: พื้นหลังสีเทา + ไอคอน ⚪

#### การวิเคราะห์อารมณ์โดยรวม
```python
def calculate_overall_emotion(emotions):
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
    
    return most_common_emotion[0]
```

## ลำดับขั้นตอนการประมวลผล

### 1. เพิ่มเพลงใหม่

```python
# ขั้นตอนที่ 1: รับข้อมูลจากผู้ใช้
youtube_link = request.form["youtube"]
lyrics = request.form["lyrics"]

# ขั้นตอนที่ 2: ดึงข้อมูลเมตาจาก YouTube API
video_id = extract_video_id(yt_link)
meta = fetch_youtube_metadata(video_id)

# ขั้นตอนที่ 3: ตรวจสอบการซ้ำ
existing = db_query("SELECT id FROM songs WHERE youtube_link=?", (yt_link,), fetch=True)
if existing:
    return render_template("index.html", songs=songs, error="⚠️ เพลงนี้ถูกเพิ่มแล้ว ไม่สามารถเพิ่มซ้ำได้")

# ขั้นตอนที่ 4: เริ่ม transaction
cur.execute("BEGIN TRANSACTION")

# ขั้นตอนที่ 5: แบ่งเนื้อเพลงเป็นส่วนๆ
segments = preprocess_lyrics(lyrics)

# ขั้นตอนที่ 6: วิเคราะห์อารมณ์แต่ละส่วน
emotions = []
for i, seg in enumerate(segments):
    e = detect_emotion(seg)
    emotions.append(e)
    cur.execute("""
        INSERT INTO segments (song_id,segment_order,text,emotion) 
        VALUES (?,?,?,?)""", (song_id, i, seg, e))

# ขั้นตอนที่ 7: สร้างกราฟการเปลี่ยนแปลงอารมณ์
trajectory_html = plot_interactive_trajectory(emotions, meta.get("title"))
cur.execute("UPDATE songs SET graph_html=? WHERE id=?", 
          (trajectory_html, song_id))

# ขั้นตอนที่ 8: commit transaction
conn.commit()
```

### 2. การค้นหาเพลง

```python
# ขั้นตอนที่ 1: แปลง query เป็นลำดับอารมณ์
raw = request.form.get("query", "")
q_tokens = parse_thai_emotion_query(raw)  # แปลงข้อความไทยเป็นลิสต์อารมณ์

# ขั้นตอนที่ 2: ดึงเพลงทั้งหมดและ segments
all_songs = db_query("SELECT id,title,view_count,like_count,upload_date,graph_html FROM songs", fetch=True)

# ขั้นตอนที่ 3: คำนวณคะแนนความตรงกัน
for s in all_songs:
    segs = db_query("SELECT emotion FROM segments WHERE song_id=? ORDER BY segment_order", (s[0],), fetch=True)
    song_seq = [_canonize(x[0]) for x in segs]

    # ตรงตามลำดับแบบ soft-subsequence ก็ถือว่า match
    if soft_subseq_match(q_tokens, song_seq):
        score = calculate_match_score(q_tokens, song_seq)
        scored_songs.append((score, s))

# ขั้นตอนที่ 4: เรียงตามคะแนน
scored_songs.sort(key=lambda x: (-x[0], -x[1][2]))
songs = [s for _, s in scored_songs]
```

### 3. การดูรายละเอียดเพลง

```python
# ดึงข้อมูลเพลงหลัก
song = db_query("""SELECT id,title,youtube_link,upload_date,view_count,like_count,graph_html,lyrics
                   FROM songs WHERE id=?""", (song_id,), fetch=True)

# ดึง segments พร้อมลำดับ
segments = db_query("""SELECT segment_order,text,emotion
                       FROM segments WHERE song_id=? ORDER BY segment_order""",
                    (song_id,), fetch=True)

# คำนวณอารมณ์โดยรวม
emotions = [seg[2] for seg in segments if seg[2]]
overall_emotion = calculate_overall_emotion(emotions)
emotion_explanation = get_emotion_explanation(overall_emotion, emotions)
```

## การใช้งานเครื่องมือและเทคโนโลยี

### 1. Machine Learning และ NLP

#### Transformers Library
```python
# emotion_model.py
from transformers import pipeline

# ใช้โมเดล BART สำหรับการจำแนกอารมณ์
ZS_MODEL = "facebook/bart-large-mnli"
_zs = pipeline("zero-shot-classification", model=ZS_MODEL)

def detect_emotion(text: str, threshold: float = 0.55, multi_label: bool = False) -> str:
    res = _zs(text, candidate_labels=CANDIDATE_LABELS, multi_label=multi_label)
    
    if multi_label:
        picked = [lbl for lbl, sc in zip(res["labels"], res["scores"]) if sc >= threshold]
        if picked:
            return picked[0]  # Return English label directly
    else:
        lbl, sc = res["labels"][0], res["scores"][0]
        if sc >= threshold:
            return lbl  # Return English label directly
    
    # Get Thai emotion and convert back to English
    thai_emotion = _lexicon_fallback(text)
    return THAI_TO_ENG.get(thai_emotion, "neutral")
```

#### การประมวลผลภาษาไทย
```python
# nlp_utils.py
from pythainlp.tokenize import word_tokenize as thai_tokenize
import nltk

def auto_tokenize(text: str) -> str:
    if not text:
        return ""
    
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
```

### 2. การเชื่อมต่อ YouTube API

```python
# youtube_utils.py
from googleapiclient.discovery import build
import re
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

def fetch_youtube_metadata(video_id):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    request = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    )
    response = request.execute()
    
    if not response["items"]:
        return None

    item = response["items"][0]
    snippet = item["snippet"]
    stats = item["statistics"]

    return {
        "title": snippet.get("title"),
        "description": snippet.get("description"),
        "tags": snippet.get("tags", []),
        "upload_date": snippet.get("publishedAt"),
        "view_count": stats.get("viewCount"),
        "like_count": stats.get("likeCount"),
    }

def extract_video_id(url: str):
    pattern = r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None
```

### 3. การค้นหาเวกเตอร์

```python
# vectorstore.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# สร้าง index FAISS
dimension = 384
index = faiss.IndexFlatL2(dimension)

# mapping (id → song/segment)
metadata = []

def add_segments_to_index(song_id, segments):
    vectors = embedder.encode(segments)
    index.add(np.array(vectors, dtype="float32"))
    for i in range(len(segments)):
        metadata.append((song_id, i))

def search_query(query, top_k=5):
    vec = embedder.encode([query])
    D, I = index.search(np.array(vec, dtype="float32"), top_k)
    results = [metadata[i] for i in I[0]]
    return results
```

### 4. การแสดงผลและการวิเคราะห์

```python
# analysis.py
import plotly.express as px
import pandas as pd

def build_trajectory(segments, emotions):
    return [(i, e) for i, e in enumerate(emotions)]

def plot_interactive_trajectory(emotions, song_name):
    df = pd.DataFrame({"step": range(len(emotions)), "emotion": emotions})
    fig = px.line(
        df, x="step", y="emotion",
        title=f"Emotion Trajectory: {song_name}",
        markers=True,
        labels={
            "step": "Step",  # X-axis label in English
            "emotion": "Emotion"  # Y-axis label in English
        }
    )
    return fig.to_html(full_html=False)  # Return HTML string
```

## ตัวอย่างโค้ดและสคริปต์

### 1. การแมปอารมณ์ไทย-อังกฤษ

```python
# emotion_model.py
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
```

### 2. อัลกอริทึมการจับคู่ลำดับอารมณ์

```python
def calculate_match_score(query_emotions, song_emotions):
    """
    คำนวณคะแนนความตรงกันระหว่าง query กับลำดับอารมณ์ของเพลง
    Returns: float (0.0 - 1.0)
    """
    if not query_emotions or not song_emotions:
        return 0.0
    
    # ทำให้เป็นภาษาเดียวกัน (อังกฤษ) เพื่อให้ตรงกับฐานข้อมูล
    def normalize_emotion(e):
        if e in THAI_TO_ENG:
            return THAI_TO_ENG[e]
        return e.lower() if e else e
    
    # แปลงทั้งสองฝั่งให้เป็นภาษาอังกฤษ (ตรงกับฐานข้อมูล)
    query_emotions = [normalize_emotion(e) for e in query_emotions]
    song_emotions = [normalize_emotion(e) for e in song_emotions]
    
    # กรณีอารมณ์คงที่: ถ้า query มีอารมณ์เดียว
    if len(set(query_emotions)) == 1:
        target_emotion = query_emotions[0]
        emotion_count = sum(1 for s in song_emotions if s == target_emotion)
        if emotion_count == 0:
            return 0.0
        # คำนวณสัดส่วนของอารมณ์ที่ตรงกัน
        return min(emotion_count / len(song_emotions), 1.0)
    
    # กรณีปกติ: ค้นหาลำดับอารมณ์
    # ใช้ Longest Common Subsequence (LCS) algorithm
    n, m = len(query_emotions), len(song_emotions)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    
    # คำนวณ LCS
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if query_emotions[i-1] == song_emotions[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[n][m]
    
    # คำนวณคะแนนตามความยาวของ LCS และตำแหน่ง
    if lcs_length == 0:
        return 0.0
    
    # Base score: สัดส่วนของ LCS ต่อความยาวของ query
    base_score = lcs_length / len(query_emotions)
    
    # Position bonus: ให้คะแนนเพิ่มถ้าลำดับตรงกันในตำแหน่งที่ถูกต้อง
    position_bonus = 0.0
    query_idx = 0
    for song_emotion in song_emotions:
        if query_idx < len(query_emotions) and song_emotion == query_emotions[query_idx]:
            query_idx += 1
            position_bonus += 0.1  # bonus สำหรับแต่ละตำแหน่งที่ตรงกัน
    
    # Normalize position bonus
    position_bonus = min(position_bonus, 0.3)  # จำกัด bonus สูงสุด 30%
    
    final_score = min(base_score + position_bonus, 1.0)
    return final_score
```

### 3. การสร้างกราฟอินเตอร์แอคทีฟ

```python
# analysis.py
import plotly.express as px
import pandas as pd

def plot_interactive_trajectory(emotions, song_name):
    df = pd.DataFrame({"step": range(len(emotions)), "emotion": emotions})
    fig = px.line(
        df, x="step", y="emotion",
        title=f"Emotion Trajectory: {song_name}",
        markers=True,
        labels={
            "step": "Step",  # X-axis label in English
            "emotion": "Emotion"  # Y-axis label in English
        }
    )
    
    # เพิ่มการตกแต่งเพิ่มเติม
    fig.update_layout(
        xaxis_title="Step",
        yaxis_title="Emotion",
        hovermode='closest',
        showlegend=False
    )
    
    return fig.to_html(full_html=False)  # Return HTML string
```

### 4. การประมวลผล Query ที่ซับซ้อน

```python
def parse_thai_emotion_query(q: str):
    """
    รับ query ภาษาธรรมชาติ → ลำดับอารมณ์พร้อมข้อมูลเพิ่มเติม
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
    
    return _parse_complex_emotion_query(q)
```

## การเชื่อมต่อกับบริการภายนอก

### 1. YouTube Data API v3

#### การดึงข้อมูลเมตา
```python
from googleapiclient.discovery import build
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

def fetch_youtube_metadata(video_id):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    request = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    )
    response = request.execute()
    
    if not response["items"]:
        return None

    item = response["items"][0]
    snippet = item["snippet"]
    stats = item["statistics"]

    return {
        "title": snippet.get("title"),
        "description": snippet.get("description"),
        "tags": snippet.get("tags", []),
        "upload_date": snippet.get("publishedAt"),
        "view_count": stats.get("viewCount"),
        "like_count": stats.get("likeCount"),
    }
```

#### การตรวจสอบลิงก์
```python
def extract_video_id(url: str):
    pattern = r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None
```

### 2. Hugging Face Transformers

#### โมเดล BART
```python
from transformers import pipeline

# ใช้โมเดลที่ stable
ZS_MODEL = "facebook/bart-large-mnli"
_zs = pipeline("zero-shot-classification", model=ZS_MODEL)

CANDIDATE_LABELS = ["sad", "lonely", "hope", "happy", "excited", "calm", "angry", "neutral"]

def detect_emotion(text: str, threshold: float = 0.55, multi_label: bool = False) -> str:
    if not text.strip():
        return "neutral" 
    
    try:
        res = _zs(text, candidate_labels=CANDIDATE_LABELS, multi_label=multi_label)
        
        if multi_label:
            picked = [lbl for lbl, sc in zip(res["labels"], res["scores"]) if sc >= threshold]
            if picked:
                return picked[0]  # Return English label directly
        else:
            lbl, sc = res["labels"][0], res["scores"][0]
            if sc >= threshold:
                return lbl  # Return English label directly
        
        # Get Thai emotion and convert back to English
        thai_emotion = _lexicon_fallback(text)
        return THAI_TO_ENG.get(thai_emotion, "neutral")
        
    except Exception:
        thai_emotion = _lexicon_fallback(text)
        return THAI_TO_ENG.get(thai_emotion, "neutral")
```

#### Sentence Transformers
```python
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# สร้างเวกเตอร์
vectors = embedder.encode(segments)
```

## วิธีการเรียกใช้

### 1. Web Routes หลัก

```python
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# หน้าหลัก - เพิ่มเพลงและดูเพลงที่มีอยู่
@app.route("/", methods=["GET","POST"])
def index():
    # การประมวลผลเพิ่มเพลงใหม่
    
# ค้นหาเพลง
@app.route("/search", methods=["GET","POST"])
def search():
    # การประมวลผลการค้นหา
    
# ดูรายละเอียดเพลง
@app.route("/song/<int:song_id>")
def song_detail(song_id):
    # การดึงข้อมูลและแสดงผล
    
# รีเฟรชการวิเคราะห์
@app.route("/song/<int:song_id>/refresh")
def refresh_song(song_id):
    # การวิเคราะห์ใหม่
    
# รีบิลด์การวิเคราะห์
@app.route("/song/<int:song_id>/rebuild", methods=["POST"])
def rebuild_song(song_id):
    # การสร้างใหม่ทั้งหมด
    
# ลบเพลง
@app.route("/song/<int:song_id>/delete", methods=["POST"])
def delete_song(song_id):
    # การลบข้อมูล
    
# สำรวจเพลง
@app.route("/explore")
def explore():
    # การแสดงสถิติและข้อมูลทั่วไป
    
# แดชบอร์ดสถิติ
@app.route("/dashboard")
def dashboard():
    # การแสดงสถิติการใช้งาน
    
# API แบ่งคำ
@app.route("/tokenize", methods=["POST"])
def tokenize_text():
    # API สำหรับการแบ่งคำข้อความ
```

### 2. API Endpoints

```bash
# เริ่มต้นเซิร์ฟเวอร์
python app.py

# เข้าใช้งานผ่านเบราว์เซอร์
http://localhost:5000

# การใช้งาน API แบ่งคำ
curl -X POST http://localhost:5000/tokenize \
  -H "Content-Type: application/json" \
  -d '{"lyrics": "เนื้อเพลงที่ต้องการแบ่งคำ"}'

# การค้นหาเพลง
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d 'query=เศร้า → หวัง'
```

### 3. การตั้งค่า

```bash
# การติดตั้ง dependencies
pip install -r requirements.txt

# ตั้งค่าตัวแปรสภาพแวดล้อม
echo "YOUTUBE_API_KEY=your_youtube_api_key_here" > .env

# เริ่มต้นฐานข้อมูล
python db_setup.py

# เรียกใช้แอปพลิเคชัน
python app.py
```

## การจัดการข้อมูล

### 1. สกีมาฐานข้อมูล SQLite

```sql
-- ตารางเพลงหลัก
CREATE TABLE songs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    youtube_link TEXT,
    description TEXT,
    tags TEXT,
    upload_date TEXT,
    view_count INTEGER,
    like_count INTEGER,
    lyrics TEXT,
    image_path TEXT,
    graph_html TEXT  -- การแสดงผล Plotly ที่แคชไว้
);

-- ตารางส่วนของเพลง
CREATE TABLE segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    song_id INTEGER,
    segment_order INTEGER,
    text TEXT,
    emotion TEXT,  -- ป้ายอารมณ์ภาษาอังกฤษ (sad, happy, ฯลฯ)
    FOREIGN KEY(song_id) REFERENCES songs(id)
);
```

### 2. การจัดการธุรกรรม

```python
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

# การเพิ่มเพลงแบบปลอดภัย
try:
    conn = sqlite3.connect("songs.db")
    cur = conn.cursor()
    
    try:
        # เริ่ม transaction
        cur.execute("BEGIN TRANSACTION")
        
        # เพิ่มข้อมูลเพลง
        cur.execute("""
            INSERT INTO songs (title,youtube_link,lyrics)
            VALUES (?,?,?)
        """, (title, link, lyrics))
        
        song_id = cur.lastrowid
        
        # เพิ่ม segments
        for i, seg in enumerate(segments):
            cur.execute("""
                INSERT INTO segments (song_id,segment_order,text,emotion) 
                VALUES (?,?,?,?)""", (song_id, i, seg, emotion))
        
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
```

### 3. การจัดการข้อมูลอารมณ์

```python
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
    กำหนดไอคอนสำหรับแต่ละอารมณ์
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
```

## ข้อดี-ข้อเสียของแต่ละองค์ประกอบ

### 1. ข้อดี

#### ความแม่นยำสูง
- **ใช้โมเดล BART**: โมเดลที่ได้รับการฝึกสอนมาอย่างดีบนข้อมูลขนาดใหญ่
- **ระบบสำรอง**: มี lexicon ภาษาไทยที่ครอบคลุมเป็นการสำรอง
- **เกณฑ์ที่ปรับได้**: สามารถปรับ threshold ตามความต้องการ

#### รองรับหลายภาษา
- **การประมวลผลไทย-อังกฤษ**: สามารถประมวลผลเนื้อเพลงที่ผสมกันได้
- **การแมปสองทิศทาง**: แปลงอารมณ์ไทย-อังกฤษอัตโนมัติ
- **การแบ่งคำอัจฉริยะ**: ใช้ PyThaiNLP และ NLTK

#### การแสดงผลสวยงาม
- **กราฟ Plotly**: แสดงผลแบบอินเตอร์แอคทีฟและตอบสนอง
- **ระบบสีและไอคอน**: แต่ละอารมณ์มีการแสดงผลที่เฉพาะเจาะจง
- **การวิเคราะห์โดยรวม**: คำอธิบายอารมณ์เป็นภาษาธรรมชาติ

#### ระบบค้นหาอัจฉริยะ
- **รองรับหลายรูปแบบ**: ลูกศร, ภาษาธรรมชาติ, อารมณ์เดียว
- **อัลกอริทึมการจับคู่**: LCS, soft subsequence matching
- **การค้นหาเชิงความหมาย**: ใช้ FAISS และ sentence transformers

#### การจัดการข้อมูลมีประสิทธิภาพ
- **SQLite**: ฐานข้อมูลที่เบาและเร็ว
- **การทำ cache**: เก็บกราฟ HTML เพื่อการโหลดที่รวดเร็ว
- **Transaction safety**: การจัดการธุรกรรมที่ปลอดภัย

### 2. ข้อเสีย

#### ความเร็ว
- **โมเดล BART**: ใช้เวลาในการประมวลผลค่อนข้างนาน
- **การประมวลผลต่อเนื่อง**: ไม่เหมาะกับข้อมูลจำนวนมาก
- **การโหลดโมเดล**: ใช้เวลาเริ่มต้นนาน

#### การใช้ทรัพยากร
- **RAM usage**: โมเดล BART ใช้ RAM ค่อนข้างสูง
- **GPU dependency**: ต้องการ GPU สำหรับประสิทธิภาพที่ดี
- **Storage**: ต้องเก็บโมเดลและข้อมูล embeddings

#### ข้อจำกัดภาษา
- **ภาษาอื่นๆ**: ไม่รองรับภาษาอื่นๆ นอกจากไทยและอังกฤษ
- **สำเนียง**: อาจไม่แม่นยำกับภาษาพูดหรือสำเนียงต่างๆ
- **บริบท**: การเข้าใจบริบทอาจจำกัด

#### การพึ่งพา API
- **YouTube API key**: ต้องมี API key ที่ถูกต้อง
- **Rate limiting**: มีข้อจำกัดจำนวนการเรียกใช้
- **ค่าใช้จ่าย**: การใช้งานเกิน limit อาจมีค่าใช้จ่าย

#### ข้อมูลจำกัด
- **คุณภาพขึ้นอยู่กับเนื้อเพลง**: ความแม่นยำขึ้นอยู่กับคุณภาพของข้อมูล
- **การตีความ**: อาจตีความผิดในบางกรณี
- **ความหลากหลาย**: อาจไม่ครอบคลุมทุกประเภทของเพลง

## แนวทางการปรับปรุงและขยายฟีเจอร์ในอนาคต

### 1. การปรับปรุงประสิทธิภาพ

#### การใช้โมเดลที่เล็กลง
```python
# ใช้ DistilBART แทน BART
ZS_MODEL = "distilbart-mnli-12-3"

# หรือใช้ T5 ที่เล็กกว่า
from transformers import T5ForConditionalGeneration, T5Tokenizer

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
```

#### การทำค้างเครื่อง
```python
# Redis สำหรับ caching
import redis
r = redis.Redis(host='localhost', port=6379, db=0)

def detect_emotion_cached(text):
    # ตรวจสอบ cache ก่อน
    cache_key = f"emotion:{hash(text)}"
    cached_result = r.get(cache_key)
    
    if cached_result:
        return cached_result.decode()
    
    # ถ้าไม่มีใน cache ให้คำนวณใหม่
    result = detect_emotion(text)
    r.setex(cache_key, 3600, result)  # Cache 1 ชั่วโมง
    
    return result
```

#### การประมวลผลแบบ Asynchronous
```python
# ใช้ Celery สำหรับ background tasks
from celery import Celery

celery_app = Celery('emotion_app', broker='redis://localhost:6379')

@celery_app.task
def analyze_song_async(song_id, lyrics):
    # วิเคราะห์ใน background
    segments = preprocess_lyrics(lyrics)
    emotions = [detect_emotion(seg) for seg in segments]
    
    # บันทึกผลลัพธ์
    update_song_emotions(song_id, emotions)
    
    return emotions

# ใช้งาน
analyze_song_async.delay(song_id, lyrics)
```

### 2. การขยายความสามารถ

#### การวิเคราะห์เสียง
```python
import librosa
import numpy as np

def analyze_audio_emotion(audio_file):
    # โหลดไฟล์เสียง
    y, sr = librosa.load(audio_file)
    
    # สกัด features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    
    # วิเคราะห์ emotion จาก features เสียง
    emotion = classify_audio_emotion(mfccs, spectral_centroids)
    
    return emotion

def classify_audio_emotion(features):
    # ใช้ ML model สำหรับจำแนกอารมณ์จากเสียง
    pass
```

#### ระบบแนะนำเพลง
```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class MusicRecommendationSystem:
    def __init__(self):
        self.user_emotion_profiles = {}
        self.song_emotion_features = {}
    
    def add_user_interaction(self, user_id, song_id, emotion_score):
        # บันทึกการโต้ตอบของผู้ใช้
        if user_id not in self.user_emotion_profiles:
            self.user_emotion_profiles[user_id] = {}
        
        self.user_emotion_profiles[user_id][song_id] = emotion_score
    
    def recommend_songs(self, user_id, target_emotion, n_recommendations=10):
        # หาเพลงที่ตรงกับอารมณ์เป้าหมาย
        user_profile = self.user_emotion_profiles.get(user_id, {})
        
        # คำนวณความคล้าย
        similarities = []
        for song_id, features in self.song_emotion_features.items():
            similarity = self.calculate_emotion_similarity(
                target_emotion, features
            )
            similarities.append((song_id, similarity))
        
        # เรียงตามความคล้าย
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n_recommendations]
```

#### การวิเคราะห์ผู้ใช้
```python
class UserEmotionAnalyzer:
    def __init__(self):
        self.user_listening_history = {}
        self.emotion_preferences = {}
    
    def track_user_emotion_preference(self, user_id, song_emotions):
        # ติดตามความชอบอารมณ์ของผู้ใช้
        if user_id not in self.emotion_preferences:
            self.emotion_preferences[user_id] = {}
        
        for emotion in song_emotions:
            self.emotion_preferences[user_id][emotion] = \
                self.emotion_preferences[user_id].get(emotion, 0) + 1
    
    def get_user_emotion_profile(self, user_id):
        # สร้างโปรไฟล์อารมณ์ของผู้ใช้
        preferences = self.emotion_preferences.get(user_id, {})
        
        # คำนวณความชอบเป็นเปอร์เซ็นต์
        total = sum(preferences.values())
        if total == 0:
            return {}
        
        return {
            emotion: count/total 
            for emotion, count in preferences.items()
        }
    
    def suggest_emotion_based_playlist(self, user_id):
        # สร้างเพลย์ลิสต์ตามความชอบ
        profile = self.get_user_emotion_profile(user_id)
        
        # เลือกเพลงตามโปรไฟล์
        recommended_songs = []
        for emotion, percentage in profile.items():
            if percentage > 0.3:  # ถ้าชอบมากกว่า 30%
                songs = get_songs_by_emotion(emotion)
                recommended_songs.extend(songs[:5])
        
        return recommended_songs
```

### 3. การปรับปรุงอินเตอร์เฟซ

#### Responsive Design
```css
/* CSS สำหรับการใช้งานบนมือถือ */
.emotion-chart {
    width: 100%;
    height: 300px;
    margin: 10px 0;
}

@media (max-width: 768px) {
    .emotion-chart {
        height: 250px;
        font-size: 12px;
    }
    
    .emotion-controls {
        flex-direction: column;
        gap: 10px;
    }
}

@media (max-width: 480px) {
    .emotion-chart {
        height: 200px;
        padding: 5px;
    }
}
```

#### การแสดงผลแบบ Real-time
```python
from flask_socketio import SocketIO, emit
import eventlet

socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/stream_analysis/<int:song_id>')
def stream_analysis(song_id):
    socketio.emit('analysis_started', {'song_id': song_id})
    
    # วิเคราะห์ทีละส่วน
    segments = preprocess_lyrics(lyrics)
    for i, seg in enumerate(segments):
        emotion = detect_emotion(seg)
        socketio.emit('segment_analyzed', {
            'song_id': song_id,
            'segment_index': i,
            'emotion': emotion,
            'progress': (i+1)/len(segments)
        })
        time.sleep(0.5)  # จำลองการประมวลผล
    
    socketio.emit('analysis_completed', {'song_id': song_id})
```

#### การส่งออกข้อมูล
```python
import pandas as pd
from flask import send_file
import io

@app.route('/export/song/<int:song_id>/<format>')
def export_song_data(song_id, format):
    # ดึงข้อมูลเพลง
    song_data = get_song_with_segments(song_id)
    
    if format == 'excel':
        # สร้าง Excel
        df = pd.DataFrame(song_data['segments'])
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Song Analysis', index=False)
            
            # เพิ่มสรุปสถิติ
            summary = create_emotion_summary(song_data['emotions'])
            summary_df = pd.DataFrame([summary])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        output.seek(0)
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'song_{song_id}_analysis.xlsx'
        )
    
    elif format == 'pdf':
        # สร้าง PDF
        pdf_buffer = create_analysis_pdf(song_data)
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'song_{song_id}_analysis.pdf'
        )
```

### 4. การเพิ่มฟีเจอร์ขั้นสูง

#### การวิเคราะห์การเปรียบเทียบ
```python
def compare_songs_emotion(song_ids):
    """
    เปรียบเทียบอารมณ์ของเพลงหลายเพลง
    """
    songs_data = []
    for song_id in song_ids:
        song = get_song_with_segments(song_id)
        songs_data.append({
            'id': song_id,
            'title': song['title'],
            'emotions': song['emotions'],
            'overall_emotion': calculate_overall_emotion(song['emotions'])
        })
    
    # สร้าง comparison chart
    comparison_data = []
    for song in songs_data:
        for i, emotion in enumerate(song['emotions']):
            comparison_data.append({
                'song': song['title'],
                'step': i,
                'emotion': emotion
            })
    
    df = pd.DataFrame(comparison_data)
    fig = px.line(
        df, x='step', y='emotion', color='song',
        title='Emotion Trajectory Comparison',
        markers=True
    )
    
    return fig.to_html(full_html=False)
```

#### การสร้างเพลย์ลิสต์อัตโนมัติ
```python
def create_emotion_based_playlist(target_emotion_sequence, target_duration=30):
    """
    สร้างเพลย์ลิสต์ตามลำดับอารมณ์ที่ต้องการ
    """
    playlist = []
    current_duration = 0
    
    for emotion in target_emotion_sequence:
        # หาเพลงที่มีอารมณ์ตรงกัน
        candidate_songs = get_songs_by_emotion(emotion)
        
        # เรียงตามความเหมาะสม (ความนิยม, คุณภาพ)
        candidate_songs.sort(key=lambda x: (x['like_count'], x['view_count']), reverse=True)
        
        for song in candidate_songs:
            song_duration = get_song_duration(song['youtube_link'])
            if current_duration + song_duration <= target_duration:
                playlist.append(song)
                current_duration += song_duration
                break
    
    return playlist
```

#### การวิเคราะห์ความรู้สึก
```python
def analyze_listener_emotion(lyrics, audio_features=None):
    """
    วิเคราะห์ว่าผู้ฟังจะรู้สึกอย่างไรเมื่อฟังเพลง
    """
    # วิเคราะห์จากเนื้อเพลง
    lyrical_emotions = detect_emotion_progression(lyrics)
    
    # วิเคราะห์จาก features เสียง (ถ้ามี)
    audio_emotions = []
    if audio_features:
        audio_emotions = classify_audio_emotion(audio_features)
    
    # รวมผลลัพธ์
    listener_response = {
        'expected_emotions': lyrical_emotions,
        'audio_influence': audio_emotions,
        'intensity_prediction': predict_emotion_intensity(lyrics),
        'emotional_arc': create_emotional_arc(lyrical_emotions),
        'listener_recommendations': generate_listener_tips(lyrical_emotions)
    }
    
    return listener_response

def generate_listener_tips(emotions):
    """
    สร้างคำแนะนำสำหรับผู้ฟัง
    """
    tips = []
    
    if 'sad' in emotions:
        tips.append("เหมาะสำหรับช่วงที่ต้องการสะท้อนความรู้สึก")
    
    if 'excited' in emotions:
        tips.append("เหมาะสำหรับการออกกำลังกายหรือปาร์ตี้")
    
    if 'calm' in emotions:
        tips.append("เหมาะสำหรับการผ่อนคลายหรือทำสมาธิ")
    
    return tips
```

### 5. การขยายการรองรับภาษา

#### ภาษาเอเชียอื่นๆ
```python
# เพิ่มการรองรับภาษาจีน
from transformers import BertTokenizer, BertForSequenceClassification

chinese_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
chinese_model = BertForSequenceClassification.from_pretrained('bert-base-chinese')

# เพิ่มการรองรับภาษาญี่ปุ่น
japanese_tokenizer = BertTokenizer.from_pretrained('bert-base-japanese')
japanese_model = BertForSequenceClassification.from_pretrained('bert-base-japanese')

# เพิ่มการรองรับภาษาเกาหลี
korean_tokenizer = BertTokenizer.from_pretrained('kykim/bert-kor-base')
korean_model = BertForSequenceClassification.from_pretrained('kykim/bert-kor-base')

def detect_emotion_multilingual(text, language='thai'):
    """
    ตรวจจับอารมณ์หลายภาษา
    """
    if language == 'chinese':
        return detect_emotion_chinese(text)
    elif language == 'japanese':
        return detect_emotion_japanese(text)
    elif language == 'korean':
        return detect_emotion_korean(text)
    else:
        return detect_emotion(text)
```

#### การแปลภาษาอัตโนมัติ
```python
from googletrans import Translator

translator = Translator()

def translate_lyrics_for_analysis(lyrics, target_lang='en'):
    """
    แปลเนื้อเพลงเพื่อการวิเคราะห์
    """
    try:
        # แยกเนื้อเพลงเป็นส่วนๆ
        sections = preprocess_lyrics(lyrics)
        translated_sections = []
        
        for section in sections:
            # แปลแต่ละส่วน
            translated = translator.translate(section, dest=target_lang)
            translated_sections.append(translated.text)
        
        return ' '.join(translated_sections)
    
    except Exception as e:
        print(f"Translation error: {e}")
        return lyrics  # กลับไปใช้ต้นฉบับ

def analyze_with_auto_translation(lyrics, original_lang='auto'):
    """
    วิเคราะห์พร้อมแปลภาษาอัตโนมัติ
    """
    # ตรวจสอบภาษา
    detected_lang = translator.detect(lyrics).lang
    
    if detected_lang != 'en':
        # แปลเป็นอังกฤษสำหรับการวิเคราะห์
        english_lyrics = translate_lyrics_for_analysis(lyrics, 'en')
        emotions = detect_emotion_progression(english_lyrics)
        
        # แปลกลับเป็นภาษาต้นฉบับสำหรับแสดงผล
        return {
            'original_language': detected_lang,
            'emotions': emotions,
            'translated_for_analysis': english_lyrics
        }
    else:
        return {
            'original_language': 'en',
            'emotions': detect_emotion_progression(lyrics),
            'translated_for_analysis': lyrics
        }
```

#### การตรวจจับภาษาอัตโนมัติ
```python
from langdetect import detect, detect_langs

def detect_language_smart(text):
    """
    ตรวจจับภาษาอัตโนมัติพร้อมความมั่นใจ
    """
    try:
        # ตรวจจับหลายภาษาที่เป็นไปได้
        probabilities = detect_langs(text)
        
        # เลือกภาษาที่มีความมั่นใจสูงสุด
        best_lang = max(probabilities, key=lambda x: x.prob)
        
        return {
            'language': best_lang.lang,
            'confidence': best_lang.prob
        }
    
    except Exception:
        return {
            'language': 'unknown',
            'confidence': 0.0
        }

def auto_language_processing(text):
    """
    ประมวลผลข้อความตามภาษาที่ตรวจจับได้
    """
    lang_info = detect_language_smart(text)
    
    if lang_info['confidence'] > 0.8:
        return {
            'detected_language': lang_info['language'],
            'processing_method': get_processing_method(lang_info['language']),
            'confidence': lang_info['confidence']
        }
    else:
        # ถ้าไม่แน่ใจ ให้ใช้การประมวลผลแบบผสม
        return {
            'detected_language': 'mixed',
            'processing_method': 'mixed_language',
            'confidence': lang_info['confidence']
        }
```

### 6. การปรับปรุงโครงสร้างระบบ

#### Microservices Architecture
```python
# emotion_service.py - Service สำหรับการตรวจจับอารมณ์
from flask import Flask, request, jsonify

emotion_app = Flask(__name__)

@emotion_app.route('/detect_emotion', methods=['POST'])
def detect_emotion_endpoint():
    data = request.json
    text = data.get('text', '')
    emotion = detect_emotion(text)
    return jsonify({'emotion': emotion})

@emotion_app.route('/batch_detect', methods=['POST'])
def batch_detect_emotion_endpoint():
    texts = request.json.get('texts', [])
    emotions = [detect_emotion(text) for text in texts]
    return jsonify({'emotions': emotions})

# search_service.py - Service สำหรับการค้นหา
search_app = Flask(__name__)

@search_app.route('/search', methods=['POST'])
def search_endpoint():
    query = request.json.get('query', '')
    results = perform_advanced_search(query)
    return jsonify({'results': results})

# music_service.py - Service สำหรับการจัดการเพลง
music_app = Flask(__name__)

@music_app.route('/add_song', methods=['POST'])
def add_song_endpoint():
    song_data = request.json
    song_id = add_song_to_database(song_data)
    return jsonify({'song_id': song_id})

@music_app.route('/get_song/<int:song_id>')
def get_song_endpoint(song_id):
    song_data = get_song_data(song_id)
    return jsonify(song_data)
```

#### Docker Containerization
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  emotion-app:
    build: .
    ports:
      - "5000:5000"
    environment:
      - YOUTUBE_API_KEY=${YOUTUBE_API_KEY}
      - DATABASE_URL=sqlite:///songs.db
    volumes:
      - ./data:/app/data
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=emotion_music
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - emotion-app

volumes:
  postgres_data:
```

#### Cloud Integration
```python
# cloud_storage.py
import boto3
from google.cloud import storage
import os

class CloudStorageManager:
    def __init__(self, provider='aws'):
        self.provider = provider
        
        if provider == 'aws':
            self.s3_client = boto3.client('s3')
            self.bucket_name = os.getenv('S3_BUCKET_NAME')
        elif provider == 'gcp':
            self.storage_client = storage.Client()
            self.bucket_name = os.getenv('GCS_BUCKET_NAME')
    
    def upload_analysis_result(self, song_id, result_data):
        """
        อัปโหลดผลการวิเคราะห์ไปยัง cloud storage
        """
        filename = f"analysis/{song_id}/result.json"
        
        if self.provider == 'aws':
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=filename,
                Body=json.dumps(result_data),
                ContentType='application/json'
            )
        elif self.provider == 'gcp':
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(filename)
            blob.upload_from_string(json.dumps(result_data))
    
    def download_audio_for_analysis(self, audio_url, local_path):
        """
        ดาวน์โหลดไฟล์เสียงสำหรับการวิเคราะห์
        """
        if self.provider == 'aws':
            # ใช้ S3 presigned URL
            response = requests.get(audio_url)
            with open(local_path, 'wb') as f:
                f.write(response.content)
        elif self.provider == 'gcp':
            # ใช้ GCS signed URL
            response = requests.get(audio_url)
            with open(local_path, 'wb') as f:
                f.write(response.content)
```

```python
# cloud_ml.py
import tensorflow as tf
import torch
from transformers import pipeline

class CloudMLService:
    def __init__(self, provider='aws'):
        self.provider = provider
        
        if provider == 'aws':
            self.sagemaker_client = boto3.client('sagemaker')
            self.endpoint_name = os.getenv('SAGEMAKER_ENDPOINT')
        elif provider == 'gcp':
            self.ai_platform = discovery.build('ml', 'v1')
            self.model_name = os.getenv('GCP_MODEL_NAME')
    
    def batch_emotion_analysis(self, lyrics_batch):
        """
        ใช้ ML service ใน cloud สำหรับการวิเคราะห์จำนวนมาก
        """
        if self.provider == 'aws':
            # ใช้ SageMaker batch transform
            response = self.sagemaker_client.transform_job(
                TransformJobName=f"emotion-analysis-{int(time.time())}",
                ModelName=self.endpoint_name,
                TransformInput={
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': 's3://my-bucket/input-data/'
                        }
                    }
                },
                TransformOutput={
                    'S3OutputPath': 's3://my-bucket/output-data/'
                }
            )
            return response['TransformJobArn']
        
        elif provider == 'gcp':
            # ใช้ AI Platform Prediction
            instances = [{"lyrics": lyrics} for lyrics in lyrics_batch]
            response = self.ai_platform.projects().predict(
                name=self.model_name,
                body={'instances': instances}
            ).execute()
            return response['predictions']
```

## สรุป

โปรแกรม Emotion Music App เป็นตัวอย่างที่ยอดเยี่ยมของการประยุกต์ใช้เทคโนโลยี Machine Learning และ Natural Language Processing ในการวิเคราะห์เนื้อเพลง ด้วยความสามารถในการประมวลผลที่หลากหลายและการแสดงผลที่สวยงาม

### จุดเด่นหลัก
- **ความแม่นยำสูง**: ใช้โมเดล BART พร้อมระบบสำรอง
- **รองรับหลายภาษา**: ประมวลผลไทยและอังกฤษได้อย่างมีประสิทธิภาพ
- **การแสดงผลที่สวยงาม**: ใช้ Plotly สำหรับกราฟอินเตอร์แอคทีฟ
- **ระบบค้นหาอัจฉริยะ**: รองรับการค้นหาหลายรูปแบบ
- **การจัดการข้อมูลที่มีประสิทธิภาพ**: ใช้ SQLite และการทำ cache

### แนวทางการพัฒนาในอนาคต
- **การปรับปรุงประสิทธิภาพ**: ใช้โมเดลที่เล็กลง, การทำ cache, asynchronous processing
- **การขยายความสามารถ**: เพิ่มการวิเคราะห์เสียง, ระบบแนะนำ, การวิเคราะห์ผู้ใช้
- **การปรับปรุงอินเตอร์เฟซ**: responsive design, real-time updates, data export
- **การเพิ่มฟีเจอร์ขั้นสูง**: การเปรียบเทียบเพลง, เพลย์ลิสต์อัตโนมัติ, การวิเคราะห์ความรู้สึก
- **การขยายการรองรับภาษา**: จีน, ญี่ปุ่น, เกาหลี, การแปลอัตโนมัติ
- **การปรับปรุงโครงสร้างระบบ**: microservices, Docker, cloud integration

โปรแกรมนี้สามารถนำไปพัฒนาต่อเป็นแอปพลิเคชันเชิงพาณิชย์หรือใช้ในการวิจัยได้ โดยมีศักยภาพในการขยายผลและปรับปรุงให้ตอบโจทย์การใช้งานที่หลากหลายมากขึ้นในอนาคต