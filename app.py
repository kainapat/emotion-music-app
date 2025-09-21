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
    # หาแบบ contains เพื่อครอบคลุมคำขยาย เช่น 'มีความสุขมาก'
    for alias, canon in ALIAS2CANON.items():
        if alias in t:
            return canon
    # ถ้าไม่เจอใน lexicon ไทย ลองหาใน lexicon อังกฤษ
    from emotion_model import ENG_TO_THAI
    if t.lower() in ENG_TO_THAI:
        return ENG_TO_THAI[t.lower()]
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
    
    # ลบคำทั่วไปที่ไม่จำเป็น
    q = re.sub(r"เพลงที่|ขอเพลง|แนว|โทน|อารมณ์|ช่วง|looking for|a|consistently|song", "", q, flags=re.IGNORECASE)
    
    # ตรวจสอบกรณีที่ query เป็นแค่ชื่ออารมณ์เดียว
    single_emotion = _canonize(q.strip())
    if single_emotion and single_emotion != "" and single_emotion != "เพลง" and single_emotion != "ที่" and single_emotion != "อารมณ์":
        return [single_emotion]
    
    # แยกคำโดย PyThaiNLP
    tokens = word_tokenize(q)
    
    # สร้าง pattern สำหรับการเปลี่ยนแปลง
    patterns = {
        "start": ["เริ่ม", "ตอนแรก", "แรกๆ", "ช่วงแรก", "starts", "begins"],
        "middle": ["แล้ว", "จากนั้น", "ต่อมา", "ตามด้วย", "then", "becomes", "changes to"],
        "end": ["สุดท้าย", "ตอนจบ", "ท้ายเพลง", "จบ", "ends", "finally"]
    }
    
    emotions = []
    current_position = None
    
    for i, token in enumerate(tokens):
        # ตรวจหาคำบ่งชี้ตำแหน่ง
        for pos, words in patterns.items():
            if any(word in token.lower() for word in words):
                current_position = pos
                break
                
        # หาอารมณ์จากคำ
        emotion = _canonize(token)
        if emotion and emotion != " ":  # ตรวจสอบว่าไม่ใช่ space
            if current_position == "start":
                emotions.insert(0, emotion)
            else:
                emotions.append(emotion)
                
    # ถ้าไม่มีอารมณ์ ลองหาคำที่เกี่ยวข้องกับอารมณ์ทั้งหมด
    if not emotions:
        emotions = [_canonize(t) for t in tokens if _canonize(t) and _canonize(t) != " "]
        
    return [e for e in emotions if e and e != " "]  # กรองเอาแต่ค่าที่ไม่ใช่ string ว่างหรือ space

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
        # ต้องพบอารมณ์นั้นอย่างน้อย 80% ของความยาว sequence
        emotion_count = sum(1 for s in seq if s == target_emotion)
        return emotion_count >= 0.8 * len(seq)
    
    # กรณีปกติ: ค้นหาลำดับอารมณ์
    i = 0
    for s in seq:
        if i < len(target) and s == target[i]:
            i += 1
    return i == len(target)

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

    songs = db_query("SELECT id,title,view_count,like_count,upload_date,graph_html FROM songs", fetch=True)
    return render_template("index.html", songs=songs, error=error)

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
        segments=segments
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