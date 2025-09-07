import sqlite3

conn = sqlite3.connect("songs.db")
c = conn.cursor()

# ตารางเพลงหลัก (เพิ่ม image_path)
c.execute("""
CREATE TABLE IF NOT EXISTS songs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    youtube_link TEXT,
    description TEXT,
    tags TEXT,
    upload_date TEXT,
    view_count INTEGER,
    like_count INTEGER,
    lyrics TEXT
)
""")

# ตาราง segment + อารมณ์
c.execute("""
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    song_id INTEGER,
    segment_order INTEGER,
    text TEXT,
    emotion TEXT,
    FOREIGN KEY(song_id) REFERENCES songs(id)
)
""")

try:
    c.execute("ALTER TABLE songs ADD COLUMN image_path TEXT")
except sqlite3.OperationalError:
    print("Column already exists")

try:
    c.execute("ALTER TABLE songs ADD COLUMN graph_html TEXT")
except sqlite3.OperationalError:
    print("graph_html column already exists")

conn.commit()
conn.close()