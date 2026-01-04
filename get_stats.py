"""
สคริปต์ดึงสถิติจากฐานข้อมูลจริง
รันด้วย: python get_stats.py
"""
import sqlite3

def get_database_stats():
    conn = sqlite3.connect("songs.db")
    cur = conn.cursor()
    
    # สถิติเพลง
    cur.execute("SELECT COUNT(*) FROM songs")
    total_songs = cur.fetchone()[0]
    
    # สถิติ segments
    cur.execute("SELECT COUNT(*) FROM segments")
    total_segments = cur.fetchone()[0]
    
    # ท่อนต่อเพลง
    cur.execute("SELECT song_id, COUNT(*) as cnt FROM segments GROUP BY song_id")
    segments_per_song = cur.fetchall()
    
    if segments_per_song:
        counts = [x[1] for x in segments_per_song]
        avg_segments = round(sum(counts) / len(counts), 2)
        min_segments = min(counts)
        max_segments = max(counts)
    else:
        avg_segments = 0
        min_segments = 0
        max_segments = 0
    
    # การกระจายอารมณ์
    cur.execute("""
        SELECT emotion, COUNT(*) as cnt
        FROM segments
        GROUP BY emotion
        ORDER BY cnt DESC
    """)
    emotion_stats = cur.fetchall()
    
    conn.close()
    
    return {
        "total_songs": total_songs,
        "total_segments": total_segments,
        "avg_segments": avg_segments,
        "min_segments": min_segments,
        "max_segments": max_segments,
        "emotion_stats": emotion_stats
    }

if __name__ == "__main__":
    stats = get_database_stats()
    
    print("=" * 50)
    print("📊 สถิติจากฐานข้อมูลจริง")
    print("=" * 50)
    print(f"จำนวนเพลงทั้งหมด: {stats['total_songs']}")
    print(f"จำนวนท่อนทั้งหมด: {stats['total_segments']}")
    print(f"จำนวนท่อนต่อเพลงเฉลี่ย: {stats['avg_segments']}")
    print(f"จำนวนท่อนต่อเพลงต่ำสุด: {stats['min_segments']}")
    print(f"จำนวนท่อนต่อเพลงสูงสุด: {stats['max_segments']}")
    print()
    print("การกระจายอารมณ์:")
    for emotion, count in stats['emotion_stats']:
        pct = round(count / stats['total_segments'] * 100, 1)
        print(f"  {emotion}: {count} ({pct}%)")
