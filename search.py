from vectorstore import search_query
import numpy as np

def calculate_emotion_similarity(query_emotions, song_emotions):
    """
    คำนวณความคล้ายของลำดับอารมณ์ระหว่าง query กับเพลง
    ใช้ Dynamic Time Warping algorithm แบบง่าย
    """
    if not query_emotions or not song_emotions:
        return 0.0
        
    # สร้าง matrix ความคล้าย
    n, m = len(query_emotions), len(song_emotions)
    dp = np.zeros((n, m))
    
    # ใส่ค่าเริ่มต้น
    dp[0][0] = 1 if query_emotions[0] == song_emotions[0] else 0
    
    # คำนวณ similarity score
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            
            score = 1 if query_emotions[i] == song_emotions[j] else 0
            prev = []
            if i > 0:
                prev.append(dp[i-1][j])
            if j > 0:
                prev.append(dp[i][j-1])
            if i > 0 and j > 0:
                prev.append(dp[i-1][j-1])
                
            dp[i][j] = score + max(prev) if prev else score
            
    return dp[n-1][m-1] / max(n, m)  # normalize score

def match_query(query_text, min_similarity=0.6):
    """
    ค้นหาเพลงที่มีลำดับอารมณ์คล้ายกับ query
    รองรับทั้งการค้นหาแบบใช้ลูกศร (→) และภาษาธรรมชาติ
    """
    from app import parse_thai_emotion_query, db_query
    
    # แยกอารมณ์จาก query
    query_emotions = parse_thai_emotion_query(query_text)
    if not query_emotions:
        return []
        
    # ดึงข้อมูลเพลงทั้งหมด
    songs = db_query("""
        SELECT s.id, s.title, s.view_count, s.like_count, s.upload_date, s.graph_html,
               GROUP_CONCAT(seg.emotion) as emotions
        FROM songs s
        LEFT JOIN segments seg ON s.id = seg.song_id
        GROUP BY s.id
    """, fetch=True)
    
    # คำนวณความคล้ายและจัดอันดับ
    results = []
    for song in songs:
        song_emotions = song[6].split(',') if song[6] else []
        similarity = calculate_emotion_similarity(query_emotions, song_emotions)
        if similarity >= min_similarity:
            results.append((similarity, song))
    
    # เรียงตามความคล้ายมากไปน้อย
    results.sort(reverse=True, key=lambda x: x[0])
    
    # ส่งคืนเฉพาะข้อมูลเพลง (ไม่รวม similarity score)
    return [song for _, song in results[:10]]