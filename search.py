from vectorstore import search_query

def match_query(query_text):
    # แปลง query เช่น "เศร้า → หวัง" เป็น "sad hope"
    q = query_text.replace("→", " ")
    return search_query(q, top_k=10)