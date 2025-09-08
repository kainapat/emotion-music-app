from googleapiclient.discovery import build
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

def fetch_youtube_metadata(video_id):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
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