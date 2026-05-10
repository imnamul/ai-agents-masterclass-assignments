# ── Notion 연결 퀵 테스트 ──────────────────────────────────────
import requests
from datetime import date
import os
from dotenv import load_dotenv
load_dotenv()

NOTION_TOKEN       = os.getenv("NOTION_TOKEN", "")
NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID", "")

# .env에서 불러온 값 확인
print(f"NOTION_TOKEN       : {'✅ 설정됨' if NOTION_TOKEN and NOTION_TOKEN != 'your-notion-token' else '❌ 미설정'}")
print(f"NOTION_DATABASE_ID : {'✅ 설정됨' if NOTION_DATABASE_ID and NOTION_DATABASE_ID != 'your-database-id' else '❌ 미설정'}")
print()

headers = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28",
}

# ── 1) DB 접근 권한 확인 ──────────────────────────────────────
print("1️⃣ DB 접근 권한 확인...")
res = requests.get(
    f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}",
    headers=headers,
    timeout=10
)
if res.status_code == 200:
    db = res.json()
    print(f"   ✅ DB 이름: {db.get('title', [{}])[0].get('plain_text', 'Unknown')}")
    print(f"   📋 속성 목록: {list(db.get('properties', {}).keys())}")
else:
    print(f"   ❌ 오류 ({res.status_code}): {res.text[:200]}")

print()

# ── 2) 테스트 페이지 생성 ─────────────────────────────────────
print("2️⃣ 테스트 페이지 생성...")
today = date.today().isoformat()

payload = {
    "parent": {"database_id": NOTION_DATABASE_ID},
    "properties": {
        "Title":         {"title":     [{"text": {"content": f"🧪 {today} Connection Test"}}]},
        "Date":          {"date":      {"start": today}},
        "Learning Goal": {"rich_text": [{"text": {"content": "Notion 연결 테스트"}}]},
        "Mood":          {"rich_text": [{"text": {"content": "테스트 중 😊"}}]},
        "Streak":        {"number":    0},
    },
    "children": [{
        "object": "block", "type": "paragraph",
        "paragraph": {"rich_text": [{"type": "text", "text": {"content": "LearnLog 연결 테스트 페이지입니다. 삭제해도 됩니다."}}]},
    }],
}

res2 = requests.post("https://api.notion.com/v1/pages", headers=headers, json=payload, timeout=10)
if res2.status_code == 200:
    url = res2.json().get("url", "")
    print(f"   ✅ 페이지 생성 성공!")
    print(f"   🔗 {url}")
else:
    print(f"   ❌ 오류 ({res2.status_code}): {res2.text[:300]}")
