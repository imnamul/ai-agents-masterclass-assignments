# LearnLog — 학습 습관 트래커 에이전트

LangGraph 기반 학습 습관 트래커. 매일 기분과 학습 성취를 체크인하고 Notion에 일기를 자동 포스팅합니다.

## 기능

- **기분 + 목표 체크인** — 매일 기분을 기록하고 기존 목표를 유지하거나 새 목표를 추가/변경
- **학습 체크인 + 스트릭** — 오늘 학습 성취를 기록하고 연속 달성 일수 추적
- **자료 검색** — Tavily로 학습 목표 관련 최신 자료 자동 검색
- **Notion 일기 자동 포스팅** — 기분·성취·검색 자료를 묶어 Notion DB에 저장

## 그래프 구조

```
START
  └─▶ goal_check (interrupt①: 기분 + 목표 의향)
        ├─ setup ─▶ goal_setup ─▶ checkin (interrupt②: 오늘 학습)
        └─ skip  ──────────────▶ checkin (interrupt②: 오늘 학습)
                                    ├─ search ─▶ resource_search ─▶ diary_writer ─▶ notion_post ─▶ END
                                    └─ write  ──────────────────▶ diary_writer ─▶ notion_post ─▶ END
```

| 노드 | 역할 |
|------|------|
| `goal_check` | interrupt①: 기분 체크 + 목표 유지/변경 의향 파악 |
| `goal_setup` | 새 목표 설정 및 일일 습관 분해 |
| `checkin` | interrupt②: 오늘 학습 내용 입력 + 스트릭 증가 |
| `resource_search` | Tavily 웹 검색으로 관련 학습 자료 수집 |
| `diary_writer` | 기분·성취를 묶어 일기 작성 |
| `notion_post` | Notion 데이터베이스에 일기 저장 |

## State

| 필드 | 타입 | 설명 |
|------|------|------|
| `messages` | `list` | 대화 히스토리 |
| `learning_goals` | `List[str]` | 전체 목표 목록 |
| `active_goal` | `str` | 오늘 활성 목표 |
| `mood` | `str` | 오늘 기분 |
| `today_achievements` | `str` | 오늘 학습 성취 |
| `streak` | `int` | 연속 달성 일수 |
| `search_results` | `str` | Tavily 검색 결과 |
| `diary_content` | `str` | 생성된 일기 |
| `next_action` | `str` | 조건 분기용 |

## 설치 및 실행

```bash
# 의존성 설치
uv sync

# .env 설정 (필수)
cp .env.example .env
# OPENAI_API_KEY, TAVILY_API_KEY, NOTION_TOKEN, NOTION_DATABASE_ID 입력

# 개발 서버 실행
langgraph dev
```

Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

## 환경변수

| 변수 | 설명 |
|------|------|
| `OPENAI_API_KEY` | OpenAI API 키 |
| `TAVILY_API_KEY` | Tavily Search API 키 |
| `NOTION_TOKEN` | Notion Integration 시크릿 |
| `NOTION_DATABASE_ID` | 일기 저장할 Notion DB ID |

## Notion DB 속성

| 속성명 | 타입 |
|--------|------|
| `Title` | title |
| `Date` | date |
| `학습 목표` | rich_text |
| `기분` | rich_text |
| `스트릭` | number |

## 테스트 (Studio UI 기준)

**1단계 — initial state 전달**
```json
{
  "messages": [], "learning_goals": [], "active_goal": "",
  "mood": "", "today_achievements": "", "streak": 0,
  "search_results": "", "diary_content": "", "next_action": ""
}
```

**2단계 — interrupt① 응답**
```
오늘 기분 좋아요! LangGraph 공부를 새로 시작하고 싶어요
```

**3단계 — interrupt② 응답**
```
오늘 interrupt()와 Command(resume) 사용법을 배웠어요. 관련 자료도 찾아주세요
```
