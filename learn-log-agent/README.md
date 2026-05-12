# LearnLog — 학습 습관 트래커 에이전트

LangGraph 기반 맞춤형 학습 습관 트래커. 매일 기분과 학습 성취를 체크인하고, 퀴즈로 복습하고, Notion에 일기를 자동 포스팅합니다.

## 주요 기능

- **기분 체크인** — 매일 기분을 기록하고 따뜻한 한 문장으로 요약
- **목표 설정** — 새 학습 목표를 추가/변경하거나 기존 목표 유지
- **맞춤 학습 계획** — 사용자 수준(입문/초급/중급/고급)과 집중 영역을 반영한 커리큘럼 자동 생성
- **동적 튜터 페르소나** — 도메인·수준에 맞는 AI 튜터 시스템 프롬프트 자동 생성
- **학습 체크인 + 스트릭** — 오늘 학습 성취 기록 및 연속 달성 일수 추적
- **자료 검색** — Tavily로 학습 목표 관련 최신 자료 자동 검색
- **퀴즈** — 오늘 학습 내용 기반 퀴즈 생성·평가·피드백 (quiz_history 누적)
- **Notion 일기 자동 포스팅** — 기분·성취를 묶어 Notion DB에 저장

## 그래프 구조

```
START
  └─▶ mood_check  (interrupt①: 기분)
        └─▶ goal_check  (interrupt②: 목표 유지/변경)
              ├─ setup ─▶ goal_detail (interrupt③: 수준 + 집중 영역)
              │             └─▶ goal_setup (커리큘럼 + 튜터 페르소나 생성)
              │                   └─▶ checkin (interrupt④: 오늘 학습)
              └─ skip  ──────────────────────▶ checkin (interrupt④: 오늘 학습)
                                                ├─ search ─▶ resource_search (interrupt⑤: 퀴즈 제안)
                                                │               ├─ quiz ─▶ quiz (interrupt⑥: 답변)
                                                │               │           └─▶ diary_writer ─▶ notion_post ─▶ END
                                                │               └─────────▶ diary_writer ─▶ notion_post ─▶ END
                                                ├─ quiz  ─▶ quiz (interrupt⑤: 답변)
                                                │               └─▶ diary_writer ─▶ notion_post ─▶ END
                                                └─ write ─▶ diary_writer ─▶ notion_post ─▶ END
```

## 노드 역할

| 노드 | interrupt | 역할 |
|------|-----------|------|
| `mood_check` | ① | 기분 입력 → LLM 요약 |
| `goal_check` | ② | 목표 유지/변경 의향 파악 + 주제명 추출 |
| `goal_detail` | ③ | 학습 수준 + 집중 영역 수집 |
| `goal_setup` | — | 도메인 분석 → 커리큘럼 → 습관 → 튜터 페르소나 통합 생성 |
| `checkin` | ④ | 오늘 학습 내용 입력 + 스트릭 증가 |
| `resource_search` | ⑤ | Tavily 검색 + 퀴즈 진행 여부 확인 |
| `quiz` | ⑤/⑥ | 퀴즈 생성 → 답변 수집 → 평가·피드백 |
| `diary_writer` | — | 기분·성취 기반 일기 작성 |
| `notion_post` | — | Notion DB에 일기 저장 |

## State

| 필드 | 타입 | 설명 |
|------|------|------|
| `messages` | `list` | 대화 히스토리 (`add_messages` reducer) |
| `learning_goals` | `List[str]` | 전체 목표 목록 |
| `active_goal` | `str` | 오늘 활성 목표 |
| `mood` | `str` | 오늘 기분 요약 |
| `today_achievements` | `str` | 오늘 학습 성취 |
| `streak` | `int` | 연속 달성 일수 |
| `search_results` | `str` | Tavily 검색 결과 |
| `diary_content` | `str` | 생성된 일기 |
| `next_action` | `str` | 조건 분기용 (`setup/skip/search/quiz/write`) |
| `curriculum` | `dict` | 주차별 커리큘럼 구조 |
| `current_week` | `int` | 현재 몇 주차 |
| `current_topic` | `str` | 오늘 학습 토픽 |
| `tutor_persona` | `str` | 동적 생성 튜터 System Prompt |
| `quiz_history` | `List[dict]` | 퀴즈 기록 (스페이스드 리피티션용) |
| `progress_pct` | `float` | 전체 진도율 (0.0 ~ 1.0) |
| `user_level` | `str` | 사용자 학습 수준 (`beginner/intermediate/advanced`) |

## 설치 및 실행

```bash
# 의존성 설치
uv sync

# .env 설정 (필수)
cp .env.example .env
# OPENAI_API_KEY, TAVILY_API_KEY, NOTION_TOKEN, NOTION_DATABASE_ID 입력
```

### Streamlit UI (권장)

```bash
uv run streamlit run app.py
```

### LangGraph Studio

```bash
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
| `Learning Goal` | rich_text |
| `Mood` | rich_text |
| `Streak` | number |

## 테스트

```bash
# 단위 테스트 (Mock LLM, API 비용 없음)
uv run pytest tests/test_nodes.py -v

# AI-as-judge 품질 테스트 (실제 LLM 호출, 비용 발생)
uv run pytest tests/test_ai_judge.py -v -m ai_judge
```

### 테스트 커버리지

| 테스트 | 대상 노드 |
|--------|-----------|
| `test_mood_node_*` | `mood_check` — 기분 요약, 인사말 분기 |
| `test_goal_check_*` | `goal_check` — 목표 추출, skip/setup 분기, fallback |
| `test_goal_detail_*` | `goal_detail` — 레벨 추출, focus 반영, fallback |
| `test_goal_setup_*` | `goal_setup` — 커리큘럼 생성, user_level 반영, fallback |
| `test_checkin_*` | `checkin` — 키워드 감지, 스트릭, 메시지 |
| `test_quiz_node_*` | `quiz` — 피드백 반환, history 누적, 페르소나 사용 |
| `test_diary_writer_*` | `diary_writer` — 기분/성취 fallback, 일기 생성 |
| `test_notion_post_*` | `notion_post` — 빈 값 guard, 성공/실패 분기 |

## 세션 초기화

테스트 데이터를 지우고 처음부터 시작하려면:

```bash
# Streamlit 종료 후
del learnlog.db   # Windows
rm learnlog.db    # Mac/Linux

uv run streamlit run app.py
```
