# LearnLog — 학습 습관 트래커 에이전트

LangGraph 기반 맞춤형 학습 습관 트래커. 학습 목표를 설정하고 커리큘럼을 생성한 뒤, 매일 체크인 → 퀴즈 복습 → Notion 일기 포스팅을 통해 학습 습관을 만들어갑니다.

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
  ├─ entry_mode="checkin" ──────────────────────────────────────────▶ checkin
  └─ entry_mode=""  ─▶ mood_check (interrupt①: 기분)
                          └─▶ goal_check (interrupt②: 목표 유지/변경)
                                ├─ setup ─▶ goal_detail (interrupt③: 수준+집중영역)
                                │             └─▶ domain_analysis (도메인 분석)
                                │                   └─▶ curriculum_build (커리큘럼+자료링크 생성)
                                │                         └─▶ persona_build (튜터 페르소나 생성)
                                │                               └─▶ checkin
                                └─ skip  ────────────────────────────▶ checkin

checkin (interrupt④: 오늘 배운 내용)
  └─▶ checkin_response (LLM 튜터 응답)
        └─▶ checkin_action (interrupt⑤: 액션 버튼 선택)
              ├─ quiz   ─▶ quiz_generate (LLM 퀴즈 생성)
              │               └─▶ quiz (interrupt⑥: 답변) ─▶ diary_writer ─▶ notion_post ─▶ END
              ├─ search ─▶ resource_search (interrupt⑦: 퀴즈 제안)
              │               ├─ quiz ─▶ quiz_generate ─▶ quiz ─▶ diary_writer ─▶ notion_post ─▶ END
              │               └─ write ─▶ diary_writer ─▶ notion_post ─▶ END
              └─ diary  ─▶ diary_writer ─▶ notion_post ─▶ END
```

## 노드 역할

| 노드 | interrupt | 역할 |
|------|-----------|------|
| `mood_check` | ① | 기분 입력 → LLM 요약 |
| `goal_check` | ② | 목표 유지/변경 의향 파악 + 주제명 추출 |
| `goal_detail` | ③ | 학습 수준 + 집중 영역 수집 |
| `domain_analysis` | — | 도메인 분석 (DomainAnalysis Structured Output) |
| `curriculum_build` | — | 주차별 커리큘럼 + 자료 링크 생성 (Curriculum Structured Output) |
| `persona_build` | — | 일일 습관 + 튜터 페르소나 생성 + 요약 메시지 |
| `checkin` | ④ | 오늘의 주제·자료 링크 표시 + 학습 내용 입력 + 진도/스트릭 계산 |
| `checkin_response` | — | 오늘 학습에 대한 LLM 튜터 응답 생성 |
| `checkin_action` | ⑤ | 다음 액션 선택 (버튼: 퀴즈/일기/검색) |
| `resource_search` | ⑦ | Tavily 검색 + 퀴즈 진행 여부 확인 |
| `quiz_generate` | — | 스페이스드 리피티션 반영 퀴즈 문제 생성 |
| `quiz` | ⑥ | 퀴즈 답변 수집 → LLM 평가·피드백 |
| `diary_writer` | — | 기분·성취 기반 일기 작성 |
| `notion_post` | — | Notion DB에 일기 저장 + 세션 완료 기록 |

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
| `quiz_questions` | `str` | 생성된 퀴즈 문제 (interrupt 재실행 방지용) |
| `progress_pct` | `float` | 전체 진도율 (0.0 ~ 1.0) |
| `domain_info` | `dict` | `domain_analysis_node` → `curriculum_build_node` 중간값 |
| `user_level` | `str` | 사용자 학습 수준 (`beginner/intermediate/advanced`) |
| `entry_mode` | `str` | 진입 모드 (`"checkin"` or `""`) |
| `session_date` | `str` | 오늘 세션 완료 날짜 (`YYYY-MM-DD`) |

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
| `DEBUG` | `true`로 설정 시 사이드바에 개발자 모드 패널 표시 (기본값: `false`) |

### 개발자 모드 (`DEBUG=true`)

사이드바 하단에 🔧 개발자 모드 패널이 활성화됩니다.

- **상태 주입** — streak, 주차, 토픽, 목표를 원하는 값으로 직접 설정
- **상태 초기화** — 모든 state 초기화 (DB 삭제 없이 처음부터 시작)
- **다음날 시뮬레이션** — 일별 데이터(mood, 성취, 일기)만 초기화, 영구 데이터(goal, curriculum, streak) 유지 → 다음날 체크인 버튼 재노출

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

---

## 로드맵

### v1.1 — 학습 자료 선행 제공 (진행 중)

> 현재 구조는 사용자가 학습 후 보고하는 방식. 학습 자료를 먼저 제공하는 방식으로 플로우를 개선합니다.

**변경 플로우:**
```
오늘 체크인
  → 오늘의 주제 + 학습 자료 링크 (Tavily 선제 검색)
  → 사용자가 자료를 보고 학습
  → "학습 완료" 클릭
  → 퀴즈 → 일기
```

**주요 변경 사항:**
- `checkin_resources_node` 추가 — 체크인 시작 시 오늘 주제로 Tavily 검색, 자료 링크 표시
- `resource_search_node` 역할 축소 — 사후 추가 검색 용도로만 유지
- 커리큘럼 생성 시 각 day의 `topic`에 맞는 공식 문서/튜토리얼 링크 포함

---

### v1.2 — 습관 리스트 + 학습 트랙 통합 (예정)

> 원래 기획 의도(습관 체크리스트)와 학습 커리큘럼을 하나의 서비스로 통합합니다.

**개념:**
```
[습관 트랙]  매일 체크하는 자유 항목
  ☑ 운동 30분
  ☑ 물 2L
  ☐ 명상 10분
  → 항목별 streak 독립 관리

[학습 트랙]  구조화된 커리큘럼
  → 오늘의 주제 + 자료 → 학습 → 퀴즈 → 일기
```

**주요 변경 사항:**
- `HabitList` 별도 State 필드 추가 (`habits: List[dict]`)
- 목표 설정 화면에서 자유 습관 항목 추가/삭제 UI
- 체크인 시 습관 체크 + 학습 내용 입력을 하나의 세션에서 처리
- Notion 일기에 습관 달성 현황 포함
- 사이드바에 습관별 streak 개별 표시

---

### v2.0 — 멀티 플랫폼 (장기)

- Google Calendar 연동 — 커리큘럼 일정 자동 등록
- 아침/저녁 세션 분리 — 아침: 오늘의 학습 자료 브리핑 / 저녁: 체크인 + 퀴즈 + 일기
- 사용자 커리큘럼 업로드 — 직접 만든 커리큘럼 파일 import
