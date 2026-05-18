# LearnLog — Claude 작업 규칙

## 코드 수정 규칙

**코드를 직접 수정하기 전에 반드시 변경사항을 먼저 보여주고 승인을 받아야 합니다.**

### 절차

1. 변경이 필요한 내용을 설명한다
2. 수정 전/후 diff 또는 변경 내용을 보여준다
3. 사용자 승인("적용합시다", "OK", "진행해줘" 등) 이후에 실제 파일을 수정한다

### 예외

아래 경우는 사전 승인 없이 바로 진행 가능하다:

- 사용자가 명시적으로 "바로 적용해줘" 또는 "수정해줘"라고 요청한 경우
- 새 파일 생성 (기존 코드에 영향 없음)
- 테스트 실행, 문법 확인 등 읽기 전용 작업

## 프로젝트 개요

LangGraph 기반 학습 습관 트래커. 자세한 내용은 README.md 참고.

## 핵심 아키텍처 규칙

- **노드당 interrupt 1개**: 하나의 노드에서 `interrupt()`를 두 번 호출하면 LangGraph 체크포인트가 불안정해진다. 반드시 노드를 분리할 것.
- **interrupt 앞에 LLM 호출 금지**: `interrupt()` 호출 전에 LLM을 호출하면 안 된다. LangGraph는 재개(resume) 시 노드를 처음부터 다시 실행하므로, interrupt 앞의 LLM이 재호출되어 다른 결과가 나올 수 있다. LLM 결과가 필요하면 별도 노드에서 state에 저장한 뒤 interrupt 노드에서 읽어야 한다.
- **JSON 파싱**: LLM 응답은 `parse_json()` 헬퍼를 사용한다. `removeprefix/removesuffix` 방식은 사용하지 않는다.
- **테스트**: 노드 변경 시 `tests/test_nodes.py`도 함께 업데이트한다.

## 주요 파일

- `graph.py` — 그래프 노드 전체 정의 (State, Pydantic 모델, 노드, 엣지)
- `app.py` — Streamlit UI
- `tests/test_nodes.py` — Mock LLM 단위 테스트 (API 비용 없음)
- `tests/test_ai_judge.py` — AI-as-judge 품질 테스트 (실제 LLM 호출)
- `tests/conftest.py` — 공통 fixture (base_state, state_with_goal 등)

## 자주 쓰는 명령어

```bash
uv run streamlit run app.py            # UI 실행
uv run pytest tests/test_nodes.py -v  # 단위 테스트 (비용 없음)
uv run pytest tests/test_ai_judge.py -v -m ai_judge  # AI judge 테스트
python -c "import ast; ast.parse(open('graph.py', encoding='utf-8').read())"  # 문법 확인
```

## 알려진 이슈

- **graph.py 편집 시 UTF-8 잘림**: Edit 툴이 멀티바이트 문자(═, 한글 등) 경계에서 파일을 잘라낼 수 있다.
  - 편집 후 반드시 `python -c "import ast; ast.parse(...)"` 로 문법 검증할 것
  - 잘린 경우 바이트 레벨로 복구:
    ```python
    with open('graph.py', 'rb') as f: data = f.read()
    src = data.decode('utf-8', errors='ignore')
    with open('graph.py', 'w', encoding='utf-8') as f: f.write(src + tail)
    ```

## 테스트 작성 규칙

- `llm.with_structured_output(Model).invoke(...)` 모킹 방법:
  ```python
  structured = MagicMock()
  structured.invoke.return_value = mock_pydantic_obj
  mock_llm.with_structured_output.return_value = structured
  ```
- interrupt가 있는 노드는 `patch("graph.interrupt", return_value="...")` 로 모킹
- 새 State 필드 추가 시 `tests/conftest.py`의 `base_state`에도 추가할 것
