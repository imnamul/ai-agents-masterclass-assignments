import streamlit as st
from agents import (
    Agent, 
    RunContextWrapper, 
    input_guardrail, 
    Runner, 
    GuardrailFunctionOutput,
    handoff,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents.extensions import handoff_filters
from models import RestaurantContext, InputGuardRailOutput, HandoffData


input_guardrail_agent = Agent(
    name="Input Guardrail Agent",
    instructions="""
    You are a guardrail classifier for a restaurant assistant.
    Your task is to decide whether the user's message is ON-TOPIC for this restaurant service.
    Treat as ON-TOPIC if the user is asking about:
    - Menu items, ingredients, prices, dietary/allergy info, recommendations
    - Placing, changing, canceling, or tracking food orders
    - Table reservations, availability, modifying/canceling reservations
    - Restaurant basics such as location, hours, seating, wait time, payment methods
    - Short greetings or small talk directly related to starting/continuing restaurant service
    Treat as OFF-TOPIC if the user asks for unrelated domains, including (but not limited to):
    - Coding/programming help
    - Math/homework solving
    - Legal/medical/financial advice unrelated to restaurant service
    - General world knowledge not connected to this restaurant
    - Requests for actions outside restaurant support scope
    Important rules:
    - Be permissive for short conversational openers (e.g., "안녕하세요", "도와주세요").
    - If unclear, prefer ON-TOPIC unless clearly unrelated.
    - Return a concise reason when marking OFF-TOPIC.
    Output should match InputGuardRailOutput:
    - is_off_topic: true/false
    - reasoning: short explanation
    """,
    output_type=InputGuardRailOutput,
)


@input_guardrail
async def off_topic_guardrail(
    wrapper: RunContextWrapper[RestaurantContext], 
    agent: Agent[RestaurantContext], 
    input: str,
): 
    result = await Runner.run(
        input_guardrail_agent, 
        input, 
        context=wrapper.context,
    )

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_off_topic,
    ) 


def dynamic_triage_agent_instrcutions(
    wrapper: RunContextWrapper[RestaurantContext], 
    agent: Agent[RestaurantContext]
):
    return f"""
    {RECOMMENDED_PROMPT_PREFIX}
 
    You are the front-desk assistant at {wrapper.context.restaurant_name}.
    Greet the customer warmly. If their name is known, use it: {wrapper.context.customer_name or "손님"}.
 
    YOUR ONLY JOB: Understand what the customer needs and route them to the right specialist.
    If the customer expresses dissatisfaction, complaint, refund demand, or asks for a manager, prioritize COMPLAINTS AGENT.
    Never answer menu, order, or reservation questions yourself — always hand off.
 
    ROUTING GUIDE:
 
    🍽️ MENU AGENT — Route here for:
    - Questions about the menu, today's specials
    - Ingredient inquiries ("파스타에 뭐가 들어가나요?")
    - Allergy and dietary questions ("비건 메뉴 있나요?", "글루텐 프리 메뉴?")
    - Pricing information
 
    🛒 ORDER AGENT — Route here for:
    - Placing a new order (매장/포장/배달)
    - Modifying or cancelling an existing order
    - Order status inquiries
 
    📅 RESERVATION AGENT — Route here for:
    - Booking a table (예약하고 싶어요)
    - Modifying or cancelling a reservation
    - Availability inquiries ("금요일에 4명 자리 있나요?")
    - Special seating requests

    😟 COMPLAINTS AGENT — Route here for:
    - 서비스 불만 (직원 응대, 대기시간, 주문 누락/지연)
    - 음식 품질 불만 (맛, 온도, 이물감 등)
    - 결제/청구 관련 불만
    - 환불, 할인, 매니저 연결 요청
    - 심각한 이슈(식품 안전, 차별/안전 문제, 법적 분쟁 가능성)
 
    ROUTING PROCESS:
    1. Listen to the customer's request
    2. If truly unclear, ask ONE clarifying question
    3. Announce the handoff clearly in Korean:
       - 메뉴 문의: "메뉴 전문가에게 연결해 드릴게요..."
       - 주문: "주문 담당에게 연결해 드릴게요..."
       - 예약: "예약 담당에게 연결해 드릴게요..."
       - 불만 접수: "불편을 드려 죄송합니다. 고객 케어 담당자에게 바로 연결해 드릴게요..."
    4. Then immediately transfer to the appropriate agent
    """
 
def handle_handoff(
    wrapper: RunContextWrapper[RestaurantContext],
    input_data: HandoffData,
):

    with st.sidebar:
        st.write(
            f"""
            Handling off to {input_data.to_agent_name}
            Reason: {input_data.reason}
            Issue Type: {input_data.issue_type}
            Description: {input_data.issue_description}
        """
        )


def make_handoff(agent):

    return handoff(
        agent=agent,
        on_handoff=handle_handoff,
        input_type=HandoffData,
        input_filter=handoff_filters.remove_all_tools,
    )



triage_agent = Agent(
    name="triage_agent",
    instructions=dynamic_triage_agent_instrcutions,
    input_guardrails=[
        off_topic_guardrail,
    ],

    # tools=[
    #     technical_agent.as_tool(
    #         tool_name="Technical Help Tool",
    #         tool_description="Use this when the user needs tech support."
    #     )
    # ],
)