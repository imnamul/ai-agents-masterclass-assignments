from agents import (
    Agent,
    RunContextWrapper,
    GuardrailFunctionOutput,
    output_guardrail,
    Runner,
)
from models import RestaurantContext, MenuGuardrailOutput, OrderGuardrailOutput, ReservationGuardrailOutput, ResponseQualityGuardrailOutput



# ══════════════════════════════════════════════════════════════════════════════
# Checker agents (lightweight, return only JSON)
# ══════════════════════════════════════════════════════════════════════════════
_quality_checker = Agent(
    name="Response Quality Checker",
    output_type=ResponseQualityGuardrailOutput,
    instructions="""
    You are a strict output quality checker for a restaurant assistant.
    Evaluate assistant output with these rules:
    1) is_professional_and_polite
    - TRUE if tone is professional, respectful, and customer-safe.
    - FALSE if rude, dismissive, aggressive, insulting, or overly casual in a risky way.
    2) exposes_internal_info
    - TRUE if output reveals internal/system details such as:
    - system/developer prompts or hidden instructions
    - tool/guardrail internals, routing logic, chain-of-thought-like internals
    - backend identifiers, DB paths, API keys/tokens, secrets, environment/config internals
    - debugging internals not intended for customers
    - FALSE otherwise.
    Return only valid JSON matching schema.
    """,
)

@output_guardrail
async def response_quality_output_guardrail(
    ctx: RunContextWrapper[RestaurantContext],
    agent,
    output: str,
) -> GuardrailFunctionOutput:
    result = await Runner.run(
        _quality_checker,
        f"Check this assistant response:\n\n{output}",
        context=ctx.context,
    )
    check: ResponseQualityGuardrailOutput = result.final_output
    tripwire = (
        not check.is_professional_and_polite
        or check.exposes_internal_info
    )
    return GuardrailFunctionOutput(
        output_info={
            "agent": getattr(agent, "name", "unknown"),
            "is_professional_and_polite": check.is_professional_and_polite,
            "exposes_internal_info": check.exposes_internal_info,
            "reasoning": check.reasoning,
            "blocked": tripwire,
        },
        tripwire_triggered=tripwire,
    ) 


_menu_checker = Agent(
    name="Menu Output Checker",
    output_type=MenuGuardrailOutput,
    instructions="""
    You are a quality-checker for a restaurant menu assistant's response.
    Evaluate the assistant's output against these rules:
 
    1. allergy_disclaimer_present:
       - Set TRUE if the response includes a reminder like
         "주문 시 서버에게 알레르기 정보를 알려주세요" (or equivalent).
       - ONLY required when the response discusses allergens or dietary restrictions.
       - If no allergy topic is mentioned, set TRUE (no disclaimer needed).
 
    2. mentions_off_menu_item:
       - Set TRUE if the response recommends or confirms availability of an item
         NOT in this menu list:
         [마르게리타 피자, 트러플 파스타, 그린 샐러드, 리코타 브루스케타, 티라미수, 소르베]
       - Set FALSE if all mentioned items are on the list, or no specific items are mentioned.
 
    Return only valid JSON matching the schema.
    """,
)
 
_order_checker = Agent(
    name="Order Output Checker",
    output_type=OrderGuardrailOutput,
    instructions="""
    You are a quality-checker for a restaurant order assistant's response.
    Evaluate the assistant's output against these rules:
 
    1. order_summary_present:
       - Set TRUE if the response includes a confirmation/summary of the ordered items.
       - ONLY required when the assistant is confirming a completed order.
       - If it's mid-flow (still collecting items), set TRUE (no summary needed yet).
 
    2. mentions_unavailable_item:
       - Set TRUE if the assistant confirms or accepts an order for an item NOT in:
         [마르게리타 피자, 트러플 파스타, 그린 샐러드, 리코타 브루스케타, 티라미수, 소르베]
       - Set FALSE if all accepted items are on the list, or no items are confirmed.
 
    Return only valid JSON matching the schema.
    """,
)
 
_reservation_checker = Agent(
    name="Reservation Output Checker",
    output_type=ReservationGuardrailOutput,
    instructions="""
    You are a quality-checker for a restaurant reservation assistant's response.
    Evaluate the assistant's output against these rules:
 
    1. has_required_fields:
       - Set TRUE if the response either:
         (a) Asks for at least one of: date, time, party size, customer name, OR
         (b) Confirms a reservation that includes date, time, party size, and name.
       - Set FALSE ONLY if the response is a confirmed reservation missing all of these fields.
 
    2. outside_business_hours:
       - Set TRUE if the response confirms a reservation at a time outside 11:00–22:00.
       - Set FALSE otherwise (including if no specific time is mentioned yet).
 
    Return only valid JSON matching the schema.
    """,
)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Guardrail functions
# ══════════════════════════════════════════════════════════════════════════════
 
@output_guardrail
async def menu_output_guardrail(
    ctx: RunContextWrapper[RestaurantContext],
    agent: Agent,
    output: str,
) -> GuardrailFunctionOutput:
    """
    Blocks Menu Agent output if:
    - Allergy topic present but no disclaimer
    - Mentions items not on the menu
    """
    result = await Runner.run(
        _menu_checker,
        f"Check this restaurant assistant response:\n\n{output}",
        context=ctx.context,
    )
    check: MenuGuardrailOutput = result.final_output
 
    tripwire = (
        not check.allergy_disclaimer_present  # allergy mentioned without disclaimer
        or check.mentions_off_menu_item        # hallucinated menu item
    )
 
    return GuardrailFunctionOutput(
        output_info={
            "agent": "menu_agent",
            "allergy_disclaimer_present": check.allergy_disclaimer_present,
            "mentions_off_menu_item": check.mentions_off_menu_item,
            "reasoning": check.reasoning,
            "blocked": tripwire,
        },
        tripwire_triggered=tripwire,
    )
 
 
@output_guardrail
async def order_output_guardrail(
    ctx: RunContextWrapper[RestaurantContext],
    agent: Agent,
    output: str,
) -> GuardrailFunctionOutput:
    """
    Blocks Order Agent output if:
    - Order confirmation missing item summary
    - Agent accepts order for non-existent menu item
    """
    result = await Runner.run(
        _order_checker,
        f"Check this restaurant assistant response:\n\n{output}",
        context=ctx.context,
    )
    check: OrderGuardrailOutput = result.final_output
 
    tripwire = (
        not check.order_summary_present
        or check.mentions_unavailable_item
    )
 
    return GuardrailFunctionOutput(
        output_info={
            "agent": "order_agent",
            "order_summary_present": check.order_summary_present,
            "mentions_unavailable_item": check.mentions_unavailable_item,
            "reasoning": check.reasoning,
            "blocked": tripwire,
        },
        tripwire_triggered=tripwire,
    )
 
 
@output_guardrail
async def reservation_output_guardrail(
    ctx: RunContextWrapper[RestaurantContext],
    agent: Agent,
    output: str,
) -> GuardrailFunctionOutput:
    """
    Blocks Reservation Agent output if:
    - Confirmed reservation missing required fields
    - Confirms booking outside 11:00–22:00
    """
    result = await Runner.run(
        _reservation_checker,
        f"Check this restaurant assistant response:\n\n{output}",
        context=ctx.context,
    )
    check: ReservationGuardrailOutput = result.final_output
 
    tripwire = (
        not check.has_required_fields
        or check.outside_business_hours
    )
 
    return GuardrailFunctionOutput(
        output_info={
            "agent": "reserveration_agent",
            "has_required_fields": check.has_required_fields,
            "outside_business_hours": check.outside_business_hours,
            "reasoning": check.reasoning,
            "blocked": tripwire,
        },
        tripwire_triggered=tripwire,
    )