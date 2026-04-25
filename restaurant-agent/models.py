from pydantic import BaseModel
from typing import Optional


class RestaurantContext(BaseModel):
    restaurant_name: str = "나물이네"
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None
    table_or_order_id: Optional[str] = None
    opening_hours: Optional[str] = "11:00 AM – 10:00 PM"
    avg_prep_minutes: Optional[int] = 20
    max_party_size: Optional[int] = 10

class InputGuardRailOutput(BaseModel):

    is_off_topic: bool
    reason: str

# ══════════════════════════════════════════════════════════════════════════════
# Shared output schemas
# ══════════════════════════════════════════════════════════════════════════════
class ResponseQualityGuardrailOutput(BaseModel):
    is_professional_and_polite: bool
    exposes_internal_info: bool
    reasoning: str 

class MenuGuardrailOutput(BaseModel):
    allergy_disclaimer_present: bool   # True  → "서버에게 알레르기 정보를 알려주세요" 포함
    mentions_off_menu_item: bool       # True  → 메뉴에 없는 항목 언급 → 차단
    reasoning: str
 
 
class OrderGuardrailOutput(BaseModel):
    order_summary_present: bool        # True  → 주문 항목 요약 포함
    mentions_unavailable_item: bool    # True  → 메뉴에 없는 항목 → 차단
    reasoning: str
 
 
class ReservationGuardrailOutput(BaseModel):
    has_required_fields: bool          # True  → 날짜/인원/이름 중 하나 이상 포함
    outside_business_hours: bool       # True  → 영업시간 외 예약 시도 → 차단
    reasoning: str
 


class HandoffData(BaseModel):

    #to_agent_name: str
    issue_type: str
    issue_description: str
    reason: str
