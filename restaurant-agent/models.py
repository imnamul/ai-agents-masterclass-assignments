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

class HandoffData(BaseModel):

    to_agent_name: str
    issue_type: str
    issue_description: str
    reason: str
