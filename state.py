from typing import TypedDict, List, Optional
from pydantic import BaseModel

# ---- STATE ----
class GuardState(BaseModel):
    guard_status: bool = False
    trusted_user: bool = False    # True (trusted/None), False (intruder)
    escalation_level: int = 0 # 0: default, 1: 1st convo 2: not leaving , 3: last warn 4: raise Alarm
    messages: List[str] = [] # conversation with intruder