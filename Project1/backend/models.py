from pydantic import BaseModel
from typing import Any, Dict, Optional, List

class ProcessRequest(BaseModel):
    # adjust fields to match what your MethodFiles need
    input_text: Optional[str] = None
    params: Dict[str, Any] = {}

class ProcessResponse(BaseModel):
    ok: bool
    result: Dict[str, Any]  # can carry plots/data/metrics
    message: Optional[str] = None
