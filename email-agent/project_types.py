from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Email(BaseModel):
    message_id: str
    date: str  # ISO 8601 string 'YYYY-MM-DD HH:MM:SS'
    subject: Optional[str] = None
    from_address: Optional[str] = None
    to_addresses: List[str] = []  # Populated from recipients table
    cc_addresses: List[str] = []  # Populated from recipients table
    bcc_addresses: List[str] = []  # Populated from recipients table
    body: Optional[str] = None
    file_name: Optional[str] = None

class Input(BaseModel):
    question: str
    inbox_address: str

class EvaluationRollOut(BaseModel):
    question: str
    agent_answer: str
    golden_answer: str

class SearchResult(BaseModel):
    message_id: str
    snippet: str
