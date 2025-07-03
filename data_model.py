from typing import Dict, List, Optional

from pydantic import BaseModel


class References(BaseModel):
    segment_id: int
    description: str


class Aspect(BaseModel):
    title: str
    description: str
    summary: str
    segments: List[References]


class ViewSummaryResponse(BaseModel):
    title: str
    description: str
    summary: str


class TopicModelResponse(BaseModel):
    topics: List[str]


class GetLightragQueryRequest(BaseModel):
    query: str
    conversation_history: Optional[List[Dict[str, str]]] = None
    echo_segment_ids: Optional[List[str]] = None
    echo_conversation_ids: Optional[List[str]] = None
    echo_project_ids: Optional[List[str]] = None
    auto_select_bool: bool = False
    get_transcripts: bool = False
    top_k: int = 60