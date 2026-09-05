# query.py - Chat endpoint.
#
# The heavy lifting now lives in the multi-agent layer: orchestrator.py routes each message
# and dispatches to the specialized agents in agents.py (SQL analyst, validator, conceptual,
# greeting, column-names, knowledge). This module is just the authenticated HTTP surface.
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from auth import get_current_user
from orchestrator import handle

ask_question = APIRouter()


class AskRequest(BaseModel):
    message: str
    table_name: str | None = None


@ask_question.post("/ask")
def ask_post(payload: AskRequest, current_user: str = Depends(get_current_user)):
    """Ask a question about the data - requires authentication."""
    try:
        return handle(payload.message, current_user, payload.table_name)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in ask_post: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
