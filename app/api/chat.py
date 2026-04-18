
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session
from app.db.session import get_session
from app.api.deps import get_current_user
from app.db.models import User, ChatMessage
from app.agents.psychiatrist import get_therapist_agent
from app.schemas.response import APIResponse

router = APIRouter()

# --- Request/Response Models ---
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponseData(BaseModel):
    response: str
    session_id: str

@router.post("/", response_model=APIResponse[ChatResponseData])
async def chat_with_therapist(
    payload: ChatRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    try:
        # 1. Get the Agent for this specific user
        agent = get_therapist_agent(user_id=str(current_user.id), session_id=payload.session_id)
        
        # 2. Run the Agent
        # We assume agent.run() returns a RunResponse object with a .content string
        result = agent.run(payload.message) # type: ignore
        
        # Check if result is valid
        if not result or not hasattr(result, "content"):
             raise HTTPException(status_code=500, detail="Agent failed to generate a response.")

        ai_text = str(result.content) # Ensure it's a string

        # 3. Save Chat Log to SQL
        # (Useful for analytics, separate from Agent's internal memory)
        user_msg = ChatMessage(user_id=current_user.id, role="user", content=payload.message)
        ai_msg = ChatMessage(user_id=current_user.id, role="assistant", content=ai_text)
        session.add(user_msg)
        session.add(ai_msg)
        session.commit()

        # 4. Return clean APIResponse
        return APIResponse(
            ErrorCode=0,
            Data=ChatResponseData(
                response=ai_text,
                session_id=payload.session_id
            ),
            Message="Success"
        )

    except Exception as e:
        return APIResponse[None](
            ErrorCode=500,
            Data=None,
            Message=f"Chat Error: {str(e)}"
        )