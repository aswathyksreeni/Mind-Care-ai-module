from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select
from app.db.session import get_session
from app.db.models import User
from app.schemas.response import APIResponse
# We will create this utility next
from app.core.vector_store import upsert_patient_embedding 

router = APIRouter()

class OnboardRequest(BaseModel):
    username: str
    age: int
    background: str
    # Optional extra data (e.g., {"diagnosis": "anxiety"})
    extra_data: Dict[str, Any] = {}

class OnboardResponse(BaseModel):
    user_id: str
    api_key: str
    message: str

@router.post("/", response_model=APIResponse[OnboardResponse])
async def onboard_user(
    payload: OnboardRequest, 
    session: Session = Depends(get_session)
):
    # 1. Check if username exists
    existing_user = session.exec(select(User).where(User.username == payload.username)).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already taken")

    # 2. Create User
    new_user = User(
        username=payload.username,
        age=payload.age,
        background=payload.background,
        profile_data=payload.extra_data
    )
    session.add(new_user)
    session.commit()
    session.refresh(new_user)

    # 3. Index User in Qdrant (Async/Background ideally, but inline for now)
    try:
        # We combine background + extra data for a rich semantic profile
        full_text = f"Age: {new_user.age}. Background: {new_user.background}. Metadata: {new_user.profile_data}"
        upsert_patient_embedding(
            user_id=new_user.id,
            text_content=full_text,
            payload={"username": new_user.username, "age": new_user.age}
        )
    except Exception as e:
        print(f"Vector Store Error: {e}") 
        # We don't fail the request if vector store fails, just log it

    # 4. Return API Key
    return APIResponse(
        ErrorCode=0,
        Data=OnboardResponse(
            user_id=str(new_user.id),
            api_key=new_user.api_key,
            message="User onboarded successfully. Save your API Key!"
        ),
        Message="Success"
    )