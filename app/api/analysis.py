from typing import cast, List, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select, desc
from app.db.session import get_session
from app.api.deps import get_current_user
from app.db.models import User, MoodEntry, ChatMessage
from app.agents.psychiatrist import get_analyst_agent, MoodAnalysisSchema
from app.core.vector_store import upsert_patient_embedding, search_similar_patients
from app.schemas.response import APIResponse
from app.agents.psychiatrist import get_wellness_agent, WellnessPlan

router = APIRouter()

# --- 1. MOOD ANALYSIS (POST) ---
@router.post("/analyze", response_model=APIResponse[MoodAnalysisSchema])
async def trigger_mood_analysis(
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_session)
):
    # Fetch recent chat history
    statement = select(ChatMessage).where(ChatMessage.user_id == current_user.id).order_by(desc(ChatMessage.timestamp)).limit(15)
    messages = db_session.exec(statement).all()
    
    if not messages:
        # Default response for empty chat (matches your UI Neutral state)
        default_data = MoodAnalysisSchema(
            overall_mood="Neutral",
            intensity_level=5,
            emotion_tags=["Calm"],
            description="No recent conversation to analyze."
        )
        return APIResponse(ErrorCode=0, Data=default_data, Message="No history found.")

    # Run Analyst Agent
    transcript = "\n".join([f"{msg.role}: {msg.content}" for msg in messages][::-1])
    analyst = get_analyst_agent()
    result = analyst.run(f"Fill out the Mood Check-In based on this:\n{transcript}") # type: ignore

    if not result or not result.content:
         raise HTTPException(status_code=500, detail="Analysis failed.")
    
    # Cast response to our Schema
    data = cast(MoodAnalysisSchema, result.content)

    # Save to Database
    new_entry = MoodEntry(
        user_id=current_user.id,
        overall_mood=data.overall_mood,
        intensity_level=data.intensity_level,
        emotion_tags=data.emotion_tags, # SQLModel saves this as JSON
        description=data.description
    )
    db_session.add(new_entry)
    db_session.commit()

    # Update Vector Profile (for Similarity Search)
    rich_text = f"Mood: {data.overall_mood}. Tags: {', '.join(data.emotion_tags)}. Desc: {data.description}"
    upsert_patient_embedding(
        user_id=current_user.id,
        text_content=rich_text,
       payload={
            "username": current_user.username,
            "mood": data.overall_mood,
            "intensity": data.intensity_level,      # <--- Added
            "tags": data.emotion_tags,              # <--- Added
            "description": data.description         # <--- Added
        }
    )

    return APIResponse(ErrorCode=0, Data=data, Message="Analysis Complete")


# --- 2. SIMILAR PATIENTS (GET) ---
# This uses the imported 'search_similar_patients' function
@router.get("/similar_patients", response_model=APIResponse[List[Any]])
async def find_similar_patients(
    current_user: User = Depends(get_current_user)
):
    # Query Qdrant using the user's background or latest mood description
    # We prioritize the background for finding long-term matches
    query_text = current_user.background
    
    matches = search_similar_patients(query_text=query_text, limit=5)
    
    # Filter out the current user from results
    similar_users = [m for m in matches if m['point_id'] != str(current_user.id)]
    
    return APIResponse(
        ErrorCode=0,
        Data=similar_users,
        Message="Similar patients retrieved."
    )
    
    
    


@router.get("/recommendations", response_model=APIResponse[WellnessPlan])
async def get_mood_activities(
    current_user: User = Depends(get_current_user),
    db_session: Session = Depends(get_session)
):
    # 1. Get the LATEST mood entry
    statement = select(MoodEntry).where(MoodEntry.user_id == current_user.id).order_by(desc(MoodEntry.timestamp)).limit(1)
    last_mood = db_session.exec(statement).first()
    
    if not last_mood:
        raise HTTPException(status_code=400, detail="No mood data found. Please check-in first.")

    # 2. Prepare context for the AI
    # We convert the JSON list of tags back to a string for the prompt
    tags_str = ", ".join(last_mood.emotion_tags) if last_mood.emotion_tags else "None"
    
    prompt = f"""
    User Status:
    - Overall Mood: {last_mood.overall_mood}
    - Intensity (1-10): {last_mood.intensity_level}
    - Specific Feelings: {tags_str}
    - User's Description: {last_mood.description}
    
    Provide a wellness plan.
    """

    # 3. Run the Wellness Agent
    coach = get_wellness_agent()
    result = coach.run(prompt) # type: ignore
    
    if not result or not result.content:
         raise HTTPException(status_code=500, detail="Failed to generate recommendations.")

    # 4. Return structured JSON
    return APIResponse(
        ErrorCode=0,
        Data=result.content,
        Message="Wellness plan generated."
    )