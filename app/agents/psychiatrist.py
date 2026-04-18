from typing import List, Literal
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.google import Gemini
from agno.db.sqlite import SqliteDb
from app.core.config import settings
from agno.models.groq import Groq


groq_model = Groq(id=settings.GROQ_MODEL_NAME,api_key=settings.GROQ_API_KEY)
gemin_model = Gemini(id=settings.GEMINI_MODEL_NAME, api_key=settings.GEMINI_API_KEY)
# --- UI-MATCHING SCHEMA ---
class MoodAnalysisSchema(BaseModel):
    overall_mood: Literal["Very Happy", "Happy", "Neutral", "Sad", "Very Sad"] = Field(
        ..., description="The primary mood category matching the 5 UI buttons."
    )
    intensity_level: int = Field(
        ..., ge=1, le=10, description="The intensity of the feeling on a scale of 1-10."
    )
    # The AI will select relevant tags from the UI's list
    emotion_tags: List[str] = Field(
        ..., 
        description="Select 1-3 relevant tags from: [Happy, Grateful, Content, Excited, Hopeful, Peaceful, Confident, Motivated, Sad, Anxious, Stressed, Angry, Frustrated, Overwhelmed, Lonely, Tired, Calm, Neutral, Thoughtful, Reflective, Uncertain, Curious]"
    )
    description: str = Field(
        ..., 
        description="A natural language summary for the 'Describe Your Feelings' text box."
    )

# --- MEMORY ---
agent_db = SqliteDb( db_file="agent_storage/memory.db")

def get_therapist_agent(user_id: str, session_id: str = "default") -> Agent:
    """The Chatbot (Unchanged logic, just context)"""
    return Agent(
        model=groq_model,
        db=agent_db,
        user_id=user_id,
        session_id=session_id,
        learning=True,
        add_history_to_context=True,
        description="You are Dr. Agno, an empathetic AI Psychiatrist.",
        instructions=["Be supportive, concise, and professional."],
        markdown=True
    )

def get_analyst_agent() -> Agent:
    """The UI-Aware Analyst"""
    return Agent(
        model=groq_model,
        description="You are a Clinical Data Analyst filling out a UI form.",
        # Forces AI to output data that perfectly fits your Frontend
        output_schema=MoodAnalysisSchema, 
        instructions=[
            "Analyze the chat history.",
            "1. Determine the 'Overall Mood' (Very Happy to Very Sad).",
            "2. Rate the 'Intensity Level' (1-10).",
            "3. Select matching 'Emotion Tags' strictly from the provided list.",
            "4. Write a brief 'Description' of what is on the user's mind."
        ]
    )
    
    



class Activity(BaseModel):
    title: str = Field(..., description="Short title of the activity (e.g., 'Box Breathing').")
    description: str = Field(..., description="One sentence instruction on what to do.")
    category: Literal["Mindfulness", "Physical", "Social", "Creative", "Rest"] = Field(..., description="Type of activity.")
    duration_minutes: int = Field(..., description="Estimated time to complete.")

class WellnessPlan(BaseModel):
    advice_summary: str = Field(..., description="A warm, empathetic 1-sentence opening.")
    activities: List[Activity] = Field(..., description="A list of 3 tailored activities.")

# --- NEW: WELLNESS AGENT ---
def get_wellness_agent() -> Agent:
    """
    Generates actionable coping strategies based on mood data.
    """
    return Agent(
        model=groq_model,
        description="You are a CBT-based Wellness Coach.",
        output_schema=WellnessPlan, 
        instructions=[
            "Review the user's mood, intensity, and tags.",
            "Suggest 3 specific, small, and achievable activities.",
            "If mood is 'Sad' or 'Low Energy', suggest gentle things (e.g., stretching, tea).",
            "If mood is 'Anxious' or 'High Intensity', suggest grounding techniques (e.g., breathing, naming objects).",
            "If mood is 'Happy', suggest capitalization strategies (e.g., journaling, sharing gratitude)."
        ]
    )