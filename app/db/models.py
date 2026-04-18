import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field, JSON, Column

# Helper for timezone-aware UTC timestamps
def utc_now():
    return datetime.now(timezone.utc)

class User(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    username: str = Field(unique=True, index=True)
    api_key: str = Field(default_factory=lambda: uuid.uuid4().hex, unique=True, index=True)
    age: int
    background: str
    profile_data: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)

class MoodEntry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: uuid.UUID = Field(foreign_key="user.id", index=True)
    
    # --- NEW UI FIELDS ---
    overall_mood: str       # e.g., "Very Happy", "Neutral"
    intensity_level: int    # 1-10 Slider value
    emotion_tags: List[str] = Field(default_factory=list, sa_column=Column(JSON)) # Chips
    description: str        # "Describe your feelings" text
    
    timestamp: datetime = Field(default_factory=utc_now)

class ChatMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: uuid.UUID = Field(foreign_key="user.id")
    role: str
    content: str
    timestamp: datetime = Field(default_factory=utc_now)