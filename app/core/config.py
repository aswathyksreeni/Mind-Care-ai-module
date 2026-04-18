from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
   GROQ_API_KEY: str 
   GEMINI_API_KEY: str
   GEMINI_MODEL_NAME: str
   GROQ_MODEL_NAME: str 
   VERSION: str 
   DATABASE_URL: str = "sqlite:///./jobcompanion.db"
   
   # --- SECURITY CONFIG ---
   SECRET_KEY: str = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7" 
   ACCESS_TOKEN_EXPIRE_MINUTES: int = 300
   ALGORITHM: str = "HS256"
   model_config = SettingsConfigDict(
      env_file=".env",
      case_sensitive=True, # Fixed: usually env vars are case sensitive
      extra="ignore"       # Fixed: ignore extra env vars to prevent crashes
   )

settings = Settings() #type: ignore