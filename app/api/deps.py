from fastapi import Security, HTTPException, status, Depends
from fastapi.security import APIKeyHeader
from sqlmodel import Session, select
from app.db.session import get_session
from app.db.models import User

# Header name used in client requests
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def get_current_user(
    api_key: str = Security(api_key_header),
    session: Session = Depends(get_session)
) -> User:
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key missing"
        )
    
    statement = select(User).where(User.api_key == api_key)
    user = session.exec(statement).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
    return user