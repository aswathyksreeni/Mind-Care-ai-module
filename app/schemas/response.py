from typing import Generic, TypeVar, Optional
from pydantic import BaseModel

T = TypeVar('T')

class APIResponse(BaseModel, Generic[T]):
    ErrorCode: int
    Data: Optional[T] = None
    Message: str