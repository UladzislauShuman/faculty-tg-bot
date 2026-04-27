from pydantic import BaseModel, Field


class EvalScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Score from 0.0 to 1.0")
    reason: str = Field(description="Explanation of the score in Russian")
