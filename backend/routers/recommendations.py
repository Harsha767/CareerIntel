
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommendation.recommend_jobs import recommend_jobs


router = APIRouter(
    prefix="/api",
    tags=["Recommendations"],
)


class RecommendationRequest(BaseModel):
    profile: str = Field(min_length=1)
    preferred_skills: Optional[list[str]] = None

    target_role: Optional[str] = None
    experience_level: Optional[str] = None

    top_k: int = Field(default=10, ge=1, le=50)
    min_similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )


@router.post("/recommendations")
def get_recommendations(request: RecommendationRequest):
    try:
        results = recommend_jobs(
            profile=request.profile,
            preferred_skills=request.preferred_skills,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
            target_role=request.target_role,
            experience_level=request.experience_level,
        )

        records = results.astype(object).where(
            results.notna(),
            None,
        ).to_dict(orient="records")

        return {
            "status": "success",
            "total": len(records),
            "recommendations": records,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(exc)}",
        )
