
import json
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.database import get_connection


router = APIRouter(
    prefix="/api",
    tags=["Profile"],
)


# ============================================================
# PROFILE REQUEST MODEL
# ============================================================

class ProfileRequest(BaseModel):
    full_name: str = Field(default="", max_length=100)
    target_role: str = Field(default="", max_length=150)
    location: str = Field(default="", max_length=150)
    education: str = Field(default="", max_length=200)
    experience_level: str = Field(default="", max_length=100)
    summary: str = Field(default="", max_length=2000)
    skills: list[str] = Field(default_factory=list)


# ============================================================
# DATABASE ROW CONVERSION
# ============================================================

def row_to_profile(row):
    return {
        "full_name": row["full_name"],
        "target_role": row["target_role"],
        "location": row["location"],
        "education": row["education"],
        "experience_level": row["experience_level"],
        "summary": row["summary"],
        "skills": json.loads(row["skills"] or "[]"),
        "updated_at": row["updated_at"],
    }


# ============================================================
# GET PROFILE
# ============================================================

@router.get("/profile")
def get_profile():
    try:
        with get_connection() as connection:
            row = connection.execute(
                "SELECT * FROM career_profile WHERE id = 1"
            ).fetchone()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail="Profile not found",
            )

        return {
            "status": "success",
            "profile": row_to_profile(row),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load profile: {str(exc)}",
        )


# ============================================================
# SAVE PROFILE
# ============================================================

@router.put("/profile")
def save_profile(request: ProfileRequest):
    try:
        # Remove blank skills and duplicate entries.
        skills = list(
            dict.fromkeys(
                skill.strip()
                for skill in request.skills
                if skill and skill.strip()
            )
        )

        updated_at = datetime.now(timezone.utc).isoformat()

        with get_connection() as connection:

            connection.execute(
                """
                INSERT INTO career_profile (
                    id,
                    full_name,
                    target_role,
                    location,
                    education,
                    experience_level,
                    summary,
                    skills,
                    updated_at
                )
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)

                ON CONFLICT(id) DO UPDATE SET
                    full_name = excluded.full_name,
                    target_role = excluded.target_role,
                    location = excluded.location,
                    education = excluded.education,
                    experience_level = excluded.experience_level,
                    summary = excluded.summary,
                    skills = excluded.skills,
                    updated_at = excluded.updated_at
                """,
                (
                    request.full_name.strip(),
                    request.target_role.strip(),
                    request.location.strip(),
                    request.education.strip(),
                    request.experience_level.strip(),
                    request.summary.strip(),
                    json.dumps(skills),
                    updated_at,
                ),
            )

            row = connection.execute(
                "SELECT * FROM career_profile WHERE id = 1"
            ).fetchone()

            connection.commit()

        return {
            "status": "success",
            "message": "Profile saved successfully",
            "profile": row_to_profile(row),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save profile: {str(exc)}",
        )
