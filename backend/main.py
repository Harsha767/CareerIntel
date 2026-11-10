
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers.recommendations import router as recommendations_router
from backend.routers.profile import router as profile_router
from backend.database import initialize_database
from backend.routers.market_insights import router as market_insights_router
app = FastAPI(
    title="CareerIntel API",
    description="AI-powered career recommendation platform",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers OUTSIDE the middleware configuration
app.include_router(recommendations_router)
app.include_router(profile_router)
app.include_router(market_insights_router)

@app.on_event("startup")
def startup_event():
    initialize_database()
@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "application": "CareerIntel",
        "message": "Backend connected successfully",
    }


@app.get("/api/dashboard")
def dashboard():
    return {
        "status": "success",
        "metrics": {
            "recommended_jobs": None,
            "average_match": None,
            "skills_identified": None,
        },
    }
