# CareerIntel – AI-Powered Career Intelligence Platform

CareerIntel is an AI-powered career intelligence platform designed to help job seekers discover relevant career opportunities, evaluate their skills, and understand the skills required in the job market.

The platform combines job recommendation techniques, skill analysis, and market insights to provide personalized career guidance.

## Key Features

### 1. AI-Powered Job Recommendations
- Recommends relevant job opportunities based on user profiles.
- Matches candidate skills with job requirements.
- Uses content similarity and skill-based scoring.
- Considers target roles and experience levels.
- Ranks recommendations using a recommendation score.

### 2. Skill Gap Analysis
- Identifies skills already possessed by the candidate.
- Highlights skills that may need improvement.
- Analyzes skills appearing in recommended job opportunities.
- Helps users identify areas for professional development.

### 3. Career Intelligence Dashboard
- Displays job recommendation statistics.
- Shows average recommendation match scores.
- Provides quick access to career insights and skill analysis.
- Presents recent job recommendations.

### 4. Market Insights
- Explores frequently requested skills.
- Provides insights into job-market requirements.
- Helps users understand industry and career trends.

### 5. User Profile Management
- Stores candidate profile information.
- Supports target roles, education, experience levels, and skills.
- Uses saved profile information to personalize recommendations.

## Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React, Vite, JavaScript |
| UI and Charts | CSS, Lucide React, Recharts |
| Backend | Python, FastAPI |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Database | SQLite |
| API Documentation | Swagger UI |

## System Architecture

CareerIntel follows a frontend-backend architecture.

1. **Frontend:** React provides the interactive dashboard and user interface.
2. **Backend:** FastAPI handles API requests, profile management, recommendations, and market insights.
3. **Recommendation Engine:** Processes job and candidate information to calculate recommendation scores.
4. **Data Processing:** Cleans and prepares job datasets for recommendation and analysis.
5. **Database:** SQLite stores user profile information.

## Project Structure

```text
CareerIntel/
│
├── backend/
│   ├── main.py
│   ├── database.py
│   └── routers/
│       ├── profile.py
│       ├── recommendations.py
│       └── market_insights.py
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── src/
│   ├── preprocessing/
│   ├── recommendation/
│   └── evaluation/
│
├── data/
├── results/
├── docs/
├── experiments/
├── tests/
├── app.py
├── pyproject.toml
├── Makefile
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or later
- Node.js and npm
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Harsha767/CareerIntel.git
cd CareerIntel
```

### 2. Set Up the Python Environment

```bash
python -m venv .venv
```

Activate the environment.

**macOS / Linux:**

```bash
source .venv/bin/activate
```

Install the Python dependencies declared in `pyproject.toml` according to the project's package configuration.

### 3. Start the Backend

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Backend URL:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs

### 4. Start the Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend URL:

http://localhost:5173

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Check backend health |
| `/api/profile` | GET | Retrieve saved profile |
| `/api/profile` | POST | Create or update profile |
| `/api/recommendations` | POST | Generate job recommendations |
| `/api/market-insights` | GET | Retrieve market insights |

Refer to Swagger UI at `/docs` for the available request parameters and response schemas.

## Recommendation Methodology

CareerIntel evaluates job opportunities using multiple recommendation signals:

- Content similarity
- Candidate skill coverage
- Job requirement coverage
- Target-role relevance
- Experience-level compatibility

The recommendation engine combines these signals into a final recommendation score to rank relevant opportunities.

## Future Enhancements

- Integration with additional job-market data sources.
- More advanced semantic job matching.
- Personalized career path recommendations.
- Expanded salary and industry analytics.
- Improved learning-path recommendations.
- Additional model evaluation and performance comparisons.

## Project Status

CareerIntel is an ongoing final-year academic project focused on AI-powered job recommendations and career intelligence.

## Author

**Harsha**

GitHub: https://github.com/Harsha767

## License

This project is developed for academic and educational purposes. A formal open-source license has not yet been specified.
