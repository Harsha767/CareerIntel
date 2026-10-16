# CareerIntel

A Hybrid Language-Model and Knowledge-Graph Framework for Skill-Aware Job Recommendation and Labour-Market Analytics.

Research extension of the capstone "AI-Enhanced Personalized Job Recommendation Platform".

## Research questions
See `docs/RESEARCH_PLAN.md`.

## Quickstart
```bash
make setup
source .venv/bin/activate
make test
python -m careerintel.data.audit data/raw/<file>.csv
```

## Structure
- `src/careerintel/` data, nlp, graph, models, eval packages
- `notebooks/` audit, EDA, skill extraction, experiments
- `configs/` experiment configuration
- `docs/` research plan, dataset datasheet
- `app/` demo (Streamlit/FastAPI), built last

## Results
Results table will be added after experiments are run.
