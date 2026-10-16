# Research plan
## Research questions
- RQ1: Do semantic embeddings outperform TF-IDF/BM25 for job matching (NDCG@10)?
- RQ2: Do graph-derived skill relations improve recall for sparse profiles?
- RQ3: Do skill-gap predictions beat a frequency-only baseline?
- RQ4 (optional): Does simulated feedback improve ranking?

## Ground truth layers
1. Synthetic profiles with held-out target jobs (proxy)
2. ~300 human-judged profile-job pairs (graded relevance, report agreement)
3. Skill-masking for gap validation

## Experiments
| ID | Model | Status |
|----|-------|--------|
| M0 | TF-IDF / BM25 | todo |
| M1 | Sentence-transformer embeddings | todo |
| M2 | Hybrid (embedding + skill overlap) | todo |
| M3 | Graph-enhanced | todo |
| M4 | Learning-to-rank | todo |

## Results
(Fill in only with real measured numbers.)
