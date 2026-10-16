"""CareerIntel hybrid job recommendation ranker.

Combines:
1. Transformer-based semantic similarity
2. Profile-to-job skill overlap
3. Job-title relevance

Scores are ranking signals, not probabilities of employment.
"""

import ast
import re
from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class HybridRanker:
    """Rank jobs using semantic, skill, and title relevance."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        semantic_weight: float = 0.60,
        skill_weight: float = 0.30,
        title_weight: float = 0.10,
    ):
        weights = [
            semantic_weight,
            skill_weight,
            title_weight,
        ]

        if any(weight < 0 for weight in weights):
            raise ValueError("Weights cannot be negative.")

        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Weights must sum to 1.0.")

        self.model = SentenceTransformer(model_name)

        self.semantic_weight = semantic_weight
        self.skill_weight = skill_weight
        self.title_weight = title_weight

        self.job_ids = []
        self.job_titles = []
        self.job_skills = []
        self.job_embeddings = None

    @staticmethod
    def _clean(value) -> str:
        """Convert missing values to empty strings and normalize spaces."""
        if value is None:
            return ""

        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return ""

        return re.sub(r"\s+", " ", str(value)).strip()

    @classmethod
    def _parse_skills(cls, value) -> set[str]:
        """Parse list-like skill data or delimited text."""

        if isinstance(value, (list, tuple, set)):
            items = value

        else:
            text = cls._clean(value)

            if not text:
                return set()

            try:
                parsed = ast.literal_eval(text)

                if isinstance(parsed, (list, tuple, set)):
                    items = parsed
                elif isinstance(parsed, str):
                    items = [parsed]
                else:
                    items = re.split(r"[,;|]", text)

            except (ValueError, SyntaxError):
                items = re.split(r"[,;|]", text)

        aliases = {
            "py": "python",
            "ml": "machine learning",
            "data visualisation": "data visualization",
            "scikit learn": "scikit-learn",
        }

        skills = set()

        for item in items:
            skill = cls._clean(item).casefold()

            if skill:
                skills.add(aliases.get(skill, skill))

        return skills

    @classmethod
    def _combine_job_text(cls, row) -> str:
        """Build searchable text from the available job fields."""

        parts = [
            cls._clean(row.get("title", "")),
            cls._clean(row.get("job_description", "")),
            cls._clean(row.get("job_text", "")),
            cls._clean(row.get("responsibilities", "")),
            cls._clean(row.get("minimum_qualifications", "")),
            cls._clean(row.get("preferred_qualifications", "")),
            cls._clean(row.get("experience_level", "")),
            cls._clean(row.get("industry", "")),
        ]

        return " ".join(part for part in parts if part)

    def fit(
        self,
        jobs: Sequence[dict],
    ) -> "HybridRanker":
        """Prepare job text, skills, and semantic embeddings."""

        if not jobs:
            raise ValueError("At least one job is required.")

        self.job_ids = []
        self.job_titles = []
        self.job_skills = []

        job_texts = []

        for job in jobs:
            job_id = self._clean(job.get("job_id", ""))
            title = self._clean(job.get("title", ""))

            if not job_id or not title:
                continue

            combined_skills = set()

            for column in [
                "normalized_skills",
                "parsed_skills",
                "skills_required",
            ]:
                combined_skills.update(
                    self._parse_skills(job.get(column, ""))
                )

            self.job_ids.append(job_id)
            self.job_titles.append(title)
            self.job_skills.append(combined_skills)

            text = self._combine_job_text(job)

            if combined_skills:
                text += " " + " ".join(sorted(combined_skills))

            job_texts.append(text)

        if not job_texts:
            raise ValueError("No valid jobs with IDs and titles were found.")

        self.job_embeddings = self.model.encode(
            job_texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return self

    def rank(
        self,
        profile_text: str,
        profile_skills: Sequence[str],
        top_k: int = 10,
    ) -> list[dict]:
        """Return the highest-ranked jobs with component scores."""

        if self.job_embeddings is None:
            raise RuntimeError("Call fit() before rank().")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        profile_text = self._clean(profile_text)

        if not profile_text:
            raise ValueError("Profile text cannot be empty.")

        user_skills = self._parse_skills(profile_skills)

        profile_embedding = self.model.encode(
            [profile_text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        # Normalized embeddings allow cosine similarity through dot product.
        semantic_scores = np.asarray(
            self.job_embeddings @ profile_embedding
        ).ravel()

        # Convert cosine similarity from [-1, 1] to [0, 1].
        semantic_scores = np.clip(
            (semantic_scores + 1.0) / 2.0,
            0.0,
            1.0,
        )

        skill_scores = []

        for required_skills in self.job_skills:
            if not required_skills:
                skill_scores.append(0.0)
                continue

            overlap = user_skills.intersection(required_skills)

            skill_scores.append(
                len(overlap) / len(required_skills)
            )

        skill_scores = np.asarray(skill_scores)

        profile_words = set(
            re.findall(r"[a-z0-9+#.]+", profile_text.casefold())
        )

        title_scores = []

        for title in self.job_titles:
            title_words = set(
                re.findall(r"[a-z0-9+#.]+", title.casefold())
            )

            if not title_words:
                title_scores.append(0.0)
                continue

            overlap = profile_words.intersection(title_words)

            title_scores.append(
                len(overlap) / len(title_words)
            )

        title_scores = np.asarray(title_scores)

        hybrid_scores = (
            self.semantic_weight * semantic_scores
            + self.skill_weight * skill_scores
            + self.title_weight * title_scores
        )

        order = np.argsort(-hybrid_scores, kind="stable")[:top_k]

        results = []

        for index in order:
            results.append({
                "job_id": self.job_ids[index],
                "job_title": self.job_titles[index],
                "hybrid_score": float(hybrid_scores[index]),
                "semantic_score": float(semantic_scores[index]),
                "skill_score": float(skill_scores[index]),
                "title_score": float(title_scores[index]),
                "matched_skills": sorted(
                    user_skills.intersection(self.job_skills[index])
                ),
            })

        return results