"""
Proficiency Estimation Pipeline
Author: Meet Ahalpara, Pratik Hirapara

This module estimates repository proficiency using explainable, rule-based scoring.
It evaluates technical detection results and extracted skills to generate a 
structured assessment of a candidate's technical maturity and repository complexity.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from analyzer.skill_extractor import SkillExtractionResult
from analyzer.tech_detector import TechDetectionResult


logger = logging.getLogger(__name__)


class ProficiencyEstimationError(Exception):
    """Raised when repository proficiency estimation cannot be completed."""


@dataclass(slots=True)
class ProficiencyArea:
    """
    Represents an estimated proficiency score for one technical area.
    """

    name: str
    score: int
    level: str
    justification: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ProficiencyEstimationResult:
    """
    Final structured output of repository proficiency estimation.
    """

    overall_score: int
    overall_level: str
    repository_complexity: str
    technical_maturity: str
    areas: list[ProficiencyArea] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


class ProficiencyEstimator:
    """
    Estimates repository proficiency using explainable, rule-based scoring.

    This estimator is intentionally conservative:
    - It estimates demonstrated proficiency based on repository evidence only.
    - It does not claim real-world mastery or full candidate capability.
    - It is designed for academic prototypes and recruiter-friendly summaries.
    """

    # ==========================================
    # PHASE 1: PROFICIENCY ESTIMATION ORCHESTRATION
    # ==========================================
    def estimate(
        self,
        detection: TechDetectionResult,
        skills: SkillExtractionResult,
    ) -> ProficiencyEstimationResult:
        """
        Estimate repository proficiency based on detection and extracted skills.

        Args:
            detection: Raw technology detection result.
            skills: Recruiter-facing extracted skill result.

        Returns:
            ProficiencyEstimationResult with overall and area-specific estimates.
        """
        if detection is None:
            raise ProficiencyEstimationError("Detection result cannot be None.")
        if skills is None:
            raise ProficiencyEstimationError("Skill extraction result cannot be None.")

        logger.info("Starting structural proficiency estimation pipeline.")

        areas = [
            self._score_frontend(detection),
            self._score_backend(detection),
            self._score_database(detection),
            self._score_testing(detection),
            self._score_architecture(detection),
            self._score_documentation(detection),
        ]

        overall_score = self._calculate_overall_score(areas, detection, skills)
        overall_level = self._score_to_level(overall_score)
        repository_complexity = self._estimate_repository_complexity(detection, skills)
        technical_maturity = self._estimate_technical_maturity(detection, skills, areas)
        summary = self._build_summary(detection, skills, overall_level, repository_complexity, technical_maturity)
        limitations = self._build_limitations(detection)

        logger.info(
            "Proficiency estimation complete. Overall score: %s, level: %s",
            overall_score,
            overall_level,
        )

        return ProficiencyEstimationResult(
            overall_score=overall_score,
            overall_level=overall_level,
            repository_complexity=repository_complexity,
            technical_maturity=technical_maturity,
            areas=areas,
            summary=summary,
            limitations=limitations,
        )

    # ==========================================
    # PHASE 2: DOMAIN-SPECIFIC SCORING HEURISTICS
    # ==========================================
    def _score_frontend(self, detection: TechDetectionResult) -> ProficiencyArea:
        """
        Evaluates frontend architectural evidence to compute a normalized proficiency score.
        """
        score = 0
        justification: list[str] = []

        frontend_frameworks = {"React", "Next.js", "Vue", "Angular"}
        found = sorted(set(detection.frameworks) & frontend_frameworks)

        if found:
            score += 3
            justification.append(f"Frontend frameworks detected: {', '.join(found)}")

        if "Component-Based UI" in detection.architecture_indicators:
            score += 2
            justification.append("Component-based UI project structure detected.")

        ts_files = detection.languages.get("TypeScript", 0)
        js_files = detection.languages.get("JavaScript", 0)
        if ts_files > 0 or js_files > 0:
            score += 1
            justification.append("Frontend-oriented language files were detected.")

        return ProficiencyArea(
            name="Frontend Development",
            score=min(score, 10),
            level=self._score_to_level(score),
            justification=justification or ["No strong frontend evidence detected."],
        )

    def _score_backend(self, detection: TechDetectionResult) -> ProficiencyArea:
        """
        Analyzes server-side frameworks and API patterns to gauge backend capability.
        """
        score = 0
        justification: list[str] = []

        backend_frameworks = {"Node.js/Express", "Flask", "Django", "FastAPI", "Spring Boot"}
        found = sorted(set(detection.frameworks) & backend_frameworks)

        if found:
            score += 3
            justification.append(f"Backend frameworks detected: {', '.join(found)}")

        if detection.api_indicators:
            score += 3
            justification.append(f"API indicators detected: {', '.join(detection.api_indicators)}")

        if "Layered Architecture" in detection.architecture_indicators or "MVC Structure" in detection.architecture_indicators:
            score += 2
            justification.append("Repository structure suggests backend/service organization.")

        backend_languages = detection.languages.get("Python", 0) + detection.languages.get("Java", 0) + detection.languages.get("JavaScript", 0) + detection.languages.get("TypeScript", 0)
        if backend_languages > 0:
            score += 1
            justification.append("Backend-capable programming languages detected.")

        return ProficiencyArea(
            name="Backend Development",
            score=min(score, 10),
            level=self._score_to_level(score),
            justification=justification or ["No strong backend evidence detected."],
        )

    def _score_database(self, detection: TechDetectionResult) -> ProficiencyArea:
        """
        Assesses database integration and structural data persistence signals.
        """
        score = 0
        justification: list[str] = []

        if detection.databases:
            db_count = len(detection.databases)
            score += min(4, db_count + 1)
            justification.append(f"Database technologies detected: {', '.join(detection.databases)}")

        if detection.languages.get("SQL", 0) > 0:
            score += 2
            justification.append("SQL files detected in the repository.")

        if detection.databases and detection.api_indicators:
            score += 2
            justification.append("Database and API signals together suggest integrated application behavior.")

        return ProficiencyArea(
            name="Database Integration",
            score=min(score, 10),
            level=self._score_to_level(score),
            justification=justification or ["No strong database integration evidence detected."],
        )

    def _score_testing(self, detection: TechDetectionResult) -> ProficiencyArea:
        """
        Quantifies the presence and maturity of automated testing disciplines.
        """
        score = 0
        justification: list[str] = []

        if detection.testing_indicators:
            test_count = len(detection.testing_indicators)
            score += min(5, test_count + 1)
            justification.append(f"Testing indicators detected: {', '.join(detection.testing_indicators)}")

        if "Testing Structure" in detection.architecture_indicators:
            score += 2
            justification.append("Testing-oriented directory structure detected.")

        return ProficiencyArea(
            name="Testing & Validation",
            score=min(score, 10),
            level=self._score_to_level(score),
            justification=justification or ["No strong testing or validation evidence detected."],
        )

    def _score_architecture(self, detection: TechDetectionResult) -> ProficiencyArea:
        """
        Evaluates the structural organization and architectural maturity of the codebase.
        """
        score = 0
        justification: list[str] = []

        if detection.architecture_indicators:
            arch_count = len(detection.architecture_indicators)
            score += min(5, arch_count + 1)
            justification.append(f"Architecture indicators detected: {', '.join(detection.architecture_indicators)}")

        if len(detection.dependency_files) >= 2:
            score += 1
            justification.append("Repository includes multiple dependency/build configuration files.")

        if detection.file_count >= 20:
            score += 1
            justification.append("Repository contains enough files to suggest non-trivial structure.")

        return ProficiencyArea(
            name="Software Architecture & Organization",
            score=min(score, 10),
            level=self._score_to_level(score),
            justification=justification or ["No strong architecture or organization evidence detected."],
        )

    def _score_documentation(self, detection: TechDetectionResult) -> ProficiencyArea:
        """
        Measures the quality and physical presence of project documentation practices.
        """
        score = 0
        justification: list[str] = []

        if detection.documentation_indicators:
            doc_count = len(detection.documentation_indicators)
            score += min(4, doc_count + 1)
            justification.append(f"Documentation indicators detected: {', '.join(detection.documentation_indicators)}")

        if any(path.lower().startswith("readme") or "/readme" in path.lower() for path in detection.dependency_files + detection.documentation_indicators):
            score += 1

        return ProficiencyArea(
            name="Documentation Practices",
            score=min(score, 10),
            level=self._score_to_level(score),
            justification=justification or ["No strong documentation evidence detected."],
        )

    # ==========================================
    # PHASE 3: MACRO-LEVEL AGGREGATION & METRICS
    # ==========================================
    def _calculate_overall_score(
        self,
        areas: list[ProficiencyArea],
        detection: TechDetectionResult,
        skills: SkillExtractionResult,
    ) -> int:
        """
        Calculates a balanced overall score mathematically utilizing area metrics and repository breadth.
        """
        if not areas:
            return 0

        area_average = sum(area.score for area in areas) / len(areas)
        breadth_bonus = min(2.0, len(skills.extracted_skills) * 0.3)
        repository_bonus = 0.0

        if detection.file_count >= 15:
            repository_bonus += 0.5
        if detection.file_count >= 40:
            repository_bonus += 0.5
        if detection.scanned_text_files >= 10:
            repository_bonus += 0.5

        final_score = round(min(10, area_average + breadth_bonus + repository_bonus))
        return int(final_score)

    def _estimate_repository_complexity(
        self,
        detection: TechDetectionResult,
        skills: SkillExtractionResult,
    ) -> str:
        """
        Estimates repository complexity strictly based on volume metrics and feature density signals.
        """
        complexity_score = 0

        if detection.file_count >= 10:
            complexity_score += 1
        if detection.file_count >= 30:
            complexity_score += 1
        if len(detection.frameworks) >= 2:
            complexity_score += 1
        if detection.databases:
            complexity_score += 1
        if detection.api_indicators:
            complexity_score += 1
        if detection.auth_indicators:
            complexity_score += 1
        if len(skills.extracted_skills) >= 4:
            complexity_score += 1

        if complexity_score >= 6:
            return "High"
        if complexity_score >= 3:
            return "Medium"
        return "Low"

    def _estimate_technical_maturity(
        self,
        detection: TechDetectionResult,
        skills: SkillExtractionResult,
        areas: list[ProficiencyArea],
    ) -> str:
        """
        Evaluates technical maturity heavily weighting best practices like testing and architecture.
        """
        maturity_score = 0

        if detection.testing_indicators:
            maturity_score += 1
        if detection.documentation_indicators:
            maturity_score += 1
        if detection.architecture_indicators:
            maturity_score += 1
        if detection.dependency_files:
            maturity_score += 1
        if len(skills.concerns) <= 1:
            maturity_score += 1
        if sum(1 for area in areas if area.score >= 5) >= 3:
            maturity_score += 1

        if maturity_score >= 5:
            return "Strong"
        if maturity_score >= 3:
            return "Moderate"
        return "Early-Stage"

    # ==========================================
    # PHASE 4: EXECUTIVE REPORTING & TEXT GENERATION
    # ==========================================
    def _build_summary(
        self,
        detection: TechDetectionResult,
        skills: SkillExtractionResult,
        overall_level: str,
        repository_complexity: str,
        technical_maturity: str,
    ) -> list[str]:
        """
        Synthesizes concise, recruiter-style summary strings for executive reporting.
        """
        summary: list[str] = []

        summary.append(
            f"Repository demonstrates an estimated overall {overall_level.lower()} level based on detected code evidence."
        )
        summary.append(
            f"Repository complexity appears {repository_complexity.lower()} with technical maturity assessed as {technical_maturity.lower()}."
        )

        if skills.inferred_roles:
            summary.append(f"Likely candidate profile(s): {', '.join(skills.inferred_roles)}.")

        if detection.frameworks:
            summary.append(f"Detected frameworks/technologies include: {', '.join(detection.frameworks[:5])}.")

        if detection.databases:
            summary.append(f"Database-related implementation evidence detected: {', '.join(detection.databases)}.")

        return summary

    def _build_limitations(self, detection: TechDetectionResult) -> list[str]:
        """
        Constructs transparent algorithmic limitations to guarantee honest, bounded reporting.
        """
        limitations = [
            "This estimate is based on repository evidence only and does not measure all real-world skills.",
            "Proficiency levels are inferred from detected code patterns, structure, and project signals.",
        ]

        if not detection.testing_indicators:
            limitations.append("Testing maturity may be underestimated because little or no automated testing evidence was detected.")

        if not detection.documentation_indicators:
            limitations.append("Documentation quality may be underestimated because limited documentation signals were found.")

        return limitations

    def _score_to_level(self, score: int | float) -> str:
        """
        Maps a computed mathematical score index to a conservative string-based proficiency level.
        """
        if score >= 8:
            return "Intermediate-Advanced"
        if score >= 6:
            return "Intermediate"
        if score >= 3:
            return "Beginner-Intermediate"
        return "Beginner"