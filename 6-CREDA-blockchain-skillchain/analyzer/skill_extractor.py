"""
Skill Extraction & Candidate Inference Pipeline
Author: Meet Ahalpara, Pratik Hirapara

This module converts low-level technical signals into higher-level candidate skill categories.
The goal is to produce output that is meaningful for recruiters, instructors,
and demo audiences, while keeping the structural reasoning transparent and mathematically 
explainable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

from analyzer.tech_detector import DetectionEvidence, TechDetectionResult


logger = logging.getLogger(__name__)


class SkillExtractionError(Exception):
    """Raised when skill extraction cannot be completed."""


@dataclass(slots=True)
class ExtractedSkill:
    """
    Represents one recruiter-friendly skill inferred from repository evidence.
    """

    name: str
    confidence: str
    evidence_summary: list[str] = field(default_factory=list)
    supporting_files: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass(slots=True)
class SkillExtractionResult:
    """
    Final structured output of extracted skills from a repository.
    """

    extracted_skills: list[ExtractedSkill] = field(default_factory=list)
    inferred_roles: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)


class SkillExtractor:
    """
    Orchestrates the translation of static technical artifacts into high-level professional skills.
    """

    FRONTEND_FRAMEWORKS = {"React", "Next.js", "Vue", "Angular"}
    BACKEND_FRAMEWORKS = {"Node.js/Express", "Flask", "Django", "FastAPI", "Spring Boot"}
    MOBILE_FRAMEWORKS = {"Android"}
    DESKTOP_FRAMEWORKS = {"JavaFX"}

    # ==========================================
    # PHASE 1: SKILL EXTRACTION ORCHESTRATION
    # ==========================================
    def extract(self, detection: TechDetectionResult) -> SkillExtractionResult:
        """
        Synthesizes recruiter-friendly skills from low-level technical detection matrices.

        Args:
            detection: Raw technology detection result from TechDetector.

        Returns:
            SkillExtractionResult containing extracted skills, roles, strengths, and concerns.
        """
        if detection is None:
            raise SkillExtractionError("Detection result cannot be None.")

        result = SkillExtractionResult()
        skill_map: dict[str, ExtractedSkill] = {}

        logger.info("Starting structural skill extraction from detection results.")

        self._extract_frontend_skill(detection, skill_map)
        self._extract_backend_skill(detection, skill_map)
        self._extract_database_skill(detection, skill_map)
        self._extract_api_skill(detection, skill_map)
        self._extract_auth_skill(detection, skill_map)
        self._extract_testing_skill(detection, skill_map)
        self._extract_documentation_skill(detection, skill_map)
        self._extract_architecture_skill(detection, skill_map)
        self._extract_mobile_skill(detection, skill_map)
        self._extract_desktop_skill(detection, skill_map)
        self._extract_data_programming_skill(detection, skill_map)

        result.extracted_skills = sorted(skill_map.values(), key=lambda skill: skill.name)
        result.inferred_roles = self._infer_candidate_roles(detection)
        result.strengths = self._build_strengths(detection, result.extracted_skills)
        result.concerns = self._build_concerns(detection)

        logger.info(
            "Skill extraction completed. Skills extracted: %s, inferred roles: %s",
            len(result.extracted_skills),
            len(result.inferred_roles),
        )

        return result

    # ==========================================
    # PHASE 2: DOMAIN-SPECIFIC EXTRACTION RULES
    # ==========================================
    def _extract_frontend_skill(
        self,
        detection: TechDetectionResult,
        skill_map: dict[str, ExtractedSkill],
    ) -> None:
        """Evaluates frontend architectural evidence to construct an actionable skill profile."""
        frameworks = sorted(set(detection.frameworks) & self.FRONTEND_FRAMEWORKS)
        if not frameworks:
            return

        evidence = self._evidence_for_names(detection.evidence, frameworks)
        skill_map["Frontend Development"] = ExtractedSkill(
            name="Frontend Development",
            confidence=self._confidence_from_count(len(frameworks) + len(evidence)),
            evidence_summary=[
                f"Detected frontend frameworks: {', '.join(frameworks)}",
                "Repository structure suggests component-based or page-based UI development."
                if "Component-Based UI" in detection.architecture_indicators
                else "Frontend framework usage indicates UI development capability.",
            ],
            supporting_files=self._collect_files(evidence),
            rationale="The repository shows direct evidence of frontend application development.",
        )

    def _extract_backend_skill(
        self,
        detection: TechDetectionResult,
        skill_map: dict[str, ExtractedSkill],
    ) -> None:
        """Analyzes server-side frameworks and API patterns to extract backend capabilities."""
        frameworks = sorted(set(detection.frameworks) & self.BACKEND_FRAMEWORKS)
        api_present = bool(detection.api_indicators)
        architecture_present = "Layered Architecture" in detection.architecture_indicators or "MVC Structure" in detection.architecture_indicators

        if not frameworks and not api_present and not architecture_present:
            return

        evidence_names = frameworks + detection.api_indicators
        evidence = self._evidence_for_names(detection.evidence, evidence_names)

        summaries = []
        if frameworks:
            summaries.append(f"Detected backend frameworks: {', '.join(frameworks)}")
        if detection.api_indicators:
            summaries.append(f"API-related implementation found: {', '.join(detection.api_indicators)}")
        if architecture_present:
            summaries.append("Repository structure suggests backend organization or service layering.")

        skill_map["Backend Development"] = ExtractedSkill(
            name="Backend Development",
            confidence=self._confidence_from_count(len(frameworks) + len(detection.api_indicators) + (1 if architecture_present else 0)),
            evidence_summary=summaries,
            supporting_files=self._collect_files(evidence),
            rationale="The repository contains backend frameworks, API patterns, or architectural evidence associated with server-side development.",
        )

    def _extract_database_skill(
        self,
        detection: TechDetectionResult,
        skill_map: dict[str, ExtractedSkill],
    ) -> None:
        """Assesses database integration and structural data persistence signals."""
        if not detection.databases:
            return

        evidence = self._evidence_for_names(detection.evidence, detection.databases)
        skill_map["Database Integration"] = ExtractedSkill(
            name="Database Integration",
            confidence=self._confidence_from_count(len(detection.databases) + len(evidence)),
            evidence_summary=[
                f"Detected database technologies: {', '.join(detection.databases)}",
                "Repository contains code or dependencies suggesting persistent data storage integration.",
            ],
            supporting_files=self._collect_files(evidence),
            rationale="The candidate appears to have implemented or configured database-related functionality.",
        )

    def _extract_api_skill(
        self,
        detection: TechDetectionResult,
        skill_map: dict[str, ExtractedSkill],
    ) -> None:
        """Evaluates API implementation patterns and HTTP client usage."""
        if not detection.api_indicators:
            return

        evidence = self._evidence_for_names(detection.evidence, detection.api_indicators)
        skill_map["API Development"] = ExtractedSkill(
            name="API Development",
            confidence=self._confidence_from_count(len(detection.api_indicators) + len(evidence)),
            evidence_summary=[
                f"Detected API indicators: {', '.join(detection.api_indicators)}",
                "Repository includes patterns commonly associated with API creation or API consumption.",
            ],
            supporting_files=self._collect_files(evidence),
            rationale="The codebase shows evidence of endpoint implementation, HTTP integration, or API-related development.",
        )

    def _extract_auth_skill(
        self,
        detection: TechDetectionResult,
        skill_map: dict[str, ExtractedSkill],
    ) -> None:
        """Detects authentication, security, and session management implementations."""
        if not detection.auth_indicators:
            return

        evidence = self._evidence_for_names(detection.evidence, detection.auth_indicators)
        skill_map["Authentication & Security"] = ExtractedSkill(
            name="Authentication & Security",
            confidence=self._confidence_from_count(len(detection.auth_indicators) + len(evidence)),
            evidence_summary=[
                f"Detected authentication/security indicators: {', '.join(detection.auth_indicators)}",
                "Repository includes code patterns related to login, authorization, sessions, tokens, or OAuth flows.",
            ],
            supporting_files=self._collect_files(evidence),
            rationale="The repository suggests practical work involving user authentication or access control flows.",
        )

    def _extract_testing_skill(
        self,
        detection: TechDetectionResult,
        skill_map: dict[str, ExtractedSkill],
    ) -> None:
        """Quantifies the presence and maturity of automated testing disciplines."""
        if not detection.testing_indicators:
            return

        evidence = self._evidence_for_names(detection.evidence, detection.testing_indicators)
        skill_map["Testing & Validation"] = ExtractedSkill(
            name="Testing & Validation",
            confidence=self._confidence_from_count(len(detection.testing_indicators) + len(evidence)),
            evidence_summary=[
                f"Detected testing indicators: {', '.join(detection.testing_indicators)}",
                "Repository includes evidence of automated test files, frameworks, or validation practices.",
            ],
            supporting_files=self._collect_files(evidence),
            rationale="The codebase demonstrates some degree of testing or validation discipline.",
        )

    def _extract_documentation_skill(
        self,
        detection: TechDetectionResult,
        skill_map: dict[str, ExtractedSkill],
    ) -> None:
        """Measures the quality and physical presence of project documentation practices."""
        if not detection.documentation_indicators:
            return

        evidence = self._evidence_for_names(detection.evidence, detection.documentation_indicators)
        skill_map["Documentation Practices"] = ExtractedSkill(
            name="Documentation Practices",
            confidence=self._confidence_from_count(len(detection.documentation_indicators)),
            evidence_summary=[
                f"Detected documentation indicators: {', '.join(detection.documentation_indicators)}",
                "Repository appears to include documentation-oriented files or API documentation signals.",
            ],
            supporting_files=self._collect_files(evidence),
            rationale="Documentation presence improves maintainability and indicates stronger project communication practices.",
        )

    def _extract_architecture_skill(
        self,
        detection: TechDetectionResult,
        skill_map: dict[str, ExtractedSkill],
    ) -> None:
        """Evaluates the structural organization and architectural maturity of the codebase."""
        if not detection.architecture_indicators:
            return

        evidence = self._evidence_for_names(detection.evidence, detection.architecture_indicators)
        skill_map["Software Architecture & Project Organization"] = ExtractedSkill(
            name="Software Architecture & Project Organization",
            confidence=self._confidence_from_count(len(detection.architecture_indicators) + len(evidence)),
            evidence_summary=[
                f"Detected architecture indicators: {', '.join(detection.architecture_indicators)}",
                "Repository structure suggests intentional code organization and modular design.",
            ],
            supporting_files=self._collect_files(evidence),
            rationale="The project layout shows signs of architectural planning or organized code separation.",
        )

    def _extract_mobile_skill(
        self,
        detection: TechDetectionResult,
        skill_map: dict[str, ExtractedSkill],
    ) -> None:
        """Detects mobile application development frameworks and patterns."""
        frameworks = sorted(set(detection.frameworks) & self.MOBILE_FRAMEWORKS)
        if not frameworks:
            return

        evidence = self._evidence_for_names(detection.evidence, frameworks)
        skill_map["Mobile Application Development"] = ExtractedSkill(
            name="Mobile Application Development",
            confidence=self._confidence_from_count(len(frameworks) + len(evidence)),
            evidence_summary=[
                f"Detected mobile development frameworks: {', '.join(frameworks)}",
                "Repository appears to include Android/mobile application development patterns.",
            ],
            supporting_files=self._collect_files(evidence),
            rationale="The codebase contains direct signals of mobile app development work.",
        )

    def _extract_desktop_skill(
        self,
        detection: TechDetectionResult,
        skill_map: dict[str, ExtractedSkill],
    ) -> None:
        """Detects desktop application development frameworks and UI patterns."""
        frameworks = sorted(set(detection.frameworks) & self.DESKTOP_FRAMEWORKS)
        if not frameworks:
            return

        evidence = self._evidence_for_names(detection.evidence, frameworks)
        skill_map["Desktop Application Development"] = ExtractedSkill(
            name="Desktop Application Development",
            confidence=self._confidence_from_count(len(frameworks) + len(evidence)),
            evidence_summary=[
                f"Detected desktop development frameworks: {', '.join(frameworks)}",
                "Repository appears to include desktop UI application development patterns.",
            ],
            supporting_files=self._collect_files(evidence),
            rationale="The repository indicates hands-on work in desktop application development.",
        )

    def _extract_data_programming_skill(
        self,
        detection: TechDetectionResult,
        skill_map: dict[str, ExtractedSkill],
    ) -> None:
        """Evaluates foundational programming and data handling capabilities."""
        python_count = detection.languages.get("Python", 0)
        sql_count = detection.languages.get("SQL", 0)

        if python_count == 0 and sql_count == 0:
            return

        summaries = []
        confidence_score = 0

        if python_count > 0:
            summaries.append(f"Python files detected: {python_count}")
            confidence_score += 1

        if sql_count > 0:
            summaries.append(f"SQL files detected: {sql_count}")
            confidence_score += 1

        if detection.databases:
            summaries.append("Data-related repository signals are reinforced by database integration evidence.")
            confidence_score += 1

        skill_map["Programming & Data Handling"] = ExtractedSkill(
            name="Programming & Data Handling",
            confidence=self._confidence_from_count(confidence_score),
            evidence_summary=summaries,
            supporting_files=[],
            rationale="The repository shows programming and data-related implementation evidence.",
        )

    # ==========================================
    # PHASE 3: ROLE INFERENCE & EXECUTIVE SUMMARIZATION
    # ==========================================
    def _infer_candidate_roles(self, detection: TechDetectionResult) -> list[str]:
        """
        Infers likely candidate professional roles based on aggregated repository signals.
        """
        roles: set[str] = set()

        frontend = bool(set(detection.frameworks) & self.FRONTEND_FRAMEWORKS)
        backend = bool(set(detection.frameworks) & self.BACKEND_FRAMEWORKS or detection.api_indicators)
        mobile = bool(set(detection.frameworks) & self.MOBILE_FRAMEWORKS)
        desktop = bool(set(detection.frameworks) & self.DESKTOP_FRAMEWORKS)
        database = bool(detection.databases)

        if frontend and backend:
            roles.add("Full-Stack Developer")
        elif frontend:
            roles.add("Frontend Developer")
        elif backend:
            roles.add("Backend Developer")

        if mobile:
            roles.add("Mobile Developer")

        if desktop:
            roles.add("Desktop Application Developer")

        if database and not frontend:
            roles.add("Database/Application Developer")

        if detection.languages.get("Python", 0) > 0 and not frontend and not mobile:
            roles.add("Python Developer")

        return sorted(roles)

    def _build_strengths(
        self,
        detection: TechDetectionResult,
        skills: list[ExtractedSkill],
    ) -> list[str]:
        """
        Synthesizes a concise list of technical strengths from the extracted evidence matrix.
        """
        strengths: list[str] = []

        if len(skills) >= 4:
            strengths.append("Repository demonstrates a broad range of technical capabilities.")

        if detection.api_indicators and detection.databases:
            strengths.append("Project shows evidence of end-to-end application functionality, including APIs and data integration.")

        if detection.testing_indicators:
            strengths.append("Automated testing or validation practices are present.")

        if "Software Architecture & Project Organization" in {skill.name for skill in skills}:
            strengths.append("Repository structure suggests organized and modular development practices.")

        if detection.documentation_indicators:
            strengths.append("Documentation signals improve maintainability and project clarity.")

        return strengths

    def _build_concerns(self, detection: TechDetectionResult) -> list[str]:
        """
        Constructs balanced, objective concerns for recruiter-style evaluation summaries.
        """
        concerns: list[str] = []

        if not detection.testing_indicators:
            concerns.append("No strong automated testing evidence was detected.")

        if not detection.documentation_indicators:
            concerns.append("Documentation signals appear limited or minimal.")

        if not detection.architecture_indicators:
            concerns.append("Repository structure does not strongly indicate formal project architecture.")

        if not detection.api_indicators and not detection.databases and len(detection.frameworks) <= 1:
            concerns.append("Technical depth may be limited based on currently detected repository signals.")

        return concerns

    # ==========================================
    # PHASE 4: EVIDENCE FILTERING & METRICS
    # ==========================================
    def _evidence_for_names(
        self,
        evidence_items: Iterable[DetectionEvidence],
        names: Iterable[str],
    ) -> list[DetectionEvidence]:
        """
        Filters raw detection evidence to isolate specific targeted indicators.
        """
        wanted = set(names)
        return [item for item in evidence_items if item.name in wanted]

    def _collect_files(self, evidence_items: Iterable[DetectionEvidence]) -> list[str]:
        """
        Aggregates a sorted, mathematically unique list of supporting file paths.
        """
        return sorted({item.file_path for item in evidence_items if item.file_path})

    def _confidence_from_count(self, score: int) -> str:
        """
        Translates a numerical evidence count into a categorical, human-readable confidence tier.
        """
        if score >= 5:
            return "High"
        if score >= 3:
            return "Medium"
        return "Low"