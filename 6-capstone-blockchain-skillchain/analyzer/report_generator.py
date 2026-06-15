"""
Executive Report Generation Pipeline
Author: Meet Ahalpara, Pratik Hirapara

This module synthesizes the outputs from the entire static analysis pipeline 
(technology detection, skill extraction, and proficiency estimation) into a 
comprehensive, highly structured, and recruiter-friendly executive report.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from analyzer.proficiency_estimator import (
    ProficiencyArea,
    ProficiencyEstimationResult,
)
from analyzer.skill_extractor import (
    ExtractedSkill,
    SkillExtractionResult,
)
from analyzer.tech_detector import (
    DetectionEvidence,
    TechDetectionResult,
)


logger = logging.getLogger(__name__)


class ReportGenerationError(Exception):
    """Raised when a final analysis report cannot be generated."""


@dataclass(slots=True)
class SkillChainReport:
    """
    Final report structure for a repository analysis.

    This is the main object that can be:
    - displayed in the UI
    - serialized to JSON
    - hashed for blockchain verification
    """

    generated_at: str
    source_input: str
    source_type: str
    repository_name: str

    repository_summary: dict[str, Any] = field(default_factory=dict)
    detected_technologies: dict[str, Any] = field(default_factory=dict)
    extracted_skills: dict[str, Any] = field(default_factory=dict)
    proficiency_assessment: dict[str, Any] = field(default_factory=dict)

    recruiter_summary: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    evidence_count: int = 0


class ReportGenerator:
    """
    Creates the final recruiter-friendly report from analysis pipeline outputs.
    """

    # ==========================================
    # PHASE 1: REPORT GENERATION ORCHESTRATION
    # ==========================================
    def generate(
        self,
        *,
        source_input: str,
        source_type: str,
        repository_name: str,
        detection: TechDetectionResult,
        skills: SkillExtractionResult,
        proficiency: ProficiencyEstimationResult,
    ) -> SkillChainReport:
        """
        Synthesizes the final executive SkillChain report by aggregating inputs from 
        the detection, extraction, and estimation layers.

        Args:
            source_input: Original repository input (URL or local path).
            source_type: Type of input source, e.g. 'github' or 'local'.
            repository_name: Display name of the repository.
            detection: Raw detection result.
            skills: Extracted skill result.
            proficiency: Proficiency estimation result.

        Returns:
            SkillChainReport

        Raises:
            ReportGenerationError: If required data is missing.
        """
        if not source_input.strip():
            raise ReportGenerationError("source_input cannot be empty.")
        if not source_type.strip():
            raise ReportGenerationError("source_type cannot be empty.")
        if not repository_name.strip():
            raise ReportGenerationError("repository_name cannot be empty.")
        if detection is None:
            raise ReportGenerationError("detection cannot be None.")
        if skills is None:
            raise ReportGenerationError("skills cannot be None.")
        if proficiency is None:
            raise ReportGenerationError("proficiency cannot be None.")

        logger.info("Generating final SkillChain report for repository: %s", repository_name)

        report = SkillChainReport(
            generated_at=self._utc_now_iso(),
            source_input=source_input,
            source_type=source_type,
            repository_name=repository_name,
            repository_summary=self._build_repository_summary(detection),
            detected_technologies=self._build_detected_technologies(detection),
            extracted_skills=self._build_extracted_skills(skills),
            proficiency_assessment=self._build_proficiency_assessment(proficiency),
            recruiter_summary=self._build_recruiter_summary(skills, proficiency),
            strengths=skills.strengths,
            concerns=skills.concerns,
            limitations=proficiency.limitations,
            evidence_count=len(detection.evidence),
        )

        logger.info("Final SkillChain report generated successfully.")
        return report

    # ==========================================
    # PHASE 2: SERIALIZATION & EXPORT
    # ==========================================
    def to_dict(self, report: SkillChainReport) -> dict[str, Any]:
        """
        Transforms the structured SkillChainReport object into a native Python dictionary 
        matrix for UI rendering and ledger hashing.

        This method is useful for:
        - UI rendering
        - JSON export
        - Downstream blockchain hashing
        """
        return asdict(report)

    def to_json(self, report: SkillChainReport, *, indent: int = 2) -> str:
        """
        Serializes the SkillChainReport dictionary into a stable, UTF-8 encoded JSON string for persistent storage.
        """
        return json.dumps(
            self.to_dict(report),
            indent=indent,
            ensure_ascii=False,
            sort_keys=True,
        )

    # ==========================================
    # PHASE 3: STRUCTURAL AGGREGATION
    # ==========================================
    def _build_repository_summary(self, detection: TechDetectionResult) -> dict[str, Any]:
        """
        Aggregates raw detection metrics into a high-level statistical summary of the repository structure.
        """
        dominant_languages = list(detection.languages.keys())[:5]

        return {
            "file_count": detection.file_count,
            "scanned_text_files": detection.scanned_text_files,
            "dominant_languages": dominant_languages,
            "dependency_files": detection.dependency_files,
            "framework_count": len(detection.frameworks),
            "database_count": len(detection.databases),
            "api_indicator_count": len(detection.api_indicators),
            "auth_indicator_count": len(detection.auth_indicators),
            "testing_indicator_count": len(detection.testing_indicators),
            "documentation_indicator_count": len(detection.documentation_indicators),
            "architecture_indicator_count": len(detection.architecture_indicators),
        }

    def _build_detected_technologies(self, detection: TechDetectionResult) -> dict[str, Any]:
        """
        Constructs the comprehensive detected technology matrix for the final report.
        """
        return {
            "languages": detection.languages,
            "frameworks": detection.frameworks,
            "databases": detection.databases,
            "api_indicators": detection.api_indicators,
            "auth_indicators": detection.auth_indicators,
            "testing_indicators": detection.testing_indicators,
            "documentation_indicators": detection.documentation_indicators,
            "architecture_indicators": detection.architecture_indicators,
            "evidence": [self._serialize_evidence(item) for item in detection.evidence],
        }

    def _build_extracted_skills(self, skills: SkillExtractionResult) -> dict[str, Any]:
        """
        Compiles the extracted professional skills and inferred candidate roles into a structured presentation format.
        """
        return {
            "skills": [self._serialize_skill(skill) for skill in skills.extracted_skills],
            "inferred_roles": skills.inferred_roles,
        }

    def _build_proficiency_assessment(
        self,
        proficiency: ProficiencyEstimationResult,
    ) -> dict[str, Any]:
        """
        Synthesizes the proficiency estimation scores and categorical levels into a unified assessment block.
        """
        return {
            "overall_score": proficiency.overall_score,
            "overall_level": proficiency.overall_level,
            "repository_complexity": proficiency.repository_complexity,
            "technical_maturity": proficiency.technical_maturity,
            "areas": [self._serialize_area(area) for area in proficiency.areas],
            "summary": proficiency.summary,
        }

    def _build_recruiter_summary(
        self,
        skills: SkillExtractionResult,
        proficiency: ProficiencyEstimationResult,
    ) -> list[str]:
        """
        Generates high-impact, recruiter-focused executive summary statements designed 
        for rapid candidate screening.
        """
        summary: list[str] = []

        summary.append(
            f"Estimated overall repository proficiency: {proficiency.overall_level}."
        )
        summary.append(
            f"Repository complexity appears {proficiency.repository_complexity.lower()} and technical maturity appears {proficiency.technical_maturity.lower()}."
        )

        if skills.inferred_roles:
            summary.append(
                f"Likely candidate fit: {', '.join(skills.inferred_roles)}."
            )

        if skills.extracted_skills:
            top_skills = [skill.name for skill in skills.extracted_skills[:4]]
            summary.append(
                f"Key demonstrated areas include: {', '.join(top_skills)}."
            )

        return summary

    # ==========================================
    # PHASE 4: DATA SERIALIZATION HELPERS
    # ==========================================
    def _serialize_evidence(self, evidence: DetectionEvidence) -> dict[str, str]:
        """
        Translates a single DetectionEvidence object into a JSON-compatible dictionary.
        """
        return {
            "category": evidence.category,
            "name": evidence.name,
            "file_path": evidence.file_path,
            "reason": evidence.reason,
        }

    def _serialize_skill(self, skill: ExtractedSkill) -> dict[str, Any]:
        """
        Translates a single ExtractedSkill object into a JSON-compatible dictionary.
        """
        return {
            "name": skill.name,
            "confidence": skill.confidence,
            "evidence_summary": skill.evidence_summary,
            "supporting_files": skill.supporting_files,
            "rationale": skill.rationale,
        }

    def _serialize_area(self, area: ProficiencyArea) -> dict[str, Any]:
        """
        Translates a single ProficiencyArea object into a JSON-compatible dictionary.
        """
        return {
            "name": area.name,
            "score": area.score,
            "level": area.level,
            "justification": area.justification,
        }

    def _utc_now_iso(self) -> str:
        """
        Resolves the current UTC temporal marker formatted to the ISO 8601 standard.
        """
        return datetime.now(timezone.utc).isoformat()