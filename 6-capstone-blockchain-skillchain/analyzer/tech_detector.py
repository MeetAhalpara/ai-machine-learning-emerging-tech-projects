"""
Technology Detection & Static Analysis Pipeline
Author: Meet Ahalpara, Pratik Hirapara

This module operates as the core static analysis engine. It leverages pragmatic 
pattern matching, heuristic file inspections, and syntax evaluations to extract 
foundational technical signals (frameworks, databases, architectural patterns) 
from raw repository files.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


logger = logging.getLogger(__name__)


class TechDetectionError(Exception):
    """Raised when repository technology detection cannot be completed."""


@dataclass(slots=True)
class DetectionEvidence:
    """Represents a single piece of detection evidence."""

    category: str
    name: str
    file_path: str
    reason: str


@dataclass(slots=True)
class TechDetectionResult:
    """
    Structured result of repository technology detection.
    """

    languages: dict[str, int] = field(default_factory=dict)
    frameworks: list[str] = field(default_factory=list)
    databases: list[str] = field(default_factory=list)
    api_indicators: list[str] = field(default_factory=list)
    auth_indicators: list[str] = field(default_factory=list)
    testing_indicators: list[str] = field(default_factory=list)
    documentation_indicators: list[str] = field(default_factory=list)
    architecture_indicators: list[str] = field(default_factory=list)
    dependency_files: list[str] = field(default_factory=list)
    evidence: list[DetectionEvidence] = field(default_factory=list)
    file_count: int = 0
    scanned_text_files: int = 0


class TechDetector:
    """
    Detects technical characteristics of a repository using pragmatic static analysis.

    Current capabilities:
    - File-extension based language estimation
    - Dependency-file inspection
    - Keyword/pattern detection for frameworks, databases, APIs, auth, testing
    - Basic documentation and architecture signal detection

    This detector is intentionally rule-based and explainable, making it suitable
    for an academic prototype and recruiter-facing evidence summaries.
    """

    MAX_FILE_SIZE_BYTES = 1_000_000  # 1 MB
    MAX_TEXT_FILES_TO_SCAN = 250

    IGNORED_DIRECTORIES = {
        ".git",
        ".idea",
        ".vscode",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        "dist",
        "build",
        "target",
        "coverage",
        ".next",
        ".pytest_cache",
        ".mypy_cache",
        ".gradle",
        ".terraform",
    }

    TEXT_FILE_EXTENSIONS = {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".java",
        ".kt",
        ".kts",
        ".go",
        ".rb",
        ".php",
        ".cs",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".swift",
        ".sql",
        ".json",
        ".yml",
        ".yaml",
        ".xml",
        ".md",
        ".txt",
        ".properties",
        ".env",
        ".html",
        ".css",
        ".scss",
    }

    LANGUAGE_EXTENSIONS = {
        ".py": "Python",
        ".js": "JavaScript",
        ".jsx": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".java": "Java",
        ".kt": "Kotlin",
        ".kts": "Kotlin",
        ".go": "Go",
        ".rb": "Ruby",
        ".php": "PHP",
        ".cs": "C#",
        ".cpp": "C++",
        ".c": "C",
        ".h": "C/C++",
        ".hpp": "C++",
        ".swift": "Swift",
        ".sql": "SQL",
        ".html": "HTML",
        ".css": "CSS",
        ".scss": "SCSS",
    }

    FRAMEWORK_PATTERNS: dict[str, list[re.Pattern[str]]] = {
        "React": [
            re.compile(r'"react"\s*:', re.IGNORECASE),
            re.compile(r"from\s+['\"]react['\"]", re.IGNORECASE),
        ],
        "Next.js": [
            re.compile(r'"next"\s*:', re.IGNORECASE),
            re.compile(r"next/config", re.IGNORECASE),
        ],
        "Vue": [
            re.compile(r'"vue"\s*:', re.IGNORECASE),
            re.compile(r"from\s+['\"]vue['\"]", re.IGNORECASE),
        ],
        "Angular": [
            re.compile(r'"@angular/core"\s*:', re.IGNORECASE),
            re.compile(r"@NgModule", re.IGNORECASE),
        ],
        "Node.js/Express": [
            re.compile(r'"express"\s*:', re.IGNORECASE),
            re.compile(r"require\(['\"]express['\"]\)", re.IGNORECASE),
            re.compile(r"from\s+['\"]express['\"]", re.IGNORECASE),
        ],
        "Flask": [
            re.compile(r"\bfrom\s+flask\s+import\b", re.IGNORECASE),
            re.compile(r"\bFlask\(", re.IGNORECASE),
        ],
        "Django": [
            re.compile(r'"django"', re.IGNORECASE),
            re.compile(r"django\.urls", re.IGNORECASE),
            re.compile(r"INSTALLED_APPS", re.IGNORECASE),
        ],
        "FastAPI": [
            re.compile(r"\bfrom\s+fastapi\s+import\b", re.IGNORECASE),
            re.compile(r"\bFastAPI\(", re.IGNORECASE),
        ],
        "Spring Boot": [
            re.compile(r"spring-boot", re.IGNORECASE),
            re.compile(r"@SpringBootApplication", re.IGNORECASE),
            re.compile(r"@RestController", re.IGNORECASE),
        ],
        "JavaFX": [
            re.compile(r"javafx\.", re.IGNORECASE),
            re.compile(r"requires\s+javafx\.", re.IGNORECASE),
        ],
        "Android": [
            re.compile(r"com\.android\.", re.IGNORECASE),
            re.compile(r"androidx\.", re.IGNORECASE),
        ],
        "Streamlit": [
            re.compile(r"\bimport\s+streamlit\b", re.IGNORECASE),
            re.compile(r"\bst\.", re.IGNORECASE),
        ],
    }

    DATABASE_PATTERNS: dict[str, list[re.Pattern[str]]] = {
        "MongoDB": [
            re.compile(r"mongoose", re.IGNORECASE),
            re.compile(r"mongodb", re.IGNORECASE),
            re.compile(r"pymongo", re.IGNORECASE),
        ],
        "PostgreSQL": [
            re.compile(r"postgres", re.IGNORECASE),
            re.compile(r"psycopg2", re.IGNORECASE),
            re.compile(r"postgresql", re.IGNORECASE),
        ],
        "MySQL": [
            re.compile(r"mysql", re.IGNORECASE),
            re.compile(r"pymysql", re.IGNORECASE),
            re.compile(r"mysql-connector", re.IGNORECASE),
        ],
        "SQLite": [
            re.compile(r"sqlite3", re.IGNORECASE),
            re.compile(r"\.db\b", re.IGNORECASE),
        ],
        "Oracle": [
            re.compile(r"oracle", re.IGNORECASE),
            re.compile(r"cx_Oracle", re.IGNORECASE),
            re.compile(r"jdbc:oracle", re.IGNORECASE),
        ],
        "SQL Server": [
            re.compile(r"sqlserver", re.IGNORECASE),
            re.compile(r"mssql", re.IGNORECASE),
            re.compile(r"jdbc:sqlserver", re.IGNORECASE),
        ],
        "Firebase": [
            re.compile(r"firebase", re.IGNORECASE),
            re.compile(r"Firestore", re.IGNORECASE),
        ],
    }

    API_PATTERNS: dict[str, list[re.Pattern[str]]] = {
        "REST Endpoints": [
            re.compile(r"@GetMapping|@PostMapping|@PutMapping|@DeleteMapping", re.IGNORECASE),
            re.compile(r"\bapp\.(get|post|put|delete|patch)\(", re.IGNORECASE),
            re.compile(r"\brouter\.(get|post|put|delete|patch)\(", re.IGNORECASE),
            re.compile(r"@app\.(get|post|put|delete|patch)", re.IGNORECASE),
        ],
        "HTTP Client Usage": [
            re.compile(r"\brequests\.", re.IGNORECASE),
            re.compile(r"\bfetch\(", re.IGNORECASE),
            re.compile(r"\baxios\.", re.IGNORECASE),
            re.compile(r"\bHttpClient\b", re.IGNORECASE),
        ],
        "GraphQL": [
            re.compile(r"graphql", re.IGNORECASE),
            re.compile(r"Apollo", re.IGNORECASE),
            re.compile(r"typeDefs|resolvers", re.IGNORECASE),
        ],
    }

    AUTH_PATTERNS: dict[str, list[re.Pattern[str]]] = {
        "JWT Authentication": [
            re.compile(r"\bjwt\b", re.IGNORECASE),
            re.compile(r"jsonwebtoken", re.IGNORECASE),
            re.compile(r"Bearer\s+", re.IGNORECASE),
        ],
        "Session Authentication": [
            re.compile(r"session", re.IGNORECASE),
            re.compile(r"express-session", re.IGNORECASE),
        ],
        "OAuth": [
            re.compile(r"oauth", re.IGNORECASE),
            re.compile(r"client_id", re.IGNORECASE),
            re.compile(r"redirect_uri", re.IGNORECASE),
        ],
        "Login/Auth Flow": [
            re.compile(r"\blogin\b", re.IGNORECASE),
            re.compile(r"\bauthenticate\b", re.IGNORECASE),
            re.compile(r"\bauthoriz", re.IGNORECASE),
            re.compile(r"\bsignin\b|\bsignup\b", re.IGNORECASE),
        ],
    }

    TEST_PATTERNS: dict[str, list[re.Pattern[str]]] = {
        "Pytest": [
            re.compile(r"\bpytest\b", re.IGNORECASE),
            re.compile(r"def\s+test_", re.IGNORECASE),
        ],
        "Unittest": [
            re.compile(r"\bunittest\b", re.IGNORECASE),
            re.compile(r"class\s+\w+\(.*TestCase.*\)", re.IGNORECASE),
        ],
        "Jest": [
            re.compile(r"\bjest\b", re.IGNORECASE),
            re.compile(r"\bdescribe\(", re.IGNORECASE),
            re.compile(r"\bit\(", re.IGNORECASE),
        ],
        "JUnit": [
            re.compile(r"\bJUnit\b", re.IGNORECASE),
            re.compile(r"@Test", re.IGNORECASE),
        ],
    }

    DOCUMENTATION_PATTERNS: dict[str, list[re.Pattern[str]]] = {
        "README Present": [
            re.compile(r"^README(\..+)?$", re.IGNORECASE),
        ],
        "API Documentation": [
            re.compile(r"swagger", re.IGNORECASE),
            re.compile(r"openapi", re.IGNORECASE),
        ],
        "Project Documentation": [
            re.compile(r"docs", re.IGNORECASE),
            re.compile(r"documentation", re.IGNORECASE),
        ],
    }

    ARCHITECTURE_PATH_PATTERNS: dict[str, list[re.Pattern[str]]] = {
        "Layered Architecture": [
            re.compile(r"controllers?", re.IGNORECASE),
            re.compile(r"services?", re.IGNORECASE),
            re.compile(r"repositories?", re.IGNORECASE),
            re.compile(r"models?", re.IGNORECASE),
        ],
        "MVC Structure": [
            re.compile(r"views?", re.IGNORECASE),
            re.compile(r"controllers?", re.IGNORECASE),
            re.compile(r"models?", re.IGNORECASE),
        ],
        "Component-Based UI": [
            re.compile(r"components?", re.IGNORECASE),
            re.compile(r"pages?", re.IGNORECASE),
            re.compile(r"layouts?", re.IGNORECASE),
        ],
        "Testing Structure": [
            re.compile(r"tests?", re.IGNORECASE),
            re.compile(r"__tests__", re.IGNORECASE),
        ],
    }

    DEPENDENCY_FILENAMES = {
        "requirements.txt",
        "pyproject.toml",
        "package.json",
        "package-lock.json",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "go.mod",
        "composer.json",
        "Gemfile",
    }

    # ==========================================
    # PHASE 1: DETECTION ORCHESTRATION
    # ==========================================
    def detect(self, repo_path: str | Path) -> TechDetectionResult:
        """
        Analyzes the raw repository matrix and orchestrates the extraction of structured technical signals.

        Args:
            repo_path: Local repository directory path.

        Returns:
            TechDetectionResult containing detected technologies and evidence.

        Raises:
            TechDetectionError: If the repository path is invalid or unreadable.
        """
        root = Path(repo_path).expanduser().resolve()
        if not root.exists():
            raise TechDetectionError(f"Repository path does not exist: {root}")
        if not root.is_dir():
            raise TechDetectionError(f"Repository path is not a directory: {root}")

        result = TechDetectionResult()
        language_counter: Counter[str] = Counter()

        logger.info("Starting tech detection for repository: %s", root)

        all_files = list(self._iter_repository_files(root))
        result.file_count = len(all_files)

        for file_path in all_files:
            relative_path = self._safe_relative_path(file_path, root)
            file_name_lower = file_path.name.lower()

            self._detect_language_from_extension(file_path, language_counter)
            self._detect_dependency_file(file_path, result, relative_path)
            self._detect_documentation_by_filename(file_path, result, relative_path)
            self._detect_architecture_by_path(relative_path, result)

        text_files = [path for path in all_files if self._should_scan_text_file(path)]
        text_files = text_files[: self.MAX_TEXT_FILES_TO_SCAN]
        result.scanned_text_files = len(text_files)

        for file_path in text_files:
            relative_path = self._safe_relative_path(file_path, root)
            content = self._read_text_file(file_path)
            if content is None:
                continue

            self._detect_frameworks(content, relative_path, result)
            self._detect_databases(content, relative_path, result)
            self._detect_apis(content, relative_path, result)
            self._detect_auth(content, relative_path, result)
            self._detect_testing(content, relative_path, result)
            self._detect_documentation_in_content(content, relative_path, result)

        result.languages = dict(language_counter.most_common())
        result.frameworks = sorted(set(result.frameworks))
        result.databases = sorted(set(result.databases))
        result.api_indicators = sorted(set(result.api_indicators))
        result.auth_indicators = sorted(set(result.auth_indicators))
        result.testing_indicators = sorted(set(result.testing_indicators))
        result.documentation_indicators = sorted(set(result.documentation_indicators))
        result.architecture_indicators = sorted(set(result.architecture_indicators))
        result.dependency_files = sorted(set(result.dependency_files))

        logger.info(
            "Tech detection completed. Files: %s, scanned text files: %s",
            result.file_count,
            result.scanned_text_files,
        )

        return result

    # ==========================================
    # PHASE 2: STRUCTURAL & PATH-BASED DETECTION
    # ==========================================
    def _iter_repository_files(self, root: Path) -> Iterable[Path]:
        """
        Yields repository file vectors while systematically pruning irrelevant or heavy build directories.
        """
        for path in root.rglob("*"):
            if any(part in self.IGNORED_DIRECTORIES for part in path.parts):
                continue
            if path.is_file():
                yield path

    def _detect_language_from_extension(self, file_path: Path, counter: Counter[str]) -> None:
        """
        Mathematically infers foundational programming languages by mapping file extension vectors.
        """
        language = self.LANGUAGE_EXTENSIONS.get(file_path.suffix.lower())
        if language:
            counter[language] += 1

    def _detect_dependency_file(
        self,
        file_path: Path,
        result: TechDetectionResult,
        relative_path: str,
    ) -> None:
        """
        Isolates and records critical build manifests and dependency management files.
        """
        if file_path.name in self.DEPENDENCY_FILENAMES:
            result.dependency_files.append(relative_path)
            result.evidence.append(
                DetectionEvidence(
                    category="dependency_file",
                    name=file_path.name,
                    file_path=relative_path,
                    reason="Dependency or build configuration file detected.",
                )
            )

    def _detect_documentation_by_filename(
        self,
        file_path: Path,
        result: TechDetectionResult,
        relative_path: str,
    ) -> None:
        """
        Detects documentation infrastructure indicators based on file naming patterns.
        """
        for label, patterns in self.DOCUMENTATION_PATTERNS.items():
            if any(pattern.search(file_path.name) for pattern in patterns):
                result.documentation_indicators.append(label)
                result.evidence.append(
                    DetectionEvidence(
                        category="documentation",
                        name=label,
                        file_path=relative_path,
                        reason=f"Documentation-related file detected: {file_path.name}",
                    )
                )

    def _detect_architecture_by_path(
        self,
        relative_path: str,
        result: TechDetectionResult,
    ) -> None:
        """
        Identifies macro-level software architecture topologies by analyzing directory structure.
        """
        for label, patterns in self.ARCHITECTURE_PATH_PATTERNS.items():
            if any(pattern.search(relative_path) for pattern in patterns):
                result.architecture_indicators.append(label)
                result.evidence.append(
                    DetectionEvidence(
                        category="architecture",
                        name=label,
                        file_path=relative_path,
                        reason="Architecture-related directory or file path pattern detected.",
                    )
                )

    # ==========================================
    # PHASE 3: HEURISTIC CONTENT ANALYSIS
    # ==========================================
    def _detect_frameworks(
        self,
        content: str,
        relative_path: str,
        result: TechDetectionResult,
    ) -> None:
        """
        Executes regex analysis to identify framework imports and usage within raw file contents.
        """
        self._apply_pattern_map(
            category="framework",
            pattern_map=self.FRAMEWORK_PATTERNS,
            content=content,
            relative_path=relative_path,
            target_list=result.frameworks,
            result=result,
        )

    def _detect_databases(
        self,
        content: str,
        relative_path: str,
        result: TechDetectionResult,
    ) -> None:
        """
        Scans content arrays to isolate database connectivity and querying libraries.
        """
        self._apply_pattern_map(
            category="database",
            pattern_map=self.DATABASE_PATTERNS,
            content=content,
            relative_path=relative_path,
            target_list=result.databases,
            result=result,
        )

    def _detect_apis(
        self,
        content: str,
        relative_path: str,
        result: TechDetectionResult,
    ) -> None:
        """
        Analyzes HTTP client implementations and REST/GraphQL endpoint definitions.
        """
        self._apply_pattern_map(
            category="api",
            pattern_map=self.API_PATTERNS,
            content=content,
            relative_path=relative_path,
            target_list=result.api_indicators,
            result=result,
        )

    def _detect_auth(
        self,
        content: str,
        relative_path: str,
        result: TechDetectionResult,
    ) -> None:
        """
        Identifies security, session management, and authentication token protocols.
        """
        self._apply_pattern_map(
            category="authentication",
            pattern_map=self.AUTH_PATTERNS,
            content=content,
            relative_path=relative_path,
            target_list=result.auth_indicators,
            result=result,
        )

    def _detect_testing(
        self,
        content: str,
        relative_path: str,
        result: TechDetectionResult,
    ) -> None:
        """
        Locates automated testing frameworks and validation logic.
        """
        self._apply_pattern_map(
            category="testing",
            pattern_map=self.TEST_PATTERNS,
            content=content,
            relative_path=relative_path,
            target_list=result.testing_indicators,
            result=result,
        )

    def _detect_documentation_in_content(
        self,
        content: str,
        relative_path: str,
        result: TechDetectionResult,
    ) -> None:
        """
        Identifies embedded API documentation tools like Swagger or OpenAPI.
        """
        if "swagger" in content.lower() or "openapi" in content.lower():
            label = "API Documentation"
            result.documentation_indicators.append(label)
            result.evidence.append(
                DetectionEvidence(
                    category="documentation",
                    name=label,
                    file_path=relative_path,
                    reason="Swagger/OpenAPI-related documentation content detected.",
                )
            )

    # ==========================================
    # PHASE 4: UTILITY & FILE PARSING HELPERS
    # ==========================================
    def _apply_pattern_map(
        self,
        category: str,
        pattern_map: dict[str, list[re.Pattern[str]]],
        content: str,
        relative_path: str,
        target_list: list[str],
        result: TechDetectionResult,
    ) -> None:
        """
        Executes a rigorous regex pattern matching sequence across text vectors to record specific evidence.
        """
        for label, patterns in pattern_map.items():
            for pattern in patterns:
                if pattern.search(content):
                    target_list.append(label)
                    result.evidence.append(
                        DetectionEvidence(
                            category=category,
                            name=label,
                            file_path=relative_path,
                            reason=f"Matched pattern '{pattern.pattern}' in file content.",
                        )
                    )
                    break

    def _should_scan_text_file(self, file_path: Path) -> bool:
        """
        Calculates file bounds to determine whether a target file is safe for deep text analysis.
        """
        if file_path.suffix.lower() not in self.TEXT_FILE_EXTENSIONS and file_path.name not in self.DEPENDENCY_FILENAMES:
            return False

        try:
            return file_path.stat().st_size <= self.MAX_FILE_SIZE_BYTES
        except OSError:
            return False

    def _read_text_file(self, file_path: Path) -> str | None:
        """
        Safely reads and decodes text files utilizing fault-tolerant encoding fallbacks.
        """
        encodings = ("utf-8", "utf-8-sig", "latin-1")

        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding, errors="ignore")
            except OSError:
                logger.warning("Failed to read file: %s", file_path)
                return None
            except UnicodeDecodeError:
                continue

        return None

    def _safe_relative_path(self, file_path: Path, root: Path) -> str:
        """
        Computes a stable, cross-platform relative path string for architectural evidence recording.
        """
        try:
            return str(file_path.relative_to(root)).replace("\\", "/")
        except ValueError:
            return str(file_path).replace("\\", "/")