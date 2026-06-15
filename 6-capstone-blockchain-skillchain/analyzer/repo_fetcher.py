"""
Repository Fetching & Initialization Pipeline
Author: Meet Ahalpara, Pratik Hirapara

This module operates as the foundational data ingestion layer for the SkillChain 
analysis pipeline. It intelligently orchestrates the retrieval of target codebases 
by validating local directory paths or securely cloning public GitHub repositories 
into isolated temporary environments for downstream static analysis.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


logger = logging.getLogger(__name__)


class RepoFetchError(Exception):
    """Raised when a repository cannot be fetched or prepared for analysis."""


@dataclass(slots=True)
class FetchResult:
    """
    Represents the prepared repository location for analysis.

    Attributes:
        repo_path: Local filesystem path to the repository/codebase.
        source_type: Indicates whether the source came from a local path or a cloned URL.
        original_input: The original user-provided value.
        cleanup_required: True when the repository was cloned into a temporary directory.
        temp_dir: Temporary directory path to delete after analysis, if applicable.
    """

    repo_path: Path
    source_type: str
    original_input: str
    cleanup_required: bool = False
    temp_dir: Path | None = None


class RepoFetcher:
    """
    Prepares a repository for analysis.

    Supported inputs:
    - Local directory path
    - Public GitHub repository URL

    Behavior:
    - Local paths are validated and returned directly.
    - GitHub URLs are shallow-cloned into a temporary directory.
    - Temporary clones can be cleaned up after analysis.
    """

    SUPPORTED_GITHUB_HOSTS = {"github.com", "www.github.com"}

    def __init__(self, clone_base_dir: str | Path | None = None) -> None:
        """
        Initialize the RepoFetcher.

        Args:
            clone_base_dir: Optional directory under which temporary clones are created.
                            If not provided, the system temp directory is used.
        """
        self.clone_base_dir = Path(clone_base_dir).resolve() if clone_base_dir else None

    # ==========================================
    # PHASE 1: PIPELINE ORCHESTRATION
    # ==========================================
    def fetch(self, source: str) -> FetchResult:
        """
        Prepares a repository for static analysis by dynamically determining the input protocol.

        Args:
            source: A local directory path or GitHub repository URL.

        Returns:
            FetchResult containing the prepared repository path.

        Raises:
            RepoFetchError: If the source is invalid or cannot be prepared.
        """
        cleaned_source = (source or "").strip()
        if not cleaned_source:
            raise RepoFetchError("Repository source cannot be empty.")

        if self._is_github_url(cleaned_source):
            return self._clone_from_github(cleaned_source)

        return self._use_local_path(cleaned_source)

    def cleanup(self, result: FetchResult) -> None:
        """
        Safely destructs isolated temporary environments post-analysis to prevent storage leakage.

        Args:
            result: FetchResult returned by fetch().

        This method is safe to call multiple times.
        """
        if not result.cleanup_required or result.temp_dir is None:
            return

        try:
            if result.temp_dir.exists():
                shutil.rmtree(result.temp_dir)
                logger.info("Cleaned up temporary repository at %s", result.temp_dir)
        except Exception as exc:
            logger.warning("Failed to clean up temp directory %s: %s", result.temp_dir, exc)

    # ==========================================
    # PHASE 2: EXTRACTION PROTOCOLS
    # ==========================================
    def _use_local_path(self, source: str) -> FetchResult:
        """
        Validates and mathematically resolves a localized repository directory path.
        """
        repo_path = Path(source).expanduser().resolve()

        if not repo_path.exists():
            raise RepoFetchError(f"Local path does not exist: {repo_path}")

        if not repo_path.is_dir():
            raise RepoFetchError(f"Local path is not a directory: {repo_path}")

        logger.info("Using local repository path: %s", repo_path)

        return FetchResult(
            repo_path=repo_path,
            source_type="local",
            original_input=source,
            cleanup_required=False,
            temp_dir=None,
        )

    def _clone_from_github(self, url: str) -> FetchResult:
        """
        Executes a secure, shallow-clone operation of a remote GitHub repository into an isolated temporary matrix.
        """
        self._ensure_git_available()

        temp_dir = self._create_temp_dir()
        repo_dir = temp_dir / "repo"

        clone_command = [
            "git",
            "clone",
            "--depth",
            "1",
            url,
            str(repo_dir),
        ]

        logger.info("Cloning GitHub repository: %s", url)

        try:
            completed = subprocess.run(
                clone_command,
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            self._safe_remove_dir(temp_dir)
            raise RepoFetchError(f"Failed to execute git clone: {exc}") from exc

        if completed.returncode != 0:
            self._safe_remove_dir(temp_dir)
            stderr = completed.stderr.strip() or "Unknown git clone error."
            raise RepoFetchError(f"Unable to clone repository. Git returned: {stderr}")

        if not repo_dir.exists() or not repo_dir.is_dir():
            self._safe_remove_dir(temp_dir)
            raise RepoFetchError("Repository clone completed, but the target directory was not created.")

        logger.info("Repository cloned successfully to %s", repo_dir)

        return FetchResult(
            repo_path=repo_dir,
            source_type="github",
            original_input=url,
            cleanup_required=True,
            temp_dir=temp_dir,
        )

    # ==========================================
    # PHASE 3: ENVIRONMENT & SYSTEM VERIFICATION
    # ==========================================
    def _ensure_git_available(self) -> None:
        """
        Verifies the presence and executability of the Git version control binary within the host system path.
        """
        try:
            completed = subprocess.run(
                ["git", "--version"],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RepoFetchError("Git is not installed or not available in PATH.") from exc

        if completed.returncode != 0:
            raise RepoFetchError("Git is installed but could not be executed correctly.")

    def _create_temp_dir(self) -> Path:
        """
        Constructs an isolated, universally unique temporary directory to house cloned architectural assets.
        """
        if self.clone_base_dir:
            self.clone_base_dir.mkdir(parents=True, exist_ok=True)
            temp_dir = Path(
                tempfile.mkdtemp(prefix="skillchain_", dir=str(self.clone_base_dir))
            ).resolve()
        else:
            temp_dir = Path(tempfile.mkdtemp(prefix="skillchain_")).resolve()

        return temp_dir

    def _safe_remove_dir(self, path: Path) -> None:
        """
        Executes a best-effort, fault-tolerant directory deletion sequence during pipeline teardown.
        """
        try:
            if path.exists():
                shutil.rmtree(path)
        except Exception as exc:
            logger.warning("Failed to remove temporary directory %s: %s", path, exc)

    def _is_github_url(self, value: str) -> bool:
        """
        Employs heuristic parsing to systematically verify if the provided string vector is a valid GitHub URL.
        """
        try:
            parsed = urlparse(value)
        except Exception:
            return False

        if parsed.scheme not in {"http", "https"}:
            return False

        if parsed.netloc.lower() not in self.SUPPORTED_GITHUB_HOSTS:
            return False

        path_parts = [part for part in parsed.path.strip("/").split("/") if part]
        if len(path_parts) < 2:
            return False

        return True