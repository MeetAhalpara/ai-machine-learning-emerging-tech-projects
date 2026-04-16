"""
Blockchain Verification Ledger Pipeline
Author: Meet Ahalpara, Pratik Hirapara

This module implements a localized, cryptographic blockchain ledger designed to 
preserve a tamper-evident record of generated analysis reports. It mathematically 
links each execution block to its predecessor, ensuring structural integrity 
and supporting explainable verification of candidate assessments.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class LedgerError(Exception):
    """Raised when ledger operations fail."""


@dataclass(slots=True)
class LedgerBlock:
    """
    Represents one block in the SkillChain verification ledger.
    """

    index: int
    timestamp: str
    repository_name: str
    source_input: str
    report_hash: str
    report_summary_hash: str
    previous_hash: str
    block_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SkillChainLedger:
    """
    Orchestrates a cryptographic, blockchain-inspired ledger for immutable report verification.
    """

    # ==========================================
    # PHASE 1: LEDGER INITIALIZATION & ORCHESTRATION
    # ==========================================
    def __init__(self, ledger_file: str | Path) -> None:
        self.ledger_file = Path(ledger_file).expanduser().resolve()
        self.ledger_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.ledger_file.exists():
            self._write_chain([])

    def add_report(
        self,
        *,
        repository_name: str,
        source_input: str,
        report_json: str,
        metadata: dict[str, Any] | None = None,
    ) -> LedgerBlock:
        """
        Appends a newly generated executive report to the cryptographic ledger as an immutable block.

        Args:
            repository_name: Name of the analyzed repository.
            source_input: Original repository input.
            report_json: Final report serialized as stable JSON.
            metadata: Optional lightweight metadata for quick inspection.

        Returns:
            Newly created LedgerBlock.
        """
        if not repository_name.strip():
            raise LedgerError("repository_name cannot be empty.")
        if not source_input.strip():
            raise LedgerError("source_input cannot be empty.")
        if not report_json.strip():
            raise LedgerError("report_json cannot be empty.")

        chain = self.load_chain()
        previous_hash = chain[-1].block_hash if chain else "GENESIS"

        index = len(chain)
        timestamp = self._utc_now_iso()

        report_hash = self._sha256(report_json)
        report_summary_hash = self._sha256(
            json.dumps(
                {
                    "repository_name": repository_name,
                    "source_input": source_input,
                    "timestamp": timestamp,
                    "report_hash": report_hash,
                },
                sort_keys=True,
                ensure_ascii=False,
            )
        )

        block_payload = {
            "index": index,
            "timestamp": timestamp,
            "repository_name": repository_name,
            "source_input": source_input,
            "report_hash": report_hash,
            "report_summary_hash": report_summary_hash,
            "previous_hash": previous_hash,
            "metadata": metadata or {},
        }

        block_hash = self._sha256(
            json.dumps(block_payload, sort_keys=True, ensure_ascii=False)
        )

        block = LedgerBlock(
            index=index,
            timestamp=timestamp,
            repository_name=repository_name,
            source_input=source_input,
            report_hash=report_hash,
            report_summary_hash=report_summary_hash,
            previous_hash=previous_hash,
            block_hash=block_hash,
            metadata=metadata or {},
        )

        chain.append(block)
        self._write_chain(chain)

        logger.info(
            "Added block %s to ledger for repository '%s'.",
            block.index,
            block.repository_name,
        )

        return block

    # ==========================================
    # PHASE 2: BLOCKCHAIN VERIFICATION & INTEGRITY
    # ==========================================
    def load_chain(self) -> list[LedgerBlock]:
        """
        Deserializes and loads the entire cryptographic ledger chain from persistent local storage.
        """
        try:
            raw_data = json.loads(self.ledger_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LedgerError(f"Ledger file is corrupted or invalid JSON: {self.ledger_file}") from exc
        except OSError as exc:
            raise LedgerError(f"Unable to read ledger file: {self.ledger_file}") from exc

        if not isinstance(raw_data, list):
            raise LedgerError("Ledger file content must be a JSON list.")

        blocks: list[LedgerBlock] = []
        for item in raw_data:
            try:
                blocks.append(LedgerBlock(**item))
            except TypeError as exc:
                raise LedgerError("Ledger file contains invalid block structure.") from exc

        return blocks

    def verify_chain(self) -> tuple[bool, list[str]]:
        """
        Executes a rigorous cryptographic audit of the chain's integrity from the genesis block to the terminal node.

        Returns:
            (is_valid, issues)
        """
        issues: list[str] = []
        chain = self.load_chain()

        if not chain:
            return True, issues

        for i, block in enumerate(chain):
            expected_previous_hash = "GENESIS" if i == 0 else chain[i - 1].block_hash
            if block.previous_hash != expected_previous_hash:
                issues.append(
                    f"Block {block.index} has invalid previous hash linkage."
                )

            recomputed_payload = {
                "index": block.index,
                "timestamp": block.timestamp,
                "repository_name": block.repository_name,
                "source_input": block.source_input,
                "report_hash": block.report_hash,
                "report_summary_hash": block.report_summary_hash,
                "previous_hash": block.previous_hash,
                "metadata": block.metadata,
            }

            recomputed_hash = self._sha256(
                json.dumps(recomputed_payload, sort_keys=True, ensure_ascii=False)
            )

            if block.block_hash != recomputed_hash:
                issues.append(
                    f"Block {block.index} hash does not match its contents."
                )

        return len(issues) == 0, issues

    # ==========================================
    # PHASE 3: METRICS & STATE MANAGEMENT
    # ==========================================
    def get_latest_block(self) -> LedgerBlock | None:
        """
        Resolves the terminal (most recent) block currently appended to the ledger chain.
        """
        chain = self.load_chain()
        return chain[-1] if chain else None

    def chain_length(self) -> int:
        """
        Calculates the total aggregate number of immutable blocks within the ledger.
        """
        return len(self.load_chain())

    def _write_chain(self, chain: list[LedgerBlock]) -> None:
        """
        Serializes the updated ledger matrix and persists it securely to disk.
        """
        serializable = [asdict(block) for block in chain]
        try:
            self.ledger_file.write_text(
                json.dumps(serializable, indent=2, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:
            raise LedgerError(f"Unable to write ledger file: {self.ledger_file}") from exc

    # ==========================================
    # PHASE 4: CRYPTOGRAPHIC HASHING UTILITIES
    # ==========================================
    def _sha256(self, value: str) -> str:
        """
        Computes a mathematically deterministic SHA-256 cryptographic hash for a given string vector.
        """
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _utc_now_iso(self) -> str:
        """
        Resolves the current UTC temporal marker formatted to the ISO 8601 standard.
        """
        return datetime.now(timezone.utc).isoformat()