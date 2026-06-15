"""
SkillChain Streamlit UI Application
Author: Meet Ahalpara, Pratik Hirapara

This module serves as the primary user interface for the SkillChain analysis pipeline.
It orchestrates the backend analysis modules (fetching, detection, extraction, 
estimation, and ledger verification) and renders the results in a comprehensive, 
interactive, and recruiter-friendly web dashboard using the Streamlit framework.
"""
from __future__ import annotations
from dataclasses import asdict
import json
import logging
from pathlib import Path
from urllib.parse import urlparse
import streamlit as st

from analyzer.proficiency_estimator import (
    ProficiencyEstimationError,
    ProficiencyEstimator,
)
from analyzer.report_generator import ReportGenerationError, ReportGenerator
from analyzer.repo_fetcher import FetchResult, RepoFetchError, RepoFetcher
from analyzer.skill_extractor import SkillExtractionError, SkillExtractor
from analyzer.tech_detector import TechDetectionError, TechDetector
from blockchain.ledger import LedgerError, SkillChainLedger


# ==========================================
# PHASE 1: APPLICATION CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="SkillChain",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


LEDGER_FILE = Path("data/skillchain_ledger.json")
REPORT_EXPORT_DIR = Path("data/reports")
REPORT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

CUSTOM_CSS = """
<style>
:root {
    --bg: #0b1020;
    --panel: rgba(18, 24, 45, 0.80);
    --panel-2: rgba(25, 32, 58, 0.92);
    --text: #f3f6ff;
    --muted: #9aa7c7;
    --accent: #7c9cff;
    --accent-2: #5eead4;
    --border: rgba(255,255,255,0.08);
    --good: #34d399;
    --warn: #fbbf24;
    --bad: #f87171;
    --shadow: 0 20px 50px rgba(0,0,0,0.30);
    --radius: 22px;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(124,156,255,0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(94,234,212,0.12), transparent 22%),
        linear-gradient(180deg, #0b1020 0%, #0d1326 45%, #0a0f1d 100%);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(11,16,32,0.95), rgba(14,20,40,0.95));
    border-right: 1px solid var(--border);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

.hero {
    background: linear-gradient(135deg, rgba(23,32,64,0.92), rgba(15,22,42,0.90));
    border: 1px solid var(--border);
    border-radius: 28px;
    padding: 2rem 2rem 1.5rem 2rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.25rem;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--text);
    margin-bottom: 0.4rem;
}

.hero-subtitle {
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.6;
    max-width: 900px;
}

.section-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.15rem 1.15rem 1rem 1.15rem;
    box-shadow: var(--shadow);
    backdrop-filter: blur(12px);
}

.metric-card {
    background: linear-gradient(180deg, rgba(22,29,53,0.95), rgba(16,22,41,0.95));
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1rem 1rem 0.9rem 1rem;
    box-shadow: var(--shadow);
    height: 100%;
}

.metric-label {
    color: var(--muted);
    font-size: 0.85rem;
    margin-bottom: 0.25rem;
}

.metric-value {
    color: var(--text);
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}

.metric-sub {
    color: var(--muted);
    font-size: 0.85rem;
    margin-top: 0.35rem;
}

.pill {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin: 0.2rem 0.35rem 0.2rem 0;
    border: 1px solid var(--border);
    background: rgba(124,156,255,0.14);
    color: #dce6ff;
}

.good-pill {
    background: rgba(52,211,153,0.14);
    color: #c8ffef;
}

.warn-pill {
    background: rgba(251,191,36,0.14);
    color: #fff1bf;
}

.bad-pill {
    background: rgba(248,113,113,0.14);
    color: #ffd4d4;
}

.small-heading {
    color: var(--text);
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
}

.muted {
    color: var(--muted);
}

.report-box {
    background: rgba(14, 20, 39, 0.95);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1rem;
}

div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 0.8rem 1rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.6rem;
    margin-bottom: 0.7rem;
}

.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 0.45rem 0.95rem;
    border: 1px solid var(--border);
}

.stTabs [aria-selected="true"] {
    background: rgba(124,156,255,0.16) !important;
}

div.stButton > button,
div.stDownloadButton > button {
    border-radius: 14px;
    border: 1px solid rgba(124,156,255,0.28);
    background: linear-gradient(135deg, rgba(124,156,255,0.18), rgba(94,234,212,0.12));
    color: white;
    font-weight: 700;
    padding: 0.65rem 1rem;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover {
    border-color: rgba(124,156,255,0.45);
    background: linear-gradient(135deg, rgba(124,156,255,0.26), rgba(94,234,212,0.18));
}

hr {
    border-color: rgba(255,255,255,0.08);
}

code {
    color: #dce6ff;
}
/* Hide Streamlit text_input help text ("Press Enter to apply") - robust for all versions */
span[data-testid="stTextInputInstructions"],
div[data-testid="stTextInputInstructions"],
.stTextInputInstructions {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    min-height: 0 !important;
    max-height: 0 !important;
    font-size: 0 !important;
    line-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}
</style>
"""

# ==========================================
# PHASE 2: UI COMPONENT HELPERS
# ==========================================
def inject_custom_css() -> None:
    """Injects custom CSS to apply a polished, professional visual theme to the application."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_hero() -> None:
    """Renders the main application title and descriptive hero section."""
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">SkillChain</div>
            <div class="hero-subtitle">
                AI-powered GitHub code analysis and blockchain-backed verification for candidate screening.
                Analyze repository evidence, detect demonstrated technical skills, estimate proficiency,
                and generate a recruiter-friendly report in one place.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, subtext: str = "") -> None:
    """Renders a single, styled metric card for the overview dashboard."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pills(items: list[str], pill_class: str = "pill") -> None:
    """Renders a list of string labels as styled, inline pill components."""
    if not items:
        st.markdown('<span class="muted">No items detected.</span>', unsafe_allow_html=True)
        return

    html = "".join(f'<span class="{pill_class}">{item}</span>' for item in items)
    st.markdown(html, unsafe_allow_html=True)


def extract_repository_name(source_input: str, fetch_result: FetchResult) -> str:
    """Resolves a clean, human-readable repository name from the source input."""
    if fetch_result.source_type == "github":
        parsed = urlparse(source_input)
        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if len(parts) >= 2:
            return parts[1].replace(".git", "")
    return fetch_result.repo_path.name


def save_report_json(repository_name: str, report_json: str) -> Path:
    """Serializes the final report JSON to a local file for download."""
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in repository_name)
    export_path = REPORT_EXPORT_DIR / f"{safe_name}_report.json"
    export_path.write_text(report_json, encoding="utf-8")
    return export_path


def badge_class_for_level(level: str) -> str:
    """Maps proficiency level strings to corresponding CSS pill classes for color-coding."""
    normalized = level.lower()
    if "advanced" in normalized or normalized == "intermediate":
        return "good-pill"
    if "beginner" in normalized and "intermediate" not in normalized:
        return "bad-pill"
    return "warn-pill"


def render_list_section(title: str, items: list[str], empty_text: str) -> None:
    """Renders a simple markdown bulleted list section with a title."""
    st.markdown(f"#### {title}")
    if not items:
        st.caption(empty_text)
        return
    for item in items:
        st.markdown(f"- {item}")


# ==========================================
# PHASE 3: CORE ANALYSIS PIPELINE
# ==========================================
def run_analysis(source_input: str) -> dict:
    """
    Orchestrates the full SkillChain static analysis pipeline from fetching to verification.
    
    Returns:
        Dictionary containing fetch_result, report, report_json, ledger_block, chain_validity.
    """
    fetcher = RepoFetcher()
    detector = TechDetector()
    extractor = SkillExtractor()
    estimator = ProficiencyEstimator()
    report_generator = ReportGenerator()
    ledger = SkillChainLedger(LEDGER_FILE)

    fetch_result: FetchResult | None = None

    try:
        fetch_result = fetcher.fetch(source_input)
        repository_name = extract_repository_name(source_input, fetch_result)

        detection = detector.detect(fetch_result.repo_path)
        extracted_skills = extractor.extract(detection)
        proficiency = estimator.estimate(detection, extracted_skills)

        report = report_generator.generate(
            source_input=source_input,
            source_type=fetch_result.source_type,
            repository_name=repository_name,
            detection=detection,
            skills=extracted_skills,
            proficiency=proficiency,
        )

        report_json = report_generator.to_json(report)
        report_export_path = save_report_json(repository_name, report_json)

        ledger_block = ledger.add_report(
            repository_name=repository_name,
            source_input=source_input,
            report_json=report_json,
            metadata={
                "overall_level": report.proficiency_assessment.get("overall_level"),
                "repository_complexity": report.proficiency_assessment.get("repository_complexity"),
                "technical_maturity": report.proficiency_assessment.get("technical_maturity"),
                "evidence_count": report.evidence_count,
            },
        )

        chain_valid, chain_issues = ledger.verify_chain()

        return {
            "fetch_result": fetch_result,
            "report": report,
            "report_json": report_json,
            "report_export_path": report_export_path,
            "ledger_block": ledger_block,
            "chain_valid": chain_valid,
            "chain_issues": chain_issues,
            "ledger_length": ledger.chain_length(),
        }

    finally:
        if fetch_result is not None:
            fetcher.cleanup(fetch_result)


# ==========================================
# PHASE 4: UI RENDERING ORCHESTRATION
# ==========================================
def render_sidebar() -> tuple[str, bool]:
    """Renders the primary sidebar controls for user input and application guidance."""
    st.sidebar.markdown("## Analysis Controls")
    mode = st.sidebar.radio(
        "Choose input type",
        options=["GitHub Repository URL", "Local Project Path"],
        index=0,
        key="repo_input_type"
    )

    default_value = ""
    placeholder = (
        "https://github.com/owner/repository"
        if mode == "GitHub Repository URL"
        else "C:/Users/YourName/Documents/project-folder"
    )


    def trigger_analysis():
        st.session_state["auto_analyze"] = True

    # Use a unique key for the text input based on the selected mode
    input_key = f"repo_input_{mode.replace(' ', '_').lower()}"
    source_input = st.sidebar.text_input(
        "",
        value=default_value,
        placeholder=placeholder,
        key=input_key,
        on_change=trigger_analysis,
        label_visibility="collapsed"
    )
    # Always sync the current input to a canonical key for downstream logic
    st.session_state["repo_input"] = source_input

    st.sidebar.markdown("---")
    st.sidebar.markdown("### What SkillChain evaluates")
    st.sidebar.markdown(
        """
        - Languages and frameworks  
        - Database integration  
        - API implementation  
        - Authentication signals  
        - Testing and validation  
        - Documentation quality  
        - Architecture and organization  
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Proficiency is estimated from repository evidence and should be interpreted as an informed technical summary, not an absolute measure of candidate ability."
    )


    # If Enter was pressed in the input, trigger analysis automatically
    analyze_clicked = st.sidebar.button("Analyze Repository", use_container_width=True) or st.session_state.get("auto_analyze", False)
    # Reset the auto trigger after use
    if st.session_state.get("auto_analyze", False):
        st.session_state["auto_analyze"] = False

    return source_input, analyze_clicked


def render_overview(report_dict: dict) -> None:
    """Renders the high-level overview section with key proficiency metrics."""
    repo_summary = report_dict["repository_summary"]
    proficiency = report_dict["proficiency_assessment"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(
            "Overall Level",
            proficiency["overall_level"],
            "Estimated from repository evidence",
        )
    with c2:
        render_metric_card(
            "Complexity",
            proficiency["repository_complexity"],
            f"{repo_summary['file_count']} files detected",
        )
    with c3:
        render_metric_card(
            "Technical Maturity",
            proficiency["technical_maturity"],
            f"{repo_summary['scanned_text_files']} text files scanned",
        )
    with c4:
        render_metric_card(
            "Evidence Count",
            str(report_dict["evidence_count"]),
            "Detection signals recorded",
        )


def render_detected_technologies(report_dict: dict) -> None:
    """Renders the 'Detected Technologies' tab, showcasing raw technical signals."""
    tech = report_dict["detected_technologies"]
    repo_summary = report_dict["repository_summary"]

    col1, col2 = st.columns([1.2, 1.0])

    with col1:
        st.markdown("#### Technology Signals")
        render_pills(list(tech["languages"].keys()))
        st.markdown("#### Frameworks")
        render_pills(tech["frameworks"])
        st.markdown("#### Databases")
        render_pills(tech["databases"])

    with col2:
        st.markdown("#### Development Signals")
        render_pills(tech["api_indicators"])
        st.markdown("#### Security & Authentication")
        render_pills(tech["auth_indicators"])
        st.markdown("#### Architecture")
        render_pills(tech["architecture_indicators"])

    st.markdown("#### Repository Summary")
    a, b, c = st.columns(3)
    a.metric("Dependency Files", len(repo_summary["dependency_files"]))
    b.metric("Testing Indicators", repo_summary["testing_indicator_count"])
    c.metric("Documentation Indicators", repo_summary["documentation_indicator_count"])

    if repo_summary["dependency_files"]:
        with st.expander("View dependency/build files"):
            for dep in repo_summary["dependency_files"]:
                st.markdown(f"- `{dep}`")


def render_extracted_skills(report_dict: dict) -> None:
    """Renders the 'Extracted Skills' tab, detailing recruiter-friendly skill categories."""
    skill_section = report_dict["extracted_skills"]
    skills = skill_section["skills"]
    roles = skill_section["inferred_roles"]

    top_left, top_right = st.columns([1.2, 1.0])

    with top_left:
        st.markdown("#### Inferred Candidate Roles")
        render_pills(roles, "good-pill" if roles else "pill")

    with top_right:
        st.markdown("#### Number of Extracted Skill Areas")
        st.metric("Skill Categories", len(skills))

    st.markdown("#### Extracted Skills")

    if not skills:
        st.info("No recruiter-facing skill categories were extracted from the repository.")
        return

    for skill in skills:
        with st.container():
            st.markdown(
                f"""
                <div class="section-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem;">
                        <div>
                            <div class="small-heading">{skill['name']}</div>
                        </div>
                        <div>
                            <span class="pill {badge_class_for_level(skill['confidence'])}">{skill['confidence']} Confidence</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for line in skill["evidence_summary"]:
                st.markdown(f"- {line}")
            if skill["rationale"]:
                st.caption(f"Rationale: {skill['rationale']}")
            if skill["supporting_files"]:
                with st.expander(f"Supporting files for {skill['name']}"):
                    for file_path in skill["supporting_files"]:
                        st.code(file_path, language="text")
            st.markdown("")


def render_proficiency(report_dict: dict) -> None:
    """Renders the 'Proficiency' tab, displaying detailed scoring across technical areas."""
    prof = report_dict["proficiency_assessment"]
    area_list = prof["areas"]

    st.markdown("#### Proficiency Assessment")
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Score", prof["overall_score"])
    c2.metric("Overall Level", prof["overall_level"])
    c3.metric("Technical Maturity", prof["technical_maturity"])

    for area in area_list:
        with st.expander(f"{area['name']} — {area['level']} (Score: {area['score']}/10)"):
            for item in area["justification"]:
                st.markdown(f"- {item}")

    st.markdown("#### Assessment Summary")
    for line in prof["summary"]:
        st.markdown(f"- {line}")


def render_recruiter_summary(report_dict: dict) -> None:
    """Renders the 'Recruiter Summary' tab, synthesizing strengths, concerns, and limitations."""
    st.markdown("#### Recruiter Summary")

    left, right = st.columns([1.15, 1.0])

    with left:
        for item in report_dict["recruiter_summary"]:
            st.markdown(f"- {item}")

        render_list_section(
            "Strengths",
            report_dict["strengths"],
            "No major strengths were summarized.",
        )

        render_list_section(
            "Concerns",
            report_dict["concerns"],
            "No major concerns were summarized.",
        )

    with right:
        render_list_section(
            "Limitations",
            report_dict["limitations"],
            "No specific limitations recorded.",
        )


def render_verification_section(analysis_result: dict) -> None:
    """Renders the 'Verification' tab, displaying blockchain ledger integrity metrics."""
    block = analysis_result["ledger_block"]
    chain_valid = analysis_result["chain_valid"]
    chain_issues = analysis_result["chain_issues"]
    ledger_length = analysis_result["ledger_length"]

    st.markdown("#### Verification Ledger")

    c1, c2, c3 = st.columns(3)
    c1.metric("Ledger Block Index", block.index)
    c2.metric("Chain Length", ledger_length)
    c3.metric("Chain Status", "Valid" if chain_valid else "Issues Found")

    st.markdown("##### Block Details")
    st.code(
        json.dumps(
            {
                "index": block.index,
                "timestamp": block.timestamp,
                "repository_name": block.repository_name,
                "report_hash": block.report_hash,
                "previous_hash": block.previous_hash,
                "block_hash": block.block_hash,
            },
            indent=2,
        ),
        language="json",
    )

    if chain_valid:
        st.success("Ledger integrity verification passed.")
    else:
        st.error("Ledger verification reported one or more issues.")
        for issue in chain_issues:
            st.markdown(f"- {issue}")


def render_evidence(report_dict: dict) -> None:
    """Renders the 'Evidence' tab, providing a raw, auditable log of all detection signals."""
    evidence = report_dict["detected_technologies"]["evidence"]
    st.markdown("#### Raw Detection Evidence")

    if not evidence:
        st.info("No evidence entries were recorded.")
        return

    for item in evidence:
        with st.expander(f"{item['category']} • {item['name']} • {item['file_path']}"):
            st.markdown(f"**Reason:** {item['reason']}")


# ==========================================
# PHASE 5: APPLICATION ENTRY POINT
# ==========================================
def main() -> None:
    """Defines the main application entry point and orchestrates the UI rendering flow."""
    inject_custom_css()
    render_hero()

    source_input, analyze_clicked = render_sidebar()

    if "last_report_json" not in st.session_state:
        st.session_state["last_report_json"] = None
    if "last_report_filename" not in st.session_state:
        st.session_state["last_report_filename"] = "skillchain_report.json"


    if not analyze_clicked:
        st.markdown("### Start an Analysis")
        st.caption(
            "Paste a GitHub repository URL or a local project path in the sidebar, then click **Analyze Repository**."
        )
        return

    if not source_input.strip():
        st.warning("Please enter a valid GitHub repository URL or local project path.")
        return

    # --- Input type validation ---
    sidebar_radio = st.session_state.get("repo_input_type", None)
    if not sidebar_radio:
        if source_input.strip().startswith("http"):
            sidebar_radio = "GitHub Repository URL"
        else:
            sidebar_radio = "Local Project Path"

    if sidebar_radio == "GitHub Repository URL":
        if source_input.strip().lower().startswith("c:") or source_input.strip().lower().startswith("d:") or source_input.strip().lower().startswith("e:"):
            st.error("You selected 'GitHub Repository URL' but entered a local path. Please provide a valid GitHub repository URL (e.g., https://github.com/owner/repo).")
            return
        if not (source_input.strip().startswith("http://") or source_input.strip().startswith("https://")) or "github.com" not in source_input.strip().lower():
            st.error("Please enter a valid GitHub repository URL (must contain github.com).")
            return
    elif sidebar_radio == "Local Project Path":
        if source_input.strip().startswith("http://") or source_input.strip().startswith("https://") or "github.com" in source_input.strip().lower():
            st.error("You selected 'Local Project Path' but entered a URL. Please provide a valid local path (e.g., C:/Users/YourName/Documents/project-folder).")
            return
        if not (source_input.strip().lower().startswith("c:") or source_input.strip().lower().startswith("d:") or source_input.strip().lower().startswith("e:")):
            st.error("Please enter a valid local project path (must start with a drive letter, e.g., C:/...")
            return

    # --- Branch URL check ---
    import re
    branch_url_pattern = r"https://github.com/[^/]+/[^/]+/tree/([^/]+)"
    m = re.match(branch_url_pattern, source_input.strip())
    if m:
        branch = m.group(1)
        if branch.lower() not in ["main", "master"]:
            st.error(f"Please provide the main or master branch link, not a feature branch, to ensure a thorough and reliable analysis.\n\n**Tip:** This helps SkillChain review your entire project and provide the best possible insights! 😊")
            return

    try:
        with st.spinner("Analyzing repository, extracting skills, estimating proficiency, and verifying report..."):
            analysis_result = run_analysis(source_input)

        report = analysis_result["report"]
        report_dict = asdict(report)
        report_json = analysis_result["report_json"]
        export_path = analysis_result["report_export_path"]

        st.session_state["last_report_json"] = report_json
        st.session_state["last_report_filename"] = export_path.name

        st.success(f"Analysis completed for repository: {report.repository_name}")

        render_overview(report_dict)

        st.markdown("---")
        tabs = st.tabs(
            [
                "Technologies",
                "Extracted Skills",
                "Proficiency",
                "Recruiter Summary",
                "Verification",
                "Evidence",
            ]
        )

        with tabs[0]:
            render_detected_technologies(report_dict)

        with tabs[1]:
            render_extracted_skills(report_dict)

        with tabs[2]:
            render_proficiency(report_dict)

        with tabs[3]:
            render_recruiter_summary(report_dict)

        with tabs[4]:
            render_verification_section(analysis_result)

        with tabs[5]:
            render_evidence(report_dict)

        st.markdown("---")
        st.download_button(
            label="Download Analysis Report (JSON)",
            data=st.session_state["last_report_json"],
            file_name=st.session_state["last_report_filename"],
            mime="application/json",
            use_container_width=True,
        )

    except (RepoFetchError, TechDetectionError, SkillExtractionError, ProficiencyEstimationError, ReportGenerationError, LedgerError, OSError, ValueError) as exc:
        logger.exception("SkillChain analysis failed.")
        # Custom error message logic
        mode = st.session_state.get("repo_input", "").startswith("http")
        if mode:
            st.error("Analysis failed: Could not fetch or analyze the GitHub repository. Please check the URL and your network connection.")
        else:
            st.error("Analysis failed: Could not find or analyze the specified local project path. Please check the path and try again.")
    except Exception as exc:  # defensive fallback
        logger.exception("Unexpected application error.")
        st.error("An unexpected error occurred. Please try again or contact support.")


if __name__ == "__main__":
    main()