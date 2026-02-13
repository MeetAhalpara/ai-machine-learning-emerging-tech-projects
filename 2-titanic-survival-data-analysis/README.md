# Titanic Survival Data Analysis

The Titanic disaster is not just a historical tragedy; it is a profound lesson in human behavior under extreme constraints. When faced with absolute crisis, who survives? This project is an analytical journey into the historical passenger manifest, designed to uncover the hidden demographic and socio-economic forces that determined life or death during the disaster.

Raw historical data is inherently flawed. It is incomplete, chaotic, and heavily fragmented. The challenge was to engineer a clean, robust exploratory data analysis pipeline that could translate over eight hundred raw records into clear, undeniable truths without introducing statistical bias.

To achieve this, a dynamic preprocessing architecture was developed. Every missing piece of information was handled programmatically. Age voids were intelligently imputed using central tendency metrics to preserve demographic integrity. Structural anomalies in embarkation ports were resolved through localized statistical mode replacement. Finally, irrecoverable data, such as the heavily corrupted Cabin matrix, was entirely excised to maintain the purity of the analysis.

## Core Analytical Insights

The resulting pipeline generated crystal-clear insights, exported into a comprehensive visual dashboard. The numbers reveal a stark reality:

* The Gender Delta: The maritime protocol of "women and children first" was undeniably enforced. Female passengers achieved an aggregate survival rate of 74.2%, contrasting sharply against male survival metrics at a mere 18.9%.
* Socio-Economic Stratification: Purchasing power dictated survival velocity. First-class ticket holders navigated the disaster with an optimal 63.0% survival rate, whereas those confined to third-class faced severe degradation, dropping to just 24.2%.

---

## Pipeline Architecture & Preprocessing

To ensure clean data profiling, the system relies on intelligent missing value handling strategies:
* Age Vector Imputation: Missing fields within the age continuum are calculated dynamically using the dataset's overall median age value (28.0). This maintains the distribution shape while preserving valuable records.
* Embarked Coordinate Processing: Missing voyage origin ports are safely defaulted to the most frequent boarding location.
* Feature Dropping: The Cabin matrix is dropped entirely due to a high density of missing fields exceeding a 70% null-value ratio, preventing model hallucination.

### Project Layout
The repository is structured to seamlessly transform raw files into visual intelligence:
* `data_analysis.py`: The core exploratory data engineering script driving the logic.
* `Titanic-Dataset.csv`: The structured input data table featuring 891 passenger records across 12 distinct analytical attributes.
* `titanic_demographic_survival_dashboard.png`: The multi-panel analytical visualization mapping out demographic survival variances.

---

## Pipeline Initialization

To initiate the analytical extraction, activate your local virtual environment and execute the core script.

```bash
.\.venv\Scripts\activate
python data_analysis.py