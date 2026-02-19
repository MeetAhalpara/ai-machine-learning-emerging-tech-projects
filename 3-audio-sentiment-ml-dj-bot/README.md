# Audio Sentiment Machine Learning DJ Bot

Music is fundamentally an expression of emotion, but to a machine, it is purely a matrix of acoustic parameters. The challenge of this project was to build a system capable of listening to a track and categorically defining its emotional state—separating high-octane energy ('Hype') from relaxed, ambient rhythms ('Chill'). This module evaluates the predictive behavior of foundational machine learning architectures, specifically Logistic Regression, Decision Trees, and K-Nearest Neighbors, using physical acoustic signals.

## Feature Architecture & Preprocessing

To bridge the gap between sound and statistics, the system isolates continuous auditory properties from an immense dataset of 232,725 historical track records. Tempo captures the rhythmic velocity, while Energy serves as the dominant structural selector. 

To train the models, a strict mathematical ground-truth was established. A binary thresholding rule was applied programmatically to the dataset:

$$\text{Target Class } (y) = \begin{cases} 1 \text{ (Hype)}, & \text{if } \text{energy} > 0.6 \\ 0 \text{ (Chill)}, & \text{otherwise} \end{cases}$$

Before training, the entire acoustic support structure underwent an 80/20 train-test validation split. The feature columns were then standardized using statistical scaling to ensure that massive numerical ranges (like tempo) did not overpower subtle decimal shifts (like energy).

---

## Model Evaluation & Engineering Insights

The models generated unprecedented performance results against the isolated testing partition:

* **Logistic Regression:**
  * **Classification Accuracy:** 1.0 (100%)
  * **Precision / Recall / F1-Score:** Flawless (1.00) across all categorical classes.
* **Decision Tree ($max\_depth=5$):**
  * **Classification Accuracy:** 1.0 (100%)
  * **Precision / Recall / F1-Score:** Flawless (1.00) across all categorical classes.
* **K-Nearest Neighbors (KNN, $k=5$):**
  * **Classification Accuracy:** 0.999119 (99.91%)
  * **Precision / Recall / F1-Score:** 1.00 / 1.00 / 1.00

### Post-Mortem Reflection: The Data Leakage Paradigm

While perfect metrics often signal success, in machine learning, they frequently reveal a structural flaw. The flawless accuracy achieved by the linear and hierarchical models exposed an explicit data leakage scenario. 

Because the target label was derived directly from the `energy` metric—and that exact metric was intentionally retained within the training feature space—the models simply reverse-engineered the absolute mathematical threshold rule instead of learning abstract musical sentiment. KNN varied marginally (99.91%) only due to localized spatial neighborhood density boundaries.

This project serves as a powerful demonstration of feature isolation, illustrating the immediate consequences of providing an algorithm with the very rule it is supposed to discover.

---

## Project Layout

* `dj_bot_pipeline.py`: The production-grade Python script containing the data extraction pipeline, model verification loops, and output reporting architecture.
* `SpotifyFeatures.csv`: The underlying foundational data matrix containing the multi-dimensional acoustic track properties.

---

## Pipeline Initialization

To initiate the evaluation pipeline, activate your local virtual environment and execute the core script.

```bash
.\.venv\Scripts\activate
python dj_bot_pipeline.py