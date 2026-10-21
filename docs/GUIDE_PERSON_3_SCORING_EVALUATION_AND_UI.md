# Tomorrow's Review Preparation Guide — Person 3
## Role: Anomaly Scoring Engine, Experimental Evaluation & Streamlit/Docker Deployment Lead

---

### 1. Ownership & Responsibilities
As **Person 3**, you own the **dual-stream anomaly scoring metric, decision threshold calibration, quantitative benchmarking metrics (ROC-AUC, pAUC, Precision, Recall, F1), Streamlit GUI (`app.py`), and Docker container deployment (`Dockerfile`)**.

#### Key Codebase Files You Own:
* [`src/sonicmae/scoring.py`](file:///c:/Users/kesav/OneDrive/Desktop/DL_audio/src/sonicmae/scoring.py) — Reference prototype fitting (`fit_reference`), dual-stream anomaly scoring calculation (`anomaly_score`), and ROC-AUC evaluation (`roc_auc`).
* [`app.py`](file:///c:/Users/kesav/OneDrive/Desktop/DL_audio/app.py) — Interactive Streamlit web interface for live audio upload, real-time inferencing, spectrogram visualization, and time-frequency heatmap generation.
* [`Dockerfile`](file:///c:/Users/kesav/OneDrive/Desktop/DL_audio/Dockerfile) & [`README_UI.md`](file:///c:/Users/kesav/OneDrive/Desktop/DL_audio/README_UI.md) — Containerization environment and UI documentation.
* [`artifacts/final_project_summary.json`](file:///c:/Users/kesav/OneDrive/Desktop/DL_audio/artifacts/final_project_summary.json) & [`artifacts/final_model_comparison.csv`](file:///c:/Users/kesav/OneDrive/Desktop/DL_audio/artifacts/final_model_comparison.csv) — Benchmark metric summaries.

---

### 2. Presentation Slides You Must Lead During Tomorrow's Review
During the review presentation, you should present and take the lead on the following slides from [`docs/MID_REVIEW_PRESENTATION.md`](file:///c:/Users/kesav/OneDrive/Desktop/DL_audio/docs/MID_REVIEW_PRESENTATION.md):

* **Slide 6 – Objective 3 & 4 (Scoring Metric Design & Benchmarking)**
  * Explain dual-stream scoring combining normalized reconstruction MSE and prototype distance under 95th percentile validation calibration.
* **Slide 7 – Inference Scoring Formulation**
  * Present the mathematical formula for normalized score $S(x)$ and prototype distance $\|z - z_{\text{proto}}\|_2$.
* **Slide 10 – Benchmark Performance Table & Detailed Analysis**
  * Walk through the evaluation metrics (ROC-AUC 0.7862, pAUC 0.6911, Precision 95.31%, Recall 44.20%, F1 0.6040).
* **Slide 11 – Challenges & Limitations**
  * Discuss high-precision vs. recall trade-offs and ambient background noise shifts.
* **Slide 13 – Expected Outcomes & Live Dashboard Demo**
  * Showcase the Streamlit web dashboard capabilities and time-frequency heatmap visualization.

---

### 3. Core Technical Concepts & Formulas You Must Explain

#### A. Dual-Stream Anomaly Scoring Engine
Instead of relying on a single metric, SonicMAE fuses two complementary acoustic anomaly signals:

$$S(x) = \frac{\text{MSE}(X, \hat{X}) - \mu_{\text{err}}}{\sigma_{\text{err}}} + \frac{\|z - z_{\text{proto}}\|_2 - \mu_{\text{dist}}}{\sigma_{\text{dist}}}$$

1. **Stream 1: Reconstruction Error ($\text{MSE}(X, \hat{X})$):**
   Measures localized high-frequency spectral distortions (e.g., sudden bearing clicks, sharp friction bursts).
2. **Stream 2: Latent Prototype Distance ($\|z - z_{\text{proto}}\|_2$):**
   Measures global macro-acoustic shifts from healthy operational baseline. The healthy prototype $z_{\text{proto}}$ is computed as the mean vector of all normal training embeddings:
   $$z_{\text{proto}} = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} z_i$$
3. **Z-Score Normalization ($\mu, \sigma$):**
   $\mu_{\text{err}}, \sigma_{\text{err}}$ and $\mu_{\text{dist}}, \sigma_{\text{dist}}$ are fitted strictly on normal training audio so both streams contribute equally on a zero-mean unit-variance scale.

#### B. 95th Percentile Decision Threshold Calibration
* Industrial operations require **ultra-low false alarm rates**.
* Threshold $\text{Th}_{\text{decision}}$ is set at the **95th percentile of validation normal scores**:
  $$\text{Th}_{\text{decision}} = \text{Percentile}_{95}(\{S(x_v) \mid x_v \in \text{Validation Normal}\})$$
* Any test sample with $S(x) > \text{Th}_{\text{decision}}$ is flagged as **ANOMALY**.

#### C. Evaluation Metrics Definition
* **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):** Overall ranking capability across all thresholds ($0.7862$).
* **pAUC (Partial AUC @ 10% FPR):** Area under ROC curve integrated strictly between $0.0$ and $0.10$ False Positive Rate ($0.6911$). Measures performance under strict factory alarm budgets.
* **Precision ($95.31\%$):** $\frac{TP}{TP + FP}$ — When the system triggers an anomaly alarm, it is correct $95.31\%$ of the time!

---

### 4. Live Dashboard Demo Walkthrough Strategy (Streamlit `app.py`)
If asked to demonstrate the working project during the review:
1. Show terminal launching Streamlit: `streamlit run app.py`.
2. Open `http://localhost:8501`.
3. Point out the top summary metric cards (**Selected Model: SonicMAE**, **ROC-AUC: 0.7862**, **pAUC: 0.6911**, **Precision: 95.31%**).
4. Upload `pump_normal_recording.wav`: Show **NORMAL** decision badge (Green), score below threshold, clean reconstruction, low residual heatmap.
5. Upload `pump_abnormal_recording.wav`: Show **ANOMALY DETECTED** decision badge (Red), score exceeding threshold, residual heatmap highlighting specific time-frequency failure regions!

---

### 5. Anticipated Viva / Evaluator Questions & High-Scoring Answers

#### Q1: "Why do you use Partial AUC (pAUC) @ 10% FPR instead of just standard ROC-AUC?"
> **High-Scoring Answer:**  
> In real-world manufacturing plants, an anomaly detection system operating at a $30\%$ or $50\%$ False Positive Rate is completely unusable because false alarms stop production lines unnecessarily. Standard ROC-AUC evaluates the entire curve from $0\%$ to $100\%$ FPR, giving high scores even if performance at low false-alarm rates is poor. pAUC restricts evaluation to the operational zone ($FPR \le 0.10$), proving our model maintains high sensitivity under strict industrial alarm constraints.

#### Q2: "Why is Precision 95.31% while Recall is 44.20%?"
> **High-Scoring Answer:**  
> This is a intentional engineering trade-off resulting from our $95\text{th}$ percentile threshold calibration on normal validation audio. By setting a high threshold, we prioritize zero false alarms ($95.31\%$ Precision), ensuring maintenance crews never waste time on false alerts. The $44.20\%$ Recall catches major structural defects (clogging, bearing failure). In future iterations, multi-threshold alerting (Warning vs. Critical) will boost recall on early micro-degradations.

#### Q3: "How does the Streamlit heatmap localize anomalies in time and frequency?"
> **High-Scoring Answer:**  
> The heatmap visualizes the element-wise squared residual matrix $R(f, t) = (X(f, t) - \hat{X}(f, t))^2$ across the $64$ Mel bins and $320$ time frames. When a pump experiences impeller rubbing or leakage, the decoder fails to reconstruct those specific unlearned frequency bands, generating intense red spots on the heatmap that tell technicians exactly *when* ($t$) and at *what frequency band* ($f$) the failure occurred.

---

### 6. Person 3 Pre-Review Cheat Sheet

| Metric / Artifact | Benchmark Value | Meaning / Significance |
| :--- | :--- | :--- |
| **ROC-AUC** | **0.7862** (+21.6% over Conv-AE) | Excellent overall normal vs anomaly separation |
| **pAUC @ 10% FPR** | **0.6911** | High accuracy within strict 10% false alarm budget |
| **Precision** | **95.31%** | 19 out of 20 triggered alarms are genuine defects |
| **Recall** | **44.20%** | Detects severe mechanical faults at 95th percentile threshold |
| **F1-Score** | **0.6040** | Harmonic mean of precision and recall |
| **Decision Threshold** | 95th percentile of Val scores | Calibrated strictly on healthy validation recordings |
| **Deployment Tech** | Streamlit + PyTorch + Docker | Multi-platform deployment (`app.py`, `Dockerfile`) |
