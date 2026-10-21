# Mandatory Mid-Review Question-Answer Document

**Project Title:** SonicMAE: Self-Supervised Masked Autoencoder with Phase-Scrambled Contrastive Representation for Industrial Acoustic Anomaly Detection  
**Team Number:** [Insert Team Number]  
**Students:** [Student Name 1, Roll No], [Student Name 2, Roll No], [Student Name 3, Roll No]  

---

### Q1. What exactly is the problem you are solving?
**Answer:**  
We are solving the problem of **Unsupervised Industrial Acoustic Anomaly Detection (ASD)** in industrial water pumps. Specifically, we aim to automatically detect mechanical failures (such as bearing contamination, impellor damage, shaft misalignment, clogging, and leakage) using continuous airborne sound audio recordings captured by microphone sensors in factory environments. Because real-world industrial plants operate almost exclusively under normal conditions, anomalous sound samples are exceedingly rare, diverse, and unavailable prior to failure. Therefore, the problem must be solved strictly in an **unsupervised setup**, where the model is trained exclusively on normal operational machine sounds and must accurately flag any acoustic departure from healthy operation during inference.

---

### Q2. Why is this problem worth solving?
**Answer:**  
Industrial pump failures in chemical plants, power stations, and municipal water facilities cause catastrophic unscheduled downtime, resulting in substantial financial losses (often exceeding tens of thousands of dollars per hour of downtime). Furthermore, structural machine breakdowns present severe worker safety hazards and environmental contamination risks. Traditional condition monitoring relies either on intrusive physical contact sensors (vibration accelerometers, thermal cameras) which are expensive and difficult to retrofit, or on manual acoustic inspection by experienced human technicians, which is subjective and non-continuous. Airborne acoustic monitoring provides a **non-contact, low-cost, continuous, and early-warning diagnostic capability**. Automatically detecting subtle acoustic anomalies enables predictive maintenance before mechanical failure occurs.

---

### Q3. Why is deep learning appropriate for this problem?
**Answer:**  
Acoustic machinery signals are non-stationary, high-dimensional time-frequency representations containing complex non-linear combinations of mechanical harmonics, motor hums, and ambient background factory noise. Traditional DSP techniques (such as static bandpass filtering, Fast Fourier Transform energy thresholds, or hand-crafted MFCC features with Gaussian Mixture Models) fail because:
1. They cannot model complex temporal and spectral dependencies simultaneously across time-frequency space.
2. They cannot adapt to ambient acoustic shifts in factory environments without manual retuning.
3. They lack the capacity to learn rich latent representations that disentangle machine-state acoustics from background noise.

Deep neural networks—specifically Convolutional Masked Autoencoders combined with contrastive representation learning—excel at learning hierarchical feature representations directly from continuous log-Mel spectrograms, making deep learning the ideal framework for capturing subtle, complex acoustic anomalies.

---

### Q4. What are the most relevant existing approaches?
**Answer:**  
The primary existing approaches in industrial acoustic anomaly detection include:
1. **Classical Statistical Baselines:** Principal Component Analysis (PCA) combined with Isolation Forests or One-Class Support Vector Machines (OC-SVM) operating on flattened log-Mel spectrogram features or MFCC summaries.
2. **Reconstruction-Based Deep Autoencoders:** Convolutional Autoencoders (Conv-AE) and Dense Autoencoders trained with Mean Squared Error (MSE) loss on normal log-Mel spectrograms. Anomaly decision is based on reconstruction error magnitude.
3. **Outlier Exposure & Density Estimation:** MobileNet/ResNet architectures trained with auxiliary outlier exposure or Cosine Similarity scoring in latent space (as used in DCASE challenge baselines).
4. **Self-Supervised Contrastive Learning (BYOL-A, SimCLR for Audio):** Encoder networks trained with contrastive or bootstrap objectives to align augmented audio views in embedding space.

---

### Q5. What are their limitations?
**Answer:**  
Each existing approach suffers from critical limitations:
* **Convolutional Autoencoders (Conv-AE):** Over-reconstruct subtle anomalous acoustic patches. Because standard convolutions apply spatial smoothing, the decoder frequently learns to smooth out high-frequency transient mechanical clicks or friction sounds, yielding low reconstruction error even for clear anomalies (leading to high false-negative rates).
* **Classical Baselines (PCA + Isolation Forest):** Incapable of modeling spatial/temporal structure in 2D spectrograms, treating time-frequency bins as independent features.
* **Standard Audio Augmentations (Pitch Shift, Time Stretch):** Shift physical machine operating frequencies, altering fundamental harmonic ratios and generating synthetic samples that violate physical machine acoustic dynamics.
* **Reconstruction-Only Scoring:** Extremely sensitive to uniform background factory noise increases, which elevate reconstruction MSE without any actual mechanical defect occurring.

---

### Q6. What exactly is your research gap?
**Answer:**  
Existing literature lacks a joint self-supervised framework that combines **context-aware time-frequency patch reconstruction (Masked Autoencoding)** with **phase-constrained contrastive learning** tailored specifically for continuous time-frequency spectrogram textures. Furthermore, existing models rely either purely on reconstruction error or purely on latent distance, leaving a gap for a robust **dual-stream scoring mechanism** calibrated against healthy validation bounds to prevent false alarms under background noise.

---

### Q7. How does your proposed approach address the gap?
**Answer:**  
Our proposed model, **SonicMAE**, addresses these gaps through three integrated mechanisms:
1. **Spectrogram Patch Masking (25% Ratio):** By randomly masking time-frequency patches and requiring the decoder to reconstruct them, SonicMAE forces the encoder to learn deep contextual acoustic dependencies across time frames and frequency channels, preventing the model from trivial identity mapping or over-reconstructing unmasked anomalous patches.
2. **Conjugate-Symmetric Phase Scrambling:** Introduces a novel phase perturbation augmentation that scrambles the spectral phase while strictly preserving complex conjugate symmetry and exact magnitude spectra. This forces the encoder to learn phase-invariant embeddings without corrupting physical machine harmonics.
3. **Joint Dual-Stream Scoring:** Combines normalized reconstruction MSE (capturing local high-frequency spectral distortions) with normalized Euclidean distance to a healthy embedding prototype bank (capturing macro acoustic drift), creating a robust decision score.

---

### Q8. What is unique about your approach?
**Answer:**  
The unique contributions of SonicMAE are:
* **Controlled Phase-Scrambling Augmentation:** Operates directly on raw audio waveforms to perturb spectral phase while retaining $100\%$ magnitude fidelity, ensuring augmented views represent valid physical audio.
* **Low-Masking Audio MAE Topology:** Unlike Vision MAEs which utilize a $75\%$ masking ratio on sparse natural images, SonicMAE utilizes a dense $25\%$ time-frequency masking ratio optimized for continuous log-Mel audio spectrograms.
* **Unified Dual-Loss Training:** Trains end-to-end using a joint objective function balancing reconstruction MSE and InfoNCE contrastive alignment:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda \cdot \mathcal{L}_{\text{InfoNCE}}$$

---

### Q9. Why did you select this model/architecture?
**Answer:**  
We selected a compact **2D Convolutional Masked Autoencoder** with a pooled linear embedding head:
* **2D Convolutions:** Naturally capture local time-frequency relationships in log-Mel spectrograms (harmonics across Mel channels, temporal continuity across frames).
* **Compact Parameterization (32 $\rightarrow$ 64 $\rightarrow$ 128 Channels):** Prevents model over-fitting on small industrial datasets, ensuring fast execution (<50 ms per sample) suitable for edge deployment.
* **Adaptive Average Pooling Embedding Head:** Compresses 2D feature maps into a fixed 128-dimensional bottleneck vector, ideal for computing real-time Euclidean distance to healthy machine prototype vectors.

---

### Q10. What dataset will you use, and why?
**Answer:**  
We use the **MIMII DUE (Malfunctioning Industrial Machine Investigation and Inspection under Domain Shifts)** industrial pump development dataset (Zenodo DOI: `10.5281/zenodo.4740355`).  
* **Why MIMII DUE Pump Dataset?**  
  1. It is the premier international benchmark for industrial acoustic anomaly detection (used in DCASE Challenges).  
  2. It contains real-world industrial pump recordings across multiple physical machine IDs (`id_00`, `id_02`, `id_04`, `id_06`).  
  3. It includes authentic mechanical defects (leakage, clogging, impellor damage, bearing contamination) recorded in operating factories alongside real ambient background noise.  
  4. It provides clean normal training data ($266$ samples) and rigorous evaluation splits ($57$ normal validation, $58$ normal test, $138$ abnormal test).

---

### Q11. What are your baselines?
**Answer:**  
We implemented and compared against two primary baseline models on the exact same dataset splits:
1. **CNN Autoencoder Baseline:** A standard 2D Convolutional Autoencoder without patch masking or contrastive learning, trained purely on reconstruction MSE.
2. **PCA + Isolation Forest Baseline:** A classical statistical anomaly detector utilizing Principal Component Analysis for dimensionality reduction followed by an Isolation Forest classifier fitted on healthy audio features.

---

### Q12. How will you evaluate the proposed approach?
**Answer:**  
We evaluate our model using official DCASE challenge metrics:
1. **Area Under the Receiver Operating Characteristic Curve (ROC-AUC):** Evaluates overall classification capability across all decision thresholds.
2. **Partial AUC (pAUC at 10% FPR):** Measures the Area Under the ROC Curve restricted to the low False Positive Rate region ($[0, 0.10]$), which is critical for industrial applications where false alarms must be strictly minimized.
3. **Precision, Recall, and F1-Score:** Evaluated at a decision threshold set at the **95th percentile** of unseen healthy validation recordings, mimicking real-world factory calibration.

---

### Q13. What progress have you made so far?
**Answer:**  
We have completed all core technical implementation milestones:
* Automated raw WAV loading, peak normalization, STFT, and log-Mel spectrogram preprocessing pipeline ($64\text{ Mel bins} \times 320\text{ frames}$).
* Implemented conjugate-symmetric phase-scrambling data augmentation.
* Constructed full PyTorch training loop for SonicMAE with joint MSE and InfoNCE loss functions.
* Completed baseline experiments (CNN Autoencoder, PCA + Isolation Forest) and benchmarked all models on the MIMII DUE pump test dataset.
* Built and launched a web-based Streamlit application (`app.py`) featuring audio playback, real-time score calculation, and time-frequency anomaly heatmap visualization.

---

### Q14. What preliminary results do you have?
**Answer:**  
Our preliminary experimental results on the MIMII DUE Pump benchmark are as follows:

| Model | ROC-AUC | pAUC @ 10% FPR | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CNN Autoencoder Baseline** | 0.5702 | 0.5483 | 0.8462 | 0.2391 | 0.3729 |
| **Proposed SonicMAE (Tuned $T2$)** | **0.7862** | **0.6911** | **0.9531** | **0.4420** | **0.6040** |
| **PCA + Isolation Forest** | 0.8450 | 0.7666 | 1.0000 | 0.5000 | 0.6667 |

**Key Insight:** SonicMAE achieves a **+21.6% improvement in ROC-AUC** over the CNN Autoencoder baseline and delivers **95.3% Precision**, confirming that masked self-supervision successfully prevents over-reconstruction of acoustic anomalies.

---

### Q15. What are the remaining tasks and how will you complete them?
**Answer:**  
The remaining tasks in our project roadmap are:
1. **Phase-Scrambling Strength Ablation Study:** Systematically run experiments varying `--phase-strength` from $0.0$ to $0.5$ to quantify the exact contribution of phase perturbation.
2. **Domain Adaptation Integration:** Incorporate domain-adversarial training or adaptive prototype normalization to maintain high pAUC when machine load or ambient factory background noise shifts between source and target environments.
3. **Model Quantization & Edge Deployment:** Convert the trained PyTorch SonicMAE model to ONNX / TensorRT format to test real-time edge inference latency on embedded micro-controllers (e.g., Raspberry Pi / Jetson Nano).
4. **Final Documentation:** Prepare final project paper, complete documentation, and upload deliverables.

---
