# SonicMAE: Presentation Script & Speaker Document
**Project:** SonicMAE: Self-Supervised Masked Autoencoder with Phase-Scrambled Contrastive Representation Learning for Acoustic Anomaly Detection  
**Team Members:**
1. **Member 1 (Sanju Sri - AM.SC.U4AIE24018)**: Problem Context, Motivation, Background & Literature Review (Slides 1–5)
2. **Member 2 (Sravani - AM.SC.U4AIE24027)**: Proposed Methodology, System Architecture & Core Innovations (Slides 6–9)
3. **Member 3 (Amrutha - AM.SC.U4AIE24066)**: Dataset & Implementation Setup, Experimental Results, Challenges & Evaluation Plan (Slides 10–14)

---

# Part 1: Member 1 – Sanju Sri (Slides 1 – 5)
### *Focus: Introduction, Industrial Motivation, Problem Formulation & Literature Foundation*

---

### Slide 1: Title Slide
* **Slide Content:** Project Title, Team Number, and Member Names.
* **Explanation / Script:**
  > *"Good morning/afternoon everyone. Today, our team is presenting our project: **'SonicMAE: Self-Supervised Masked Autoencoder with Phase-Scrambled Contrastive Representation Learning for Acoustic Anomaly Detection'**. My name is Sanju Sri, and I will be covering the problem background, industrial motivation, and existing literature before passing on to my teammates to explain our architecture and experimental findings."*

---

### Slide 2: Problem Statement & Industrial Motivation
* **Key Points on Slide:**
  * **Problem Statement:** Industrial machinery (e.g., water pumps) continuously produces acoustic signals; subtle changes indicate mechanical faults.
  * **Goal:** Build an unsupervised deep-learning system learning strictly from normal pump sounds to detect anomalies.
  * **Target Anomalies:** Bearing contamination, Impeller damage, Shaft misalignment, Clogging, and Leakage.
  * **Why It Matters:** Predictive maintenance, zero unexpected downtime, non-contact monitoring, cheaper than dense physical vibration sensors.
  * **Why Deep Learning:** Industrial acoustics are non-stationary, noisy, and feature rich time-frequency dynamics that fail with classical statistics.
* **Explanation / Script:**
  > *"Industrial machines like centrifugal water pumps operate under harsh, continuous conditions. When components like bearings or impellers degrade, their acoustic signature changes well before catastrophic failure occurs. Our goal is to build an unsupervised acoustic anomaly detection model that learns only from normal machine operation sound and accurately flags deviations.*  
  > *This approach enables non-contact predictive maintenance, drastically lowering downtime without requiring expensive contact-sensor retrofitting. Because industrial audio is noisy, dynamic, and non-stationary, conventional statistical models fail—making deep representation learning essential."*

---

### Slide 3: Industrial Acoustic Anomaly Detection (Normal-Only Learning)
* **Key Points on Slide:**
  * **Pipeline:** Normal Audio $\rightarrow$ Feature Extraction $\rightarrow$ Representation Learning $\rightarrow$ Anomaly Score $\rightarrow$ Decision.
  * **Key Characteristics:** Signals vary over time, high background factory noise, subtle fault signatures, severe scarcity of abnormal/fault samples, and critical importance of time-frequency structure.
* **Explanation / Script:**
  > *"In real factory settings, machines operate normally 99% of the time. Collecting labeled abnormal data across every possible fault mode is practically impossible. Hence, our framework adopts a **Normal-only (One-Class / Unsupervised)** learning paradigm.*  
  > *The pipeline takes raw normal sound, extracts rich spectral features, learns normal acoustic representations, and computes an anomaly score at inference. If the score crosses a threshold, an anomaly is flagged. The main challenge is detecting subtle micro-faults buried inside loud factory background noise without generating false alarms."*

---

### Slide 4: Common Approaches & Their Limitations
* **Key Points on Slide:**
  * **PCA + Isolation Forest:** Fails to model 2D spatial-temporal spectrogram relationships.
  * **Convolutional Autoencoders (AE):** Can generalize too well and reconstruct subtle anomalies without showing large reconstruction errors.
  * **Density / Outlier Methods:** Fragile and heavily dependent on handcrafted feature quality.
  * **Contrastive Learning:** Learns rich representations, but lacks direct pixel-level reconstruction scoring.
  * **Key Takeaway:** Convert raw 1D waveforms into **Log-Mel Spectrograms** to preserve both time and frequency dynamics.
* **Explanation / Script:**
  > *"When surveying standard approaches, we identified fundamental limitations. Classical methods like PCA + Isolation Forest cannot capture 2D spectrogram geometries. Standard convolutional autoencoders often suffer from over-generalization—they inadvertently reconstruct subtle anomalies, masking the fault.*  
  > *Conversely, contrastive learning builds strong global representations but lacks direct reconstruction-error scoring. This led us to our key idea: process audio as **Log-Mel Spectrograms** and combine both reconstruction and contrastive objectives to get the best of both worlds."*

---

### Slide 5: Literature Review
* **Key Points on Slide:**
  * **Purohit et al. (2019):** Introduced MIMII dataset (benchmark for malfunctioning industrial machines).
  * **Doherty et al. (2021):** DCASE ASD challenge highlighting domain shifts and acoustic variability.
  * **He et al. (2022) [MAE]:** Masked Autoencoders demonstrating that reconstructing masked patches forces deep contextual representation learning.
  * **Niizumi et al. (2021) [BYOL-A]:** Self-supervised representation learning for general audio.
  * **Schölkopf et al. (2001):** Foundational One-Class SVM for boundary estimation from normal data.
* **Explanation / Script:**
  > *"Our work builds directly on pioneering literature: Purohit et al.'s MIMII dataset established industrial acoustic benchmarks, and DCASE challenges emphasized domain variability. In self-supervised learning, He et al.'s Masked Autoencoders showed that predicting masked regions forces models to learn intrinsic structural dependencies, while BYOL-A proved self-supervision works effectively on audio.*  
  > *We bridge these paradigms into a dedicated framework for industrial machines. I will now hand over to Sravani to present the research gap, architecture, and core innovations of SonicMAE."*

---
---

# Part 2: Member 2 – Sravani (Slides 6 – 9)
### *Focus: Research Gap, Objectives, SonicMAE End-to-End Pipeline & Novelty*

---

### Slide 6: Research Gap: What Existing Methods Miss
* **Key Points on Slide:**
  * **Reconstruction-Based:** Detects anomalies strictly via pixel/MSE reconstruction error thresholding.
  * **Contrastive/Self-Supervised:** Focuses only on high-level latent feature representation.
  * **Classical Methods:** Simple scoring but lacks deep feature modeling.
  * **Identified Gap:** Lack of a joint self-supervised framework combining context-aware masked time-frequency reconstruction, physics-informed contrastive learning, and complementary anomaly scoring.
  * **Proposed Direction:** Masked Time-Frequency Reconstruction + Phase-Constrained Contrastive Learning + Embedding Distance + Reconstruction Error.
* **Explanation / Script:**
  > *"Thank you, Sanju. Looking at existing methods, there is a clear divide: reconstruction models focus only on low-level spectrogram reconstruction, while contrastive learning focuses only on high-level latent representations.*  
  > *Our research gap identifies the need for a **joint self-supervised framework** that forces the model to understand local time-frequency continuity while maintaining global semantic clustering of healthy machine states. We unify masked reconstruction with phase-constrained contrastive learning to generate a far more discriminative anomaly representation."*

---

### Slide 7: Objectives & Scope
* **Key Points on Slide:**
  * **Acoustic Pipeline:** Robust audio preprocessing and 2D spectrogram conversion.
  * **Masked Autoencoder:** Context-aware reconstruction of masked spectrogram patches.
  * **Self-Supervised Learning:** Phase-scrambled contrastive representation learning.
  * **Dual-Stream Scoring:** Joint anomaly scoring using both reconstruction error and latent embedding distance.
  * **Performance Evaluation:** Rigorous benchmarking against baselines on MIMII DUE.
* **Explanation / Script:**
  > *"To address this gap, we established five key project objectives:*
  > 1. *Develop an end-to-end acoustic preprocessing pipeline.*
  > 2. *Design a 2D CNN Masked Autoencoder tailored for acoustic spectrograms.*
  > 3. *Incorporate phase-scrambled contrastive learning to enforce acoustic invariance.*
  > 4. *Implement a dual-stream anomaly scoring mechanism combining latent distance and spatial reconstruction error.*
  > 5. *Benchmark and evaluate detection performance under industrial noise conditions."*

---

### Slide 8: System Pipeline and Workflow
* **Key Points on Slide:**
  * **Input:** Raw `.wav` audio (16 kHz, mono).
  * **Preprocessing:** Peak Normalization $\rightarrow$ STFT $\rightarrow$ Mel Filterbank $\rightarrow$ Log-Mel Spectrogram $(64 \times 320)$.
  * **Augmentation & Masking:** Additive noise, Time shift, Phase scrambling, Random patch masking (25% mask ratio).
  * **SonicMAE Model:** 2D CNN Encoder $\rightarrow$ 128-D Embedding (Contrastive InfoNCE Loss) $\rightarrow$ Transpose CNN Decoder $\rightarrow$ Reconstructed Spectrogram.
  * **Dual Scoring:** Reconstruction Error (MSE) + Embedding Distance to Healthy Prototype Bank.
  * **Decision:** Threshold comparison $\rightarrow$ Classify as Normal vs Anomaly.
* **Explanation / Script:**
  > *"Here is the complete architecture of SonicMAE. The input raw pump audio is sampled at 16 kHz and converted into a $64 \times 320$ Log-Mel spectrogram via STFT. During training, we apply time-shifts, additive noise, phase scrambling, and mask 25% of the spectrogram patches.*  
  > *The masked input passes through a 2D CNN Encoder to produce a compact 128-dimensional latent embedding, trained with InfoNCE contrastive loss. Simultaneously, a Transposed-CNN Decoder reconstructs the missing spectrogram regions.*  
  > *During inference on unseen audio, we compute two scores: the **spectrogram reconstruction MSE** and the **Euclidean distance** to a bank of healthy prototype embeddings. A weighted combination gives the final anomaly score."*

---

### Slide 9: Uniqueness & Novelty
* **Key Points on Slide:**
  * **Joint Learning:** Synchronous masked reconstruction + contrastive representation learning.
  * **Controlled Masking:** Tuned 25% masking ratio suited for audio continuity (unlike 75% in computer vision).
  * **Phase-Constrained Augmentation:** Modulates signal phase to simulate acoustic perturbations while preserving harmonic physics.
  * **Dual-Stream Anomaly Scoring:** Combines spatial reconstruction residual with latent semantic distance.
  * **Overall Contribution:** A unified, physics-aware self-supervised framework for acoustic anomaly detection.
* **Explanation / Script:**
  > *"What makes SonicMAE unique compared to standard MAEs?*  
  > *First, in vision MAEs, mask ratios are typically 75%, but audio spectrograms have strong continuous harmonics where heavy masking destroys context; we empirically found 25% optimal.*  
  > *Second, we introduce **Phase-Constrained Augmentation**, scrambling the STFT phase to create realistic physical acoustic variations without corrupting energy distribution.*  
  > *Finally, our **Dual-Stream Scoring** ensures that if an anomaly slips past the reconstructor, the embedding distance detects it, giving high sensitivity. Now, Amrutha will explain our experimental setup, preliminary results, and challenges."*

---
---

# Part 3: Member 3 – Amrutha (Slides 10 – 14)
### *Focus: Dataset, Hyperparameters, Results, Challenges, Evaluation & References*

---

### Slide 10: Dataset Overview, Audio Preprocessing & Hyperparameters
* **Key Points on Slide:**
  * **Dataset:** MIMII DUE (Pump Development Dataset, Zenodo DOI: 10.5281/zenodo.4740355).
  * **Split Breakdown:** Training: 266 Normal | Validation: 57 Normal | Testing: 58 Normal / 138 Abnormal.
  * **Audio Preprocessing:** 16 kHz sampling, FFT 1024, Hop size 512, 64 Mel bins, 320 frames, standardized using training set statistics.
  * **Hyperparameters:** PyTorch framework, AdamW optimizer, Learning Rate $10^{-3}$, Batch Size 16, 30 Epochs, 0.25 Mask Ratio, 0.20 Contrastive Loss Weight.
* **Explanation / Script:**
  > *"Thank you, Sravani. To validate SonicMAE, we used the benchmark **MIMII DUE Pump Dataset**. The training and validation sets contain exclusively normal operating sounds—266 and 57 clips respectively—while the test set contains 58 normal and 138 real-world abnormal sounds.*  
  > *Audio is resampled to 16 kHz, transformed with an FFT window of 1024 and hop size of 512 into $64 \times 320$ spectrograms, normalized strictly using training statistics to prevent data leakage. We implemented this in PyTorch with AdamW, learning rate $10^{-3}$, batch size 16, training for 30 epochs with a tuned contrastive loss weight of 0.20."*

---

### Slide 11: Preliminary Results & Project Roadmap
* **Key Points on Slide:**
  * **Completed (Phase 1):** Data ingestion & preprocessing pipeline, CNN Autoencoder baseline, PCA + Isolation Forest baseline, SonicMAE training pipeline, model benchmarking, and interactive Streamlit demo interface.
  * **In Progress (Phase 2):** Phase-scrambling ablation study to isolate the impact of phase augmentation.
  * **Planned (Phase 3):** Domain adaptation under varying SNR/loads, and lightweight model quantization for Edge deployment.
* **Explanation / Script:**
  > *"We have structured our project into three phases. In Phase 1, we successfully built the ingestion pipeline, implemented two baseline models (PCA+Isolation Forest and standard CNN Autoencoder), trained SonicMAE, and deployed a Streamlit dashboard for real-time visualization.*  
  > *Currently in Phase 2, we are conducting ablation studies to evaluate exact gains from phase scrambling versus standard augmentations. In Phase 3, we plan to address domain adaptation across different machine load conditions and optimize the architecture for edge microcontrollers."*

---

### Slide 12: Challenges Encountered
* **Key Points on Slide:**
  * **1. Domain Shift:** Machine acoustic signatures change across operating loads, environments, and background factory noise.
  * **2. Precision-Recall Trade-Off:** Achieving high precision without missing subtle, low-intensity anomalies.
  * **3. Reconstruction Sensitivity:** Ambient factory noise can inflate reconstruction error on healthy pumps, causing false alarms.
  * **4. Phase Perturbation Limits:** Over-scrambling phase destroys critical harmonic relationships.
  * **5. Edge Deployment:** Hardware constraints and millisecond latency limits on embedded edge hardware.
* **Explanation / Script:**
  > *"Throughout development, we identified five key technical challenges:*  
  > *First, **Domain Shift**—acoustic properties shift when pump loads vary.  
  > Second, balancing **Precision and Recall** so sensitive thresholds do not trigger false alarms.  
  > Third, high background noise can cause reconstruction errors on healthy machines.  
  > Fourth, carefully tuning phase perturbations so we augment without corrupting harmonic physics.  
  > And fifth, optimizing inference latency for real-time edge processing."*

---

### Slide 13: Expected Outcomes & Evaluation Plan
* **Key Points on Slide:**
  * **Primary Metrics:** Area Under ROC Curve (ROC-AUC) and partial AUC (pAUC @ 10% FPR) standard to DCASE.
  * **Threshold Metrics:** Precision, Recall, and F1-score.
  * **Comparative Baselines:** SonicMAE vs CNN Autoencoder vs PCA + Isolation Forest.
  * **Ablation Studies:** Phase-strength ablation, mask-ratio sensitivity, and reconstruction error heatmaps for visual interpretability.
* **Explanation / Script:**
  > *"Our evaluation follows the official DCASE benchmark standards, using **ROC-AUC** and **partial AUC at 10% False Positive Rate (pAUC)** as our primary metrics, alongside Precision, Recall, and F1-score.*  
  > *We directly compare SonicMAE against PCA+Isolation Forest and standard CNN Autoencoders. Additionally, our pipeline generates reconstruction error heatmaps, allowing plant engineers to visually inspect exactly which frequency bands triggered the anomaly flag."*

---

### Slide 14: Key References & Conclusion
* **Key Points on Slide:**
  * MIMII Dataset (Purohit et al., 2019)
  * DCASE Task 2 (Doherty et al., 2021)
  * Masked Autoencoders (He et al., 2022)
  * BYOL for Audio (Niizumi et al., 2021)
  * One-Class Support Estimation (Schölkopf et al., 2001)
* **Explanation / Script:**
  > *"To conclude, SonicMAE combines masked spectrogram reconstruction with phase-aware contrastive representation learning, providing a robust, interpretable, normal-only anomaly detection system for industrial machinery. Here are our key literature references. Thank you, and we are now open to questions."*

---
