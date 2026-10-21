# Technical Roadmap & Architectural Upgrade Strategy for Final Review
## Project: SonicMAE — Next-Generation Industrial Acoustic Anomaly Detection

---

### Executive Summary
Building upon the successful baseline implementation of **SonicMAE** (achieving **0.7862 ROC-AUC** and **95.31% Precision** on the MIMII DUE industrial pump benchmark), this document details the future technical roadmap, architectural enhancements, transfer learning paradigms, domain adaptation strategies, and edge deployment plans scheduled for implementation prior to the **Final Project Review**.

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   FINAL REVIEW ARCHITECTURE ROADMAP                              │
├──────────────────────────┬──────────────────────────┬──────────────────────────┬─────────────────┤
│    TRANSFER LEARNING     │ CUSTOMIZED ARCHITECTURES │    DOMAIN ADAPTATION     │ EDGE DEPLOYMENT │
│                          │                          │                          │                 │
│ • Audio Spectrogram      │ • Conformer (CNN-Trans)  │ • Domain Adversarial Net │ • INT8 Model    │
│   Transformer (AST)      │   Hybrid Backbone        │   (DANN / MMD)           │   Quantization  │
│ • BEATs / AudioMAE       │ • Multi-Scale Patch      │ • Multi-Centroid Memory  │ • ONNX / TRT    │
│   Pre-trained Weights    │   Masking (4x4, 8x16)    │   Bank (Multi-Speed)     │   Edge Engine   │
│ • LoRA Parameter-        │ • Latent Memory          │ • Noise-Robust           │ • Real-Time     │
│   Efficient Fine-Tuning  │   Attention Stream       │   Augmentation           │   Technician UI │
└──────────────────────────┴──────────────────────────┴──────────────────────────┴─────────────────┘
```

---

### 1. Transfer Learning & Foundation Audio Models

While current SonicMAE relies on a compact 2D CNN autoencoder trained from scratch, industrial audio datasets are often limited in sample size. For the final review, we will incorporate pre-trained **Audio Foundation Models** using Transfer Learning paradigms.

#### Proposed Approach:
1. **Audio Spectrogram Transformer (AST) & AudioMAE Pre-trained Encoders:**
   * Leverage pre-trained weights from **AudioMAE** / **AST** (trained on AudioSet with 2+ million audio clips).
   * Extract high-level semantic acoustic representations without requiring million-sample industrial training datasets.
2. **Parameter-Efficient Fine-Tuning (PEFT / LoRA):**
   * Instead of fine-tuning all Transformer parameters (which risks catastrophic forgetting of general acoustic features), inject **Low-Rank Adaptation (LoRA)** matrices into the Transformer multi-head self-attention projections ($W_q, W_v$).
   * Trainable parameters will be reduced by **>90%**, ensuring fast adaptation to industrial pump operational sounds while preserving foundation acoustic priors.

```text
Pre-trained AudioMAE Backbone (AudioSet Weights - Frozen)
             │
             ├──> [LoRA Adapter W_q, W_v] (Trainable on MIMII Pump Audio)
             │
             └──> Latent Embedding Space (Rich Acoustic Representations)
```

---

### 2. Next-Generation Customized Architecture Upgrades

To address the research gaps identified during the mid-review (such as modeling long-range temporal dependencies in machine operating cycles and handling variable motor speeds), we propose three customized architectural upgrades:

#### Upgrade A: Conformer (Convolutional Transformer) Hybrid Backbone
* **Motivation:** CNNs excel at capturing local time-frequency textures (bearing clicks, high-frequency friction), while Transformers excel at capturing global temporal cycles (motor rotation speed, periodic pump piston strokes).
* **Architecture Design:** Replace the standard 2D CNN encoder with stacked **Conformer Blocks** combining:
  1. Depthwise Separable 2D Convolutional Modules (local spectral receptive fields).
  2. Multi-Head Self-Attention Modules (global temporal sequence context).
  3. Feed-Forward SwiGLU Networks with Layer Normalization.

#### Upgrade B: Multi-Scale Hierarchical Patch Masking
* **Motivation:** Uniform $25\%$ random pixel masking treats high-frequency clicks and low-frequency motor hums identically.
* **Mechanism:** Implement **Hierarchical Patch Masking** operating across multiple spatial scales:
  * Fine Patches ($4 \times 4$ Mel-time bins): Forces model to reconstruct transient acoustic impulses.
  * Coarse Patches ($8 \times 16$ Mel-time bins): Forces model to reconstruct multi-frame temporal harmonics and continuous spectral hums.

#### Upgrade C: Multi-Centroid Attention Prototype Memory Bank
* **Motivation:** Industrial pumps operate under variable speed and load conditions, generating multi-modal "healthy" embedding clusters rather than a single unimodal mean vector $z_{\text{proto}}$.
* **Mechanism:** Replace single prototype vector $z_{\text{proto}}$ with a **Multi-Centroid Latent Memory Bank** ($K=5$ centroids discovered via K-Means). Softmax attention weights assign input embeddings dynamically:
  $$d_{\text{proto}}(z) = \sum_{k=1}^{K} \text{Softmax}\left(-\frac{\|z - c_k\|_2}{\tau}\right) \cdot \|z - c_k\|_2$$

---

### 3. Cross-Machine Unsupervised Domain Adaptation (UDA)

#### Problem to Solve:
A major challenge in industrial condition monitoring is **Domain Shift**: an anomaly detection model trained on Pump `id_00` suffers performance drops when deployed on Pump `id_02` due to slight differences in motor power, installation acoustics, or background factory noise.

#### Proposed Solution for Final Review:
1. **Domain Adversarial Neural Networks (DANN):**
   * Attach a **Domain Classifier Head** to the encoder bottleneck $z$.
   * Use a **Gradient Reversal Layer (GRL)** during training to force the encoder to learn machine-invariant latent features while preserving anomaly detection capability.
2. **Maximum Mean Discrepancy (MMD) Latent Regularization:**
   * Minimize RKHS distance between source machine embeddings $Z_S$ and unlabeled target machine embeddings $Z_T$:
     $$\mathcal{L}_{\text{MMD}} = \left\| \frac{1}{N_S} \sum_{i=1}^{N_S} \phi(z_i^S) - \frac{1}{N_T} \sum_{j=1}^{N_T} \phi(z_j^T) \right\|_{\mathcal{H}}^2$$

---

### 4. Edge Hardware Deployment, Quantization & Real-Time Pipeline

For final review, we will transition SonicMAE from a desktop prototype to an **Edge-Ready Micro-Server Pipeline**:

```text
PyTorch Model (.pt) ──> ONNX Export ──> TensorRT / OpenVINO ──> INT8 Quantized Edge Engine (<15ms)
```

1. **Post-Training INT8 Quantization (PTQ):**
   * Quantize model weights and activation tensors from 32-bit Floating Point (`FP32`) to 8-bit Integer (`INT8`).
   * Reduces model footprint by **~75%** (from ~15 MB down to ~3.8 MB) with $<0.5\%$ loss in ROC-AUC.
2. **ONNX & TensorRT Export:**
   * Convert PyTorch graph into optimized ONNX runtime format for deployment on Raspberry Pi 4 / NVIDIA Jetson Orin Nano edge hardware.
   * Target inference latency: **$< 15\text{ ms}$ per 10-second audio stream** (enabling real-time streaming anomaly detection).

---

### 5. Advanced Explainability & Technician Diagnostic Dashboard

To make model predictions actionable for industrial maintenance teams, we will extend the Streamlit UI (`app.py`) with advanced Explainable AI (XAI) tools:

1. **Integrated Gradients & Grad-CAM Spectrogram Overlay:**
   * Compute pixel-level saliency gradients directly back to the log-Mel spectrogram input.
   * Highlight exact physical frequencies responsible for anomaly alerts.
2. **Automated Defect Diagnostics & Recommendation Engine:**
   * Map spectrogram anomaly patterns to likely physical root causes:
     * **High-frequency band energy spikes ($>5\text{ kHz}$):** Bearing contamination / lack of lubrication.
     * **Low-frequency periodic pulsations ($<500\text{ Hz}$):** Pump impeller cavitation / shaft misalignment.
     * **Broadband noise elevation:** Fluid leakage or seal degradation.

---

### 6. Timeline & Task Allocation for Final Review

| Enhancement Phase | Target Milestone | Primary Lead | Expected Performance Gain |
| :--- | :--- | :--- | :--- |
| **Phase 1: Foundation Models** | Pre-trained AudioMAE + LoRA Integration | Person 2 | ROC-AUC: $0.786 \rightarrow \mathbf{0.840+}$ |
| **Phase 2: Custom Architecture** | Conformer Backbone + Multi-Centroid Memory | Person 2 & Person 1 | pAUC @ 10% FPR: $0.691 \rightarrow \mathbf{0.750+}$ |
| **Phase 3: Domain Adaptation** | DANN / MMD for Cross-Machine Adaptation | Person 1 | Cross-Pump Generalization (+15% AUC on target IDs) |
| **Phase 4: Hardware & XAI UI** | INT8 ONNX Export + Grad-CAM Technician UI | Person 3 | Sub-15ms Latency + Visual Root-Cause Diagnostics |
