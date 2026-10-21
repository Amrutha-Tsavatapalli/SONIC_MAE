# Tomorrow's Review Preparation Guide — Person 2
## Role: Deep Learning Architecture, Loss Formulation & Model Training Lead

---

### 1. Ownership & Responsibilities
As **Person 2**, you own the **neural network architecture (SonicMAE Encoder, Decoder, Embedding Head), joint loss optimization ($\mathcal{L}_{\text{MSE}} + \lambda \mathcal{L}_{\text{InfoNCE}}$), model training loop, and comparative baseline implementations**.

#### Key Codebase Files You Own:
* [`src/sonicmae/model.py`](file:///c:/Users/kesav/OneDrive/Desktop/DL_audio/src/sonicmae/model.py) — SonicMAE PyTorch module (`nn.Conv2d`, `nn.BatchNorm2d`, `nn.GELU`, `nn.AdaptiveAvgPool2d`, `nn.ConvTranspose2d`).
* [`src/sonicmae/train.py`](file:///c:/Users/kesav/OneDrive/Desktop/DL_audio/src/sonicmae/train.py) — Training loop (`train_sonicmae`), AdamW optimizer, InfoNCE loss computation, and checkpointing.
* [`scripts/run_experiment.py`](file:///c:/Users/kesav/OneDrive/Desktop/DL_audio/scripts/run_experiment.py) — Execution script for baseline CNN Autoencoder, PCA + Isolation Forest, and tuned SonicMAE experiments.
* [`docs/LITERATURE_REVIEW_TABLE.md`](file:///c:/Users/kesav/OneDrive/Desktop/DL_audio/docs/LITERATURE_REVIEW_TABLE.md) — Literature benchmarking against Purohit et al., Harada et al., Niizumi et al.

---

### 2. Presentation Slides You Must Lead During Tomorrow's Review
During the review presentation, you should present and take the lead on the following slides from [`docs/MID_REVIEW_PRESENTATION.md`](file:///c:/Users/kesav/OneDrive/Desktop/DL_audio/docs/MID_REVIEW_PRESENTATION.md):

* **Slide 4 – Literature Review & Comparative Analysis**
  * Present prior models: Conv-AE (Purohit et al.), MobileNetV2 (Harada et al. DCASE), BYOL-A (Niizumi et al.), AST MAE (Huang et al.).
* **Slide 5 – Research Gap**
  * Explain over-reconstruction in standard CNN autoencoders and single-metric failure modes.
* **Slide 6 – Objective 1 (Architecture & Self-Supervision)**
  * State the core architectural objective of developing the self-supervised Masked Autoencoder with contrastive representation learning.
* **Slide 7 – Proposed Solution / Methodology & Mathematical Formulation**
  * Present the joint loss function $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + 0.20 \cdot \mathcal{L}_{\text{InfoNCE}}$.
* **Slide 10 – Implementation Status & Preliminary Results (Model Comparison)**
  * Lead the model comparison discussion (CNN Autoencoder vs. PCA + Isolation Forest vs. SonicMAE).

---

### 3. Core Technical Concepts & Formulas You Must Explain

#### A. SonicMAE Architecture Topology
```text
Input Spectrogram X [1, 64, 320] ──> (25% Mask Applied) ──> Masked X
                                                               │
                                                               ▼
                     ┌─────────────────────────────────────────────────┐
                     │ Encoder:                                        │
                     │ Conv2d(1 -> 32, k=4, s=2, p=1) + BN + GELU     │
                     │ Conv2d(32 -> 64, k=4, s=2, p=1) + BN + GELU    │
                     │ Conv2d(64 -> 128, k=3, s=1, p=1) + BN + GELU   │
                     └────────────────────────┬────────────────────────┘
                                              │ Latent Feature Map [128, 16, 80]
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
┌──────────────────────────────────────────────┐   ┌───────────────────────────────────┐
│ Embedding Head:                              │   │ Decoder:                          │
│ AdaptiveAvgPool2d((1,1)) -> Flatten          │   │ ConvTranspose2d(128->64, k=3, p=1)│
│ Linear(128 -> 128)                           │   │ GELU + ConvTranspose2d(64->32)    │
│ -> Bottleneck Vector z in R^128              │   │ GELU + ConvTranspose2d(32->1)     │
└──────────────────────────────────────────────┘   │ -> Reconstructed Spectrogram X_hat│
                                                   └───────────────────────────────────┘
```

#### B. Joint Optimization Loss Formulation
The model is trained end-to-end minimizing a dual loss objective combining **Reconstruction Error** and **InfoNCE Contrastive Alignment**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda \cdot \mathcal{L}_{\text{InfoNCE}}$$

1. **Masked Spectrogram Reconstruction Loss ($\mathcal{L}_{\text{MSE}}$):**
   Evaluated strictly over the masked time-frequency patches $M$:
   $$\mathcal{L}_{\text{MSE}} = \frac{1}{\|M\|_1} \sum_{i,j} \left( X_{i,j} - \hat{X}_{i,j} \right)^2 \odot M_{i,j}$$

2. **InfoNCE Contrastive Representation Alignment ($\mathcal{L}_{\text{InfoNCE}}$):**
   Given original view $z_i$ and augmented view $\tilde{z}_i$ (generated via phase scrambling & spectrogram jitter), InfoNCE pulls matching latent pairs together while pushing non-matching pairs apart in the mini-batch of size $N$:
   $$\mathcal{L}_{\text{InfoNCE}} = -\sum_{i=1}^{N} \log \frac{\exp(\text{sim}(z_i, \tilde{z}_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(z_i, z_j) / \tau)}$$
   * $\text{sim}(u, v) = \frac{u^T v}{\|u\|_2 \|v\|_2}$ (Cosine Similarity)
   * $\tau = 0.07$ (Temperature hyperparameter)
   * $\lambda = 0.20$ (Contrastive weighting factor)

---

### 4. Anticipated Viva / Evaluator Questions & High-Scoring Answers

#### Q1: "Why use GELU activations instead of standard ReLU in SonicMAE?"
> **High-Scoring Answer:**  
> Standard ReLU zeroes out all negative gradients ($\text{ReLU}(x) = \max(0, x)$), which leads to the 'dying ReLU' problem in autoencoders where quiet acoustic spectrogram bins (negative log-Mel energy values) lose gradient flow completely. GELU (Gaussian Error Linear Unit) provides a smooth, non-monotonic probabilistic curvature ($\text{GELU}(x) = x \cdot \Phi(x)$), allowing small negative gradients to flow during autoencoder reconstruction, yielding smoother spectral patch reconstruction.

#### Q2: "What is the role of InfoNCE loss when we already have Reconstruction MSE?"
> **High-Scoring Answer:**  
> Reconstruction MSE operates in pixel/spectral space and forces the decoder to reconstruct individual Mel-frequency bins. However, pure MSE is prone to over-fitting background factory noise. InfoNCE operates in the 128-dimensional latent space $z$. It aligns phase-scrambled acoustic views of the same machine state, forcing the encoder to extract invariant high-level semantic features of healthy operation rather than memorizing low-level background noise.

#### Q3: "Why did SonicMAE outperform standard CNN Autoencoders by +21.6% in ROC-AUC?"
> **High-Scoring Answer:**  
> Standard CNN Autoencoders (Purohit et al.) train on unmasked audio with pure MSE loss. Because CNNs apply continuous spatial convolutions, the decoder learns to smooth over high-frequency anomalous clicks/friction, reconstructing anomalies with low error (causing false negatives). SonicMAE enforces $25\%$ patch masking (forcing contextual prediction) AND joint InfoNCE latent alignment. Anomalous audio fails both patch reconstruction AND shifts far outside the InfoNCE healthy embedding space, driving up ROC-AUC from $0.5702$ to $0.7862$.

---

### 5. Person 2 Pre-Review Cheat Sheet

| Parameter / Layer | Exact Value / Spec | Purpose |
| :--- | :--- | :--- |
| **Encoder Conv Layers** | 32 -> 64 -> 128 channels | Spatial-temporal feature extraction from spectrogram |
| **Kernel & Strides** | Conv 4x4 (s=2) & 3x3 (s=1) | Downsamples spectrogram from [64, 320] to [128, 16, 80] |
| **Embedding Head** | AdaptiveAvgPool2d(1) + Linear(128, 128) | Bottleneck latent representation vector $z \in \mathbb{R}^{128}$ |
| **Contrastive Weight ($\lambda$)** | $0.20$ | Balances reconstruction vs contrastive alignment |
| **Temperature ($\tau$)** | $0.07$ | Scales cosine similarity in InfoNCE denominator |
| **Optimizer & Learning Rate** | AdamW ($\text{lr} = 10^{-3}$, weight decay $= 10^{-4}$) | Stable optimizer preventing weight explosion |
| **Epochs & Batch Size** | $30$ Epochs, Batch Size $= 16$ | Converges stably on MIMII DUE pump audio split |
