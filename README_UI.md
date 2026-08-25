# SonicMAE demo dashboard

This Streamlit UI uses the exact CNN SonicMAE architecture and preprocessing from the completed Kaggle notebook.

## Add Kaggle artifacts

Create `artifacts/` beside `app.py`, then copy these Kaggle outputs into it:

```text
artifacts/
  sonicmae_selected_final_model.pt
  sonicmae_preprocessed_features.npz
  final_project_summary.json              # optional, enables metric cards
  final_model_comparison.csv              # optional, enables comparison table
```

The checkpoint alone is not enough: the feature cache is required to rebuild the healthy embedding prototype and validation-based decision threshold exactly as in the experiment.

## Run locally

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r ui_requirements.txt
streamlit run app.py
```

Open the local address Streamlit prints (normally `http://localhost:8501`). Upload a WAV recording to obtain the anomaly decision, score, waveform, input/reconstruction spectrograms, and time-frequency heatmap.

## Deployment

### Option 1: Docker Container Deployment

1. **Build the Docker Image**:
   ```bash
   docker build -t sonicmae-dashboard .
   ```

2. **Run the Container**:
   ```bash
   docker run -d -p 8501:8501 --name sonicmae-app sonicmae-dashboard
   ```
   Access the dashboard at `http://localhost:8501`.

### Option 2: Streamlit Community Cloud (Free Hosting)

1. Push this repository to GitHub (ensure `artifacts/` folder with checkpoint files is committed or uploaded via Git LFS).
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your repository, select branch `main`, and set Main file path to `app.py`.
4. Streamlit Cloud automatically uses `packages.txt` for `libsndfile1` audio dependencies and `ui_requirements.txt` / `requirements.txt` for Python dependencies.

