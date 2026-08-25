# Dataset: MIMII DUE pump development split

This project uses `dev_data_pump.zip` from the official public MIMII DUE Zenodo record (DOI: 10.5281/zenodo.4740355). It includes normal and anomalous industrial pump sounds and source/target domain conditions. The selected split is approximately 1.1 GB.

```powershell
python scripts/download_mimii_due_pump.py --extract
```

The downloader resumes a partial archive automatically. Raw audio and archives should not be committed. Fit the model only on normal recordings; reserve labelled normal/anomaly samples for evaluation. Follow the Zenodo record for license and citation requirements.
