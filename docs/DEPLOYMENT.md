# Deploying the LIGHTRACK dashboard

## Recommended host

Use Streamlit Community Cloud. The app supports two data modes:

- **Local/server folder** for development on the measurement computer.
- **Upload recordings** for hosted use. Uploaded files are temporary and are not added to the repository.

The full raw dataset is about 17.9 GB and must not be committed to GitHub.

## Files required in GitHub

Commit at least these files on the branch you want to deploy:

- `fbg_analysis_dashboard.py`
- `self_calibrating_axle_detector.py`
- `sensor12_peak_tracker.py`
- `requirements.txt`
- `.streamlit/config.toml`

The two-period overview is shown only when its derived summary CSV exists on the server. The recording explorer remains available without that summary.

## Local deployment test

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run fbg_analysis_dashboard.py
```

Confirm both modes:

1. Select **Local/server folder** and load the local raw-data directory.
2. Select **Upload recordings** and upload a few timestamped `.txt` files.

## Streamlit Community Cloud

1. Push the required files to GitHub.
2. Open `https://share.streamlit.io` and sign in with GitHub.
3. Select **Create app** and choose the repository.
4. Select the deployment branch `Asude-Branch`.
5. Set the entrypoint to `fbg_analysis_dashboard.py`.
6. Open **Advanced settings** and select Python 3.12.
7. Deploy the app.
8. In the hosted app, choose **Upload recordings**.

Do not add local Windows paths, raw recordings, generated figures, spreadsheets, or output folders to the repository.
