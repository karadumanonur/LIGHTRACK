# Train Monitoring — Asude Branch

Research code and a local analysis application for fibre Bragg grating (FBG) railway measurements. This branch focuses on robust axle counting, sensor comparison, frequency-domain inspection and matching FBG events with Infrabel axle-count records.

## Application

Start the dashboard from Python:

```powershell
python -m pip install -r requirements.txt
python launch_train_monitoring.py
```

The Windows and macOS release packages include Python and the required libraries. Raw recordings are deliberately not included; select the local `raw_data` folder when Train Monitoring starts.

## Repository layout

| Location | Purpose |
|---|---|
| `fbg_analysis_dashboard_strict.py` | Current Streamlit entry point |
| `train_monitoring_ui.py` | Current interactive recording-analysis dashboard |
| `axle_detector.py` | Canonical latest axle-counting algorithm |
| `recording_io.py` | Recording loading and physical-FBG mapping |
| `legacy/` | Previous detectors and dashboard retained for research reference |
| `research/` | Reproducible comparison, matching and movement-study scripts |
| `exploratory/` | Earlier familiarisation and investigation scripts |
| `packaging/` | Windows and macOS application builders and instructions |
| `docs/` | Deployment and release documentation |

## Self-calibrating axle detector

The latest detector keeps the validated 0.5–40 Hz Butterworth band while learning record-specific decision limits:

1. identify quiet background blocks;
2. estimate background scale with calibrated MAD;
3. select the direct-response polarity;
4. learn height and prominence limits from background extremes;
5. estimate response spacing from autocorrelation and candidate gaps;
6. apply peak-width and shape checks;
7. count accepted axle candidates.

The active UI imports this implementation directly from `axle_detector.py`.
Previous standalone algorithms are isolated under `legacy/`.

## Data policy

GitHub tracks source code and documentation, not the 17.9 GB dataset or reproducible outputs. The `.gitignore` excludes raw text recordings, spreadsheets, XML, plots, result folders and application builds. Each user supplies the measurement data locally.

## Builds and releases

GitHub Actions builds three downloadable packages:

- Windows 64-bit;
- macOS Apple Silicon;
- macOS Intel.

See `docs/DEPLOYMENT.md` for deployment details and the files in `packaging/` for platform-specific instructions.

## Status

This is a research prototype. Axle counts, train matching and sensor-health indicators require validation before operational use.
