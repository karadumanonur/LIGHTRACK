# Asude — FBG railway monitoring code

This branch contains my Python scripts for analysing railway FBG recordings.

The code covers:

- plotting and inspecting individual FBG channels;
- detecting and counting axle peaks;
- comparing alternative peak-detection algorithms;
- matching FBG events with Infrabel Excel axle-count records;
- reconstructing recordings split across adjacent files;
- estimating train direction and velocity from multiple FBG sensors.

Raw FBG recordings, Infrabel data files, generated plots, and result folders are intentionally not included.

## Main scripts

- `sensor12_peak_tracker.py` — signal processing and peak detection.
- `find_different_axle_count.py` — batch axle counting and peak plots.
- `compare_peak_algorithms.py` — benchmark of alternative algorithms.
- `showcase_peak_algorithms.py` — visual comparison for one recording.
- `match_fbg_to_infrabel_excel.py` — matching against the Infrabel Excel data.
- `detect_train_direction_velocity.py` — multi-sensor direction and speed estimation.

## Installation

```powershell
python -m pip install -r requirements.txt
```

Run any script with `--help` to see its available arguments.
