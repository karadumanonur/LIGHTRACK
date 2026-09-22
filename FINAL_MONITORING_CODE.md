# Final Train Monitoring code

Open **Open Latest Train Monitoring.bat** as before. It starts the application at
http://127.0.0.1:8767/. The final application uses these four files:

| File | Responsibility | Start reading here |
| --- | --- | --- |
| `launch_train_monitoring.py` | Start Streamlit, wait for readiness, and open the browser | `main()` |
| `train_monitoring_ui.py` | Existing controls, plots, frequency analysis, two-period overview, exports, and display calculations | `main()`, `calculate_axle_heights()`, `analyse_recording()` |
| `recording_io.py` | Read a recording and select a physical FBG from either stored column order | `load_recording_frame()`, `load_physical_fbg_from_frame()` |
| `axle_detector.py` | All filtering, calibration, polarity, and axle-counting calculations | `detect_axles()` |

```text
launch_train_monitoring
    -> train_monitoring_ui
        -> recording_io: timestamps and selected raw sensor signal
        -> axle_detector: filtered signal, active interval, thresholds, axle positions
        -> train_monitoring_ui: plots, statistics, tables, exports
```

## Follow the detector

`detect_axles()` is placed immediately after the settings and result classes.
Its helpers are organized below it into filtering/event boundaries, background
estimation, and candidate acceptance. It performs the same calculations as the
previous strict detector:

1. Validate the selected sensor signal and fixed 500 Hz timestamps.
2. Remove linear drift and apply the fourth-order 0.5-40 Hz Butterworth filter
   with `sosfiltfilt`.
3. Return zero if the largest absolute filtered excursion from the recording
   median is below 0.020 nm (20 pm).
4. Define one padded active interval from the first and last 20 pm crossings.
5. Estimate correlation length and quiet background location/scale.
6. Find peaks and apply learned height/prominence, evidence, width, valley,
   ringing, and broad-response rules.
7. Restore preliminary candidates whose height and prominence both reach
   20 pm, reapplying ringing and broad-response checks.
8. Return the final accepted positions and diagnostics. FBG 9 evaluates both
   polarities; the other FBGs use positive polarity.

The 20 pm gate is defined once in `AdaptiveSettings.absolute_gate_nm`.
It is not a universal minimum height for every accepted peak.

## Column order

`recording_io.load_physical_fbg_from_frame()` compares the median wavelength in
sensor columns 1 and 12. Ascending order maps physical FBG `n` to column `n`;
descending order maps it to column `13 - n`. Column 0 is time. This decision is
made per recording before detection.

## UI preservation

The existing plotting and UI functions remain unchanged. Frame caching stays
in the UI: two recently loaded recordings and twelve recent sensor analyses.
The UI calls its calculation functions directly; it no longer replaces
functions inside another dashboard module at runtime.

`launch_latest_train_monitoring.py` and `fbg_analysis_dashboard_strict.py` are
small compatibility entry points for old shortcuts and commands. They forward
to the new launcher/UI. Streamlit uses the small `fbg_analysis_dashboard_strict.py`
entry point so the full UI module is imported once rather than redefining it
on each interaction. This entry point performs no calculation or UI changes.
The older `fbg_analysis_dashboard.py`,
`configured_strict_axle_detector.py`, `self_calibrating_axle_detector.py`, and
`sensor12_peak_tracker.py` remain for existing standalone tools and regression
reference; the final UI does not import them.

To start manually from this directory:

```text
python launch_train_monitoring.py
```

Or:

```text
python -m streamlit run fbg_analysis_dashboard_strict.py --server.port 8767
```

## Refactor verification

- All 1,119 previously evaluated manual FBG 1 recording counts are identical.
- Complete detector/UI calculation results and detected-passage limits match
  in 48 physical-sensor checks: all 12 FBGs on a normal, split, manual-zero,
  and Period 2 recording. Both stored column orders are covered.
- Nine Streamlit scenarios render without application exceptions and match
  the previous controls, selections, metrics, chart counts, and table counts.
- All 21 moved numerical definitions are unchanged apart from the entry-point
  name. All 28 retained UI/plotting function definitions are unchanged, and
  the two strict calculation adapters are equivalent after import renaming.

Regression artifacts are in `../../outputs/refactor_verification/`.
