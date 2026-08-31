# Research scripts

These scripts support experiments and validation but are not required by the
Train Monitoring desktop application.

| Script | Purpose |
|---|---|
| `compare_peak_algorithms.py` | Compare axle-peak detection approaches |
| `showcase_peak_algorithms.py` | Visualise several detectors on one recording |
| `find_different_axle_count.py` | Find recordings with unusual axle counts |
| `match_fbg_to_infrabel_excel.py` | Match FBG events with Infrabel records |
| `detect_train_direction_velocity.py` | Study multi-sensor direction and speed |

Run these commands from the repository root. Use `--help` to see the inputs
required by each experiment, for example:

```powershell
python research\compare_peak_algorithms.py --help
python research\match_fbg_to_infrabel_excel.py --help
```

Generated figures, tables and raw measurement data remain outside version
control.
