# LIGHTRACK

FBG axle counting between Aiseau and Farciennes. An internship at UMONS for
B-sens.

The code lives on the branches, one per person: `Onur-Branch` holds the live
counter, `Asude-Branch` the peak-algorithm and matching studies. `main` holds
only the harness that runs both over the same recordings and prints the two
answers side by side.

```powershell
python compare/run.py path/to/records
```

See [`compare/README.md`](compare/README.md).
