# LIGHTRACK — FBG axle counting

This repository holds my Python code for the LIGHTRACK project, an internship
at UMONS for B-sens. Twelve fibre Bragg gratings are bonded to the rail between
Aiseau and Farciennes, in three heads set 15 m and 10 m apart. Every axle that
rolls over them strains the rail and shifts the Bragg wavelength, and the
interrogator records all twelve channels at 500 Hz.

Starting from those raw recordings, the code answers three questions:

- how many axles went past;
- which way the train was travelling;
- how fast it was going.

It then checks all three against Infrabel's own records, because a number that
has never been compared to anything is not a measurement.

## Scripts

Run them in numeric order — each one writes CSV files that the next one reads.

- `10_records.py` — inventory of the delivered files; cadence and retention.
- `20_heads.py` — groups the twelve channels into the three heads, then proves
  the grouping with geometry.
- `30_speed_direction.py` — speed and direction from a geometry-constrained
  time of flight, plus the fit-score gate.
- `40_axles.py` — axle counting by X-crossing and D-crossing, the safety chain.
- `50_passages.py` — extracts each passage and runs the occupancy balance.
- `60_speed_crosscheck.py` — what the gate excluded, and time-of-flight speed
  against bogie-wheelbase speed.
- `70_reference_tables.py` — builds the reference tables from the Infrabel data.
- `80_validate.py` — axles, direction and speed against those references.

## Data

Recordings are whitespace-separated text files: one time column, twelve
wavelength columns, 500 Hz, about 144 s each. Recording is contiguous, so a file
that is missing was discarded rather than never made — which means the discarded
ones can be counted instead of guessed at.

Two Infrabel sources are used for validation:

- the ACAT axle-counter log at Le Campinaire (section 06 / CAT_KJ);
- the punctuality feed, which timestamps each train at Aiseau and Farciennes.

Raw recordings, Infrabel data files and result folders are intentionally not
included here.

## Results

From the POC1 campaign — 93 records, 14 May to 2 June 2026.

**Axles.** Each passage is matched to the axle counter on time alone; the
matcher never looks at a count, since 72 % of trains carry 12 or 24 axles and a
count-aware matcher would mostly succeed by construction. On the 44 passages
that matched:

| output | exact | rate | 95 % CI |
|---|---|---|---|
| head 1 median vote | 40/44 | 90.9 % | 78.3–97.5 % |
| head 2 median vote | 40/44 | 90.9 % | 78.3–97.5 % |
| diagnostic chain | 43/44 | 97.7 % | 88.0–99.9 % |

Every error sits in a freight consist of more than 32 axles. Below that, both
heads agree with the counter on all 40 passages.

**Direction.** Taken from the order of the two station timestamps, never from
the relation label: 51 of 56 passages are consistent with a single mapping
(91.1 %).

**Speed.** Compared against the station-to-station leg average. That average
covers 2.726 km bracketed by two stops, so it is a lower bound on the speed at
the sensor rather than a target to hit. The ratio comes out at 1.03 overall, and
0.95 on the fastest quartile — the trains least affected by a station stop. So
the sensor reads about 5 % low, and since every distance in the project is speed
times time, the same 5 % rides along with all of them. Fastest train seen:
141 km/h.

## What this does not show

- The ACAT reference runs from 13 May to 24 June 2026 and does not reach July.
- ACAT is incomplete in its own right — 28 of 76 passages have no partner in
  it — so it cannot settle whether a detection was missed, in either direction.
- The punctuality feed only sees passenger services that call at these stops.
  Freight and non-stopping trains are invisible to it.
- The 15 m and 10 m head spacings are imposed by the solver. Their ratio is
  tested by the fit; their absolute scale is not, which is exactly what the
  speed comparison above is poking at.
- Head-to-head agreement measures internal consistency, not accuracy. No SIL4
  claim is made or implied.

## Installation

```powershell
python -m pip install numpy pandas scipy matplotlib openpyxl
```

Point `[paths]` in `config.toml` at your data directory before running anything.
