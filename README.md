# LIGHTRACK — a live FBG axle counter

This repository holds the real-time counter from the LIGHTRACK project, an
internship at UMONS for B-sens. Twelve fibre Bragg gratings are bonded to the
rail on line 130 between Aiseau and Farciennes, in three heads set 15 m and 10 m
apart. Every axle that rolls over them strains the rail and shifts the Bragg
wavelength, and the interrogator streams all twelve channels at 500 Hz.

The counter reads that stream a block at a time and reports axles as they pass.
It never looks ahead: every threshold it uses comes from samples already gone by.

## Running it

```powershell
python -m analysis record.txt
```

```
record.txt
   t=   65.3 s    12 axles   X 12 / D 12      86 km/h   confirmed   head 1 first

1 train(s) in 1 file(s)
```

One line per train: when it arrived, how many axles, what each of the two
counting chains said, the speed, whether a second sensor on the same head
confirmed the count, and which head the train reached first.

## Using it on a live feed

The counter consumes blocks and keeps its state between them, so a file and a
live feed look the same to it. Replace the reader with whatever the interrogator
provides:

```python
from analysis import StreamingCounter

c = StreamingCounter()
for block in feed:
    c.push(block)
    for train in c.pop_trains():
        print(train.axles, train.speed_kmh, train.agree)
```

A train is closed once the line has been quiet for three seconds; `pop_trains`
returns the ones finished since the last call, and `flush` closes the one still
in progress.

- `analysis/counter.py` — the counter
- `analysis/reader.py` — reads a record from disk into blocks
- `analysis/__main__.py` — the command line above
- `analysis/config.py`, `analysis/config.toml` — the parameters

## What it does to the signal

Each stage is derived from something measured, not chosen by hand. The width of
one axle peak sets almost everything: an axle loads about 2.2 m of track, so at
a given speed its peak lasts `2.2 / (v/3.6)` seconds, and the filter cutoff, the
minimum time a peak must hold, and the shortest segment the derivative chain
will judge all follow from it.

1. **Baseline** — subtract a trailing 1 s median. Thermal drift moves in
   minutes, so a lagging estimate costs nothing, and the window is nine times
   longer than an axle peak, so the median steps over the train.
2. **Filter** — a 31-tap linear-phase FIR at a speed-adaptive cutoff. Its group
   delay is constant and identical on every channel, so the count is unaffected
   and the head-to-head lag that gives speed survives intact.
3. **Speed** — from the bogie wheelbase at the first head, available about
   0.2 s into a train. It is only as good as the assumed wheelbase, so it exists
   to get a filter running; `set_speed()` takes the time-of-flight value when it
   arrives about a second later.
4. **Threshold** — a running percentile of the peaks already seen. Before any
   peak exists it falls back to the excursion that separates a train from an
   empty record.
5. **Counting** — two independent chains. X-crossing arms above one threshold
   and confirms below another; D-crossing enters lower and tests the shape of
   the peak instead of its height. When they agree, that is the count. When they
   do not, the disagreement is the output — a counting head that cannot decide
   should say so rather than emit a number that might be a phantom axle.

## What it costs

Latency from the first axle to its confirmation is about 245 ms at the median,
which includes the filter delay and waiting for the peak to come back down.

## How well it counts

Measured against Infrabel's own axle counter at Le Campinaire, on the 44 POC1
passages that could be matched to it on time alone — the matcher never reads a
count, which would make the figure circular.

| output | exact |
|---|---|
| PAD 1 alone | 41/44 = 93.2 % |
| head 1 median vote, four sensors | 39/44 = 88.6 % |

Every remaining error is a freight consist over 100 axles, where the axle loads
vary several fold and the sensors disagree with each other by 40 axles or more.
The head reports that disagreement rather than a number.

## What this is not

It is not a safety product and carries no SIL claim; that would need failure
counts three orders of magnitude larger than anything measured here. The head
spacings of 15 m and 10 m are imposed by the geometry, and a comparison against
station-to-station timings suggests the sensor reads about 5 % low, which
applies to every distance derived from speed times time.

The offline analysis pipeline that produced the validation above is kept
separately and is not part of this repository.

## Installation

```powershell
python -m pip install numpy scipy
```
