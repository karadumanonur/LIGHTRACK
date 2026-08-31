"""Research utility for matching FBG detections to Infrabel ACAT records.

The FBG recording timestamp is UTC (``Z``). It is converted to Brussels local
time before matching. Consecutive ACAT counter updates are first collapsed into
complete train events (zero -> axle count -> zero). Either event boundary may
then satisfy the default 3--7 minute window. Exact axle-count agreement is the
strongest ranking criterion.
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sensor12_peak_tracker import (
    butterworth_process,
    detect_peaks,
    load_fbg_file,
    make_peak_table,
    plot_results,
    sampling_rate,
)


BRUSSELS = ZoneInfo("Europe/Brussels")
INFRABEL_COLUMNS = [
    "infrabel_time_local",
    "source",
    "event_code",
    "metric",
    "counter_before",
    "counter_after",
]
CONFIDENCE_FOLDERS = {
    "high": "match high confidence",
    "medium": "match medium confidence",
    "low": "match low confidence",
    "none": "no match candidates",
}


def group_infrabel_counter_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse continuous ACAT counter updates into candidate train events.

    A sequence is continuous only when the next row's ``counter_before`` is
    equal to the preceding row's ``counter_after``.  A mismatch marks missing
    data or a different counter event and must not be bridged merely because a
    later row eventually returns to zero.
    """
    events: list[dict] = []
    active: list[pd.Series] = []

    def finish(rows: list[pd.Series]) -> None:
        if not rows:
            return
        block = pd.DataFrame(rows).sort_values("infrabel_time_local")
        first = block.iloc[0]
        last = block.iloc[-1]
        events.append(
            {
                "infrabel_event_id": len(events) + 1,
                "infrabel_start_local": first["infrabel_time_local"],
                "infrabel_end_local": last["infrabel_time_local"],
                "infrabel_axles": int(
                    round(block[["counter_before", "counter_after"]].max().max())
                ),
                "start_excel_row": int(first["excel_row"]),
                "end_excel_row": int(last["excel_row"]),
                "counter_update_count": len(block),
                "complete_counter_cycle": bool(
                    first["counter_before"] == 0 and last["counter_after"] == 0
                ),
                "start_direction_last_axle": first.get("direction_last_axle", np.nan),
                "end_direction_last_axle": last.get("direction_last_axle", np.nan),
                "source": first["source"],
            }
        )

    for _, row in frame.sort_values("infrabel_time_local").iterrows():
        if active:
            previous_after = active[-1]["counter_after"]
            counter_is_continuous = bool(row["counter_before"] == previous_after)
            starts_new_cycle = bool(
                row["counter_before"] == 0 and row["counter_after"] > 0
            )
            if not counter_is_continuous or starts_new_cycle:
                finish(active)
                active = []

        active.append(row)
        if row["counter_after"] == 0 and row["counter_before"] > 0:
            finish(active)
            active = []
    finish(active)

    result = pd.DataFrame(events)
    result["counter_cycle_duration_min"] = (
        result["infrabel_end_local"] - result["infrabel_start_local"]
    ).dt.total_seconds() / 60.0
    return result


def load_infrabel_workbook(path: Path, sheet: str, skiprows: int) -> pd.DataFrame:
    """Read ACAT rows and reconstruct complete Infrabel train events."""
    frame = pd.read_excel(
        path,
        sheet_name=sheet,
        header=None,
        skiprows=skiprows,
        usecols="A:F",
        names=INFRABEL_COLUMNS,
        engine="openpyxl",
    )
    # Excel row number is useful when manually checking a match in the source.
    frame["excel_row"] = np.arange(skiprows + 1, skiprows + 1 + len(frame))
    frame["infrabel_time_local"] = pd.to_datetime(
        frame["infrabel_time_local"], errors="coerce"
    )
    direction_rows = frame[
        frame["metric"].astype(str).str.strip().str.casefold().eq("directionlastaxle")
    ].drop_duplicates("infrabel_time_local")
    direction_rows = direction_rows.assign(
        direction_last_axle=(
            direction_rows["counter_before"].astype(str)
            + "_to_"
            + direction_rows["counter_after"].astype(str)
        )
    )
    direction_map = direction_rows.set_index("infrabel_time_local")[
        "direction_last_axle"
    ]

    axle_rows = frame[
        frame["metric"].astype(str).str.strip().str.casefold().eq("axlecount")
    ].copy()
    axle_rows["counter_before"] = pd.to_numeric(
        axle_rows["counter_before"], errors="coerce"
    )
    axle_rows["counter_after"] = pd.to_numeric(
        axle_rows["counter_after"], errors="coerce"
    )
    axle_rows = axle_rows.dropna(
        subset=["infrabel_time_local", "counter_before", "counter_after"]
    )
    axle_rows["direction_last_axle"] = axle_rows["infrabel_time_local"].map(
        direction_map
    )
    return group_infrabel_counter_events(axle_rows)


def merge_split_events(frame: pd.DataFrame, maximum_gap_s: float) -> pd.DataFrame:
    """Combine events continued in the immediately following FBG frame."""
    if frame.empty:
        return frame

    ordered = frame.sort_values("event_time_utc").reset_index(drop=True)
    merged: list[pd.Series] = []
    current = ordered.iloc[0].copy()
    current_files = [str(current["file"])]
    current_count = 1

    for _, following in ordered.iloc[1:].iterrows():
        gap_s = (following["event_time_utc"] - current["event_end_utc"]).total_seconds()
        if -maximum_gap_s <= gap_s <= maximum_gap_s:
            current_files.append(str(following["file"]))
            current_count += 1
            current["detected_axles"] += following["detected_axles"]
            current["peak_count"] = current["detected_axles"]
            current["event_end_utc"] = max(
                current["event_end_utc"], following["event_end_utc"]
            )
            current["event_end_s"] = (
                current["event_end_utc"] - current["recording_time_utc"]
            ).total_seconds()
        else:
            current["file"] = " | ".join(current_files)
            current["split_frame_count"] = current_count
            merged.append(current)
            current = following.copy()
            current_files = [str(current["file"])]
            current_count = 1

    current["file"] = " | ".join(current_files)
    current["split_frame_count"] = current_count
    merged.append(current)
    return pd.DataFrame(merged).reset_index(drop=True)


def load_fbg_summary(
    path: Path, time_reference: str, merge_gap_s: float | None
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"file", "peak_count", "event_start_s", "event_end_s"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"FBG summary is missing columns: {', '.join(missing)}")
    recording_column = (
        "recording_utc" if "recording_utc" in frame.columns else "recording_time_utc"
    )
    if recording_column not in frame.columns:
        raise ValueError(
            "FBG summary is missing a recording timestamp column "
            "(recording_utc or recording_time_utc)"
        )

    frame = frame.copy()
    frame["recording_time_utc"] = pd.to_datetime(
        frame[recording_column], format="mixed", utc=True, errors="coerce"
    )
    frame["event_start_s"] = pd.to_numeric(frame["event_start_s"], errors="coerce")
    frame["event_end_s"] = pd.to_numeric(frame["event_end_s"], errors="coerce")
    frame["detected_axles"] = pd.to_numeric(frame["peak_count"], errors="coerce")
    frame["event_time_utc"] = frame["recording_time_utc"] + pd.to_timedelta(
        frame["event_start_s"], unit="s"
    )
    frame["event_end_utc"] = frame["recording_time_utc"] + pd.to_timedelta(
        frame["event_end_s"], unit="s"
    )
    frame = frame.dropna(
        subset=["recording_time_utc", "event_time_utc", "event_end_utc", "detected_axles"]
    )
    if "split_frame_count" not in frame.columns:
        frame["split_frame_count"] = 1
    if merge_gap_s is not None:
        frame = merge_split_events(frame, merge_gap_s)
    frame["recording_time_local"] = frame["recording_time_utc"].dt.tz_convert(BRUSSELS)
    frame["event_time_local"] = frame["event_time_utc"].dt.tz_convert(BRUSSELS)
    frame["comparison_time_local"] = frame[
        "recording_time_local" if time_reference == "recording" else "event_time_local"
    ]
    return frame.dropna(subset=["comparison_time_local", "detected_axles"])


def time_window_mask(delta_min: pd.Series, args: argparse.Namespace) -> pd.Series:
    if args.direction == "infrabel_after":
        comparable = delta_min
    elif args.direction == "infrabel_before":
        comparable = -delta_min
    else:
        comparable = delta_min.abs()
    return comparable.between(args.offset_min, args.offset_max, inclusive="both")


def candidate_row(
    fbg_row,
    infrabel_row,
    target_offset: float,
) -> dict:
    detected = int(round(fbg_row.detected_axles))
    axle_difference = int(infrabel_row.infrabel_axles - detected)
    return {
        "fbg_file": fbg_row.file,
        "recording_time_utc": fbg_row.recording_time_utc.isoformat(),
        "recording_time_local": fbg_row.recording_time_local.isoformat(),
        "event_start_s": fbg_row.event_start_s,
        "event_time_local": fbg_row.event_time_local.isoformat(),
        "comparison_time_local": fbg_row.comparison_time_local.isoformat(),
        "detected_axles": detected,
        "split_frame_count": int(fbg_row.split_frame_count),
        "infrabel_event_id": int(infrabel_row.infrabel_event_id),
        "infrabel_start_local": infrabel_row.infrabel_start_local.isoformat(),
        "infrabel_end_local": infrabel_row.infrabel_end_local.isoformat(),
        "infrabel_time_local": infrabel_row._matched_time.isoformat(),
        "matched_counter_boundary": infrabel_row._matched_boundary,
        "infrabel_axles": int(infrabel_row.infrabel_axles),
        "start_excel_row": int(infrabel_row.start_excel_row),
        "end_excel_row": int(infrabel_row.end_excel_row),
        "counter_update_count": int(infrabel_row.counter_update_count),
        "complete_counter_cycle": bool(infrabel_row.complete_counter_cycle),
        "counter_cycle_duration_min": infrabel_row.counter_cycle_duration_min,
        "start_direction_last_axle": infrabel_row.start_direction_last_axle,
        "end_direction_last_axle": infrabel_row.end_direction_last_axle,
        "signed_time_difference_min": infrabel_row._signed_delta,
        "absolute_time_difference_min": abs(infrabel_row._signed_delta),
        "axle_difference": axle_difference,
        "exact_axle_match": axle_difference == 0,
        "time_error_from_target_min": abs(
            abs(infrabel_row._signed_delta) - target_offset
        ),
        "match_status": "candidate",
    }


def match_fbg_to_infrabel(
    fbg: pd.DataFrame, infrabel: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_rows: list[dict] = []

    for fbg_row in fbg.itertuples(index=False):
        # Infrabel timestamps are local and naive in the supplied workbook.
        comparison_naive = fbg_row.comparison_time_local.replace(tzinfo=None)
        start_deltas = (
            infrabel["infrabel_start_local"] - comparison_naive
        ).dt.total_seconds() / 60.0
        end_deltas = (
            infrabel["infrabel_end_local"] - comparison_naive
        ).dt.total_seconds() / 60.0
        valid_start = time_window_mask(start_deltas, args)
        valid_end = time_window_mask(end_deltas, args)
        candidates = infrabel.loc[valid_start | valid_end].copy()

        # If both counter boundaries are in range, retain the one closest to
        # the calibrated target offset. The pair still remains one candidate.
        candidates["_use_start"] = valid_start.loc[candidates.index] & (
            ~valid_end.loc[candidates.index]
            | (
                (start_deltas.loc[candidates.index].abs() - args.target_offset).abs()
                <= (end_deltas.loc[candidates.index].abs() - args.target_offset).abs()
            )
        )
        candidates["_matched_boundary"] = np.where(
            candidates["_use_start"], "start", "end"
        )
        candidates["_signed_delta"] = np.where(
            candidates["_use_start"],
            start_deltas.loc[candidates.index],
            end_deltas.loc[candidates.index],
        )
        candidates["_matched_time"] = np.where(
            candidates["_use_start"],
            candidates["infrabel_start_local"],
            candidates["infrabel_end_local"],
        )

        if args.require_axle_match:
            candidates = candidates[
                candidates["infrabel_axles"].eq(round(fbg_row.detected_axles))
            ]

        if candidates.empty:
            output_rows.append(
                {
                    "fbg_file": fbg_row.file,
                    "recording_time_utc": fbg_row.recording_time_utc.isoformat(),
                    "recording_time_local": fbg_row.recording_time_local.isoformat(),
                    "event_start_s": fbg_row.event_start_s,
                    "event_time_local": fbg_row.event_time_local.isoformat(),
                    "comparison_time_local": fbg_row.comparison_time_local.isoformat(),
                    "detected_axles": int(round(fbg_row.detected_axles)),
                    "split_frame_count": int(fbg_row.split_frame_count),
                    "match_status": "no_candidate",
                }
            )
            continue

        for _, infrabel_row in candidates.iterrows():
            output_rows.append(
                candidate_row(
                    fbg_row,
                    infrabel_row,
                    args.target_offset,
                )
            )

    all_matches = pd.DataFrame(output_rows)
    candidate_mask = all_matches["match_status"].eq("candidate")
    candidates = all_matches.loc[candidate_mask].copy()

    if len(candidates):
        candidates["_absolute_axle_difference"] = candidates["axle_difference"].abs()
        candidates = candidates.sort_values(
            [
                "fbg_file",
                "exact_axle_match",
                "_absolute_axle_difference",
                "time_error_from_target_min",
            ],
            ascending=[True, False, True, True],
        )
        candidates["candidate_rank"] = candidates.groupby("fbg_file").cumcount() + 1
        exact_mask = candidates["exact_axle_match"].eq(True)
        exact_counts = (
            candidates[exact_mask]
            .groupby("fbg_file")
            .size()
        )
        candidates["exact_candidate_count"] = (
            candidates["fbg_file"].map(exact_counts).fillna(0).astype(int)
        )
        candidates["match_confidence"] = np.select(
            [
                exact_mask
                & candidates["complete_counter_cycle"].eq(True)
                & candidates["exact_candidate_count"].eq(1),
                exact_mask,
            ],
            ["high", "medium"],
            default="low",
        )
        rank_lookup = candidates.set_index(
            ["fbg_file", "infrabel_event_id"]
        )["candidate_rank"]
        all_matches["candidate_rank"] = [
            rank_lookup.get((row.fbg_file, row.infrabel_event_id), np.nan)
            if row.match_status == "candidate"
            else np.nan
            for row in all_matches.itertuples(index=False)
        ]
        confidence_lookup = candidates.set_index(
            ["fbg_file", "infrabel_event_id"]
        )["match_confidence"]
        exact_count_lookup = candidates.set_index(
            ["fbg_file", "infrabel_event_id"]
        )["exact_candidate_count"]
        all_matches["match_confidence"] = [
            confidence_lookup.get((row.fbg_file, row.infrabel_event_id), "none")
            if row.match_status == "candidate"
            else "none"
            for row in all_matches.itertuples(index=False)
        ]
        all_matches["exact_candidate_count"] = [
            exact_count_lookup.get((row.fbg_file, row.infrabel_event_id), 0)
            if row.match_status == "candidate"
            else 0
            for row in all_matches.itertuples(index=False)
        ]

    best_candidates = candidates[candidates["candidate_rank"].eq(1)].drop(
        columns=["_absolute_axle_difference"], errors="ignore"
    )
    unmatched = all_matches[all_matches["match_status"].eq("no_candidate")]
    best_matches = pd.concat(
        [best_candidates, unmatched], ignore_index=True, sort=False
    ).sort_values("recording_time_utc")
    return all_matches, best_matches


def create_numbered_peak_plot(input_path: Path, output_path: Path) -> int:
    """Regenerate the validated sensor-12 plot for an observation folder."""
    time_s, raw_nm = load_fbg_file(input_path, 12)
    fs_hz = sampling_rate(time_s)
    processed_nm = butterworth_process(raw_nm, fs_hz, 0.5, 40.0)
    indices, signs, _ = detect_peaks(
        processed_nm,
        fs_hz,
        "positive",
        8.0,
        None,
        0.06,
        0.04,
    )
    peaks = make_peak_table(time_s, raw_nm, processed_nm, indices, signs)
    plot_results(
        input_path,
        time_s,
        raw_nm,
        processed_nm,
        peaks,
        output_path,
        12,
        "filter",
        False,
    )
    return len(peaks)


def display_value(value, digits: int = 2) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def display_duration_minutes(value) -> str:
    """Format decimal minutes as an unambiguous minutes-and-seconds duration."""
    try:
        total_seconds = round(abs(float(value)) * 60.0)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(total_seconds):
        return "—"
    minutes, seconds = divmod(int(total_seconds), 60)
    return f"{minutes} min {seconds:02d} s"


def write_html_gallery(
    folder: Path,
    confidence: str,
    entries: list[dict],
) -> None:
    cards = []
    for entry in entries:
        row = entry["row"]
        images = "".join(
            f'<img src="{html.escape(name)}" alt="FBG numbered peak plot">'
            for name in entry["plots"]
        )
        cards.append(
            f"""
            <article>
              <h2>{html.escape(str(row['fbg_file']))}</h2>
              <div class="facts">
                <span>FBG axles: <b>{display_value(row.get('detected_axles'), 0)}</b></span>
                <span>Infrabel axles: <b>{display_value(row.get('infrabel_axles'), 0)}</b></span>
                <span>Time difference: <b>{display_duration_minutes(row.get('absolute_time_difference_min'))}</b></span>
                <span>Boundary: <b>{html.escape(display_value(row.get('matched_counter_boundary')))}</b></span>
                <span>Split frames: <b>{display_value(row.get('split_frame_count'), 0)}</b></span>
              </div>
              <p><a href="{html.escape(entry['details'])}">Open match details CSV</a></p>
              {images}
            </article>
            """
        )
    title = CONFIDENCE_FOLDERS[confidence].title()
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background:#f4f7fa; color:#172033; }}
h1 {{ margin-bottom:4px; }} .summary {{ color:#526078; margin-top:0; }}
article {{ background:white; padding:18px; margin:20px 0; border-radius:10px; box-shadow:0 2px 10px #0001; }}
.facts {{ display:flex; flex-wrap:wrap; gap:8px 20px; margin-bottom:8px; }}
img {{ display:block; max-width:100%; height:auto; margin:14px 0; border:1px solid #d9e0e8; }}
a {{ color:#075fa8; }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p class="summary">{len(entries)} FBG train events. See matches.csv for the complete table.</p>
{''.join(cards)}
</body></html>"""
    (folder / "index.html").write_text(document, encoding="utf-8")


def organize_observation_folders(
    best_matches: pd.DataFrame,
    raw_data_dir: Path,
    output_root: Path,
) -> None:
    """Create confidence-ranked folders containing graphs and match evidence."""
    output_root.mkdir(parents=True, exist_ok=True)
    errors: list[dict] = []
    root_links = []

    for confidence, folder_name in CONFIDENCE_FOLDERS.items():
        folder = output_root / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        group = best_matches[
            best_matches["match_confidence"].fillna("none").eq(confidence)
        ].copy()
        group.to_csv(folder / "matches.csv", index=False, float_format="%.4f")
        entries: list[dict] = []

        for _, row in group.iterrows():
            source_names = [part.strip() for part in str(row["fbg_file"]).split("|")]
            first_stem = Path(source_names[0]).stem
            event_stem = (
                first_stem
                if len(source_names) == 1
                else f"{first_stem}__plus_{len(source_names) - 1}_frames"
            )
            details_name = f"{event_stem}.match.csv"
            pd.DataFrame([row]).to_csv(
                folder / details_name, index=False, float_format="%.4f"
            )
            plot_names = []
            for source_name in source_names:
                source_path = raw_data_dir / source_name
                plot_name = f"{Path(source_name).stem}_sensor12_peaks.png"
                try:
                    create_numbered_peak_plot(source_path, folder / plot_name)
                    plot_names.append(plot_name)
                except Exception as exc:
                    errors.append(
                        {
                            "confidence": confidence,
                            "fbg_file": source_name,
                            "error": str(exc),
                        }
                    )
            entries.append(
                {"row": row, "plots": plot_names, "details": details_name}
            )

        write_html_gallery(folder, confidence, entries)
        root_links.append(
            f'<li><a href="{html.escape(folder_name)}/index.html">{html.escape(folder_name.title())}</a> '
            f'({len(group)} events)</li>'
        )

    if errors:
        pd.DataFrame(errors).to_csv(output_root / "plot_errors.csv", index=False)
    root_index = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>FBG–Infrabel Match Review</title></head>
<body style="font-family:Segoe UI,Arial,sans-serif;margin:32px">
<h1>FBG–Infrabel Match Review</h1><ul>{''.join(root_links)}</ul></body></html>"""
    (output_root / "index.html").write_text(root_index, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match FBG peak counts to the Infrabel ACAT axle-count workbook."
    )
    parser.add_argument("infrabel_workbook", type=Path)
    parser.add_argument(
        "--fbg-summary", type=Path, default=Path("new_axle_results.csv")
    )
    parser.add_argument("--sheet", default="Comptage ACAT KJ")
    parser.add_argument("--skiprows", type=int, default=5)
    parser.add_argument(
        "--time-reference",
        choices=("recording", "event"),
        default="event",
        help=(
            "Use the UTC filename/recording time or detected event time. "
            "Default: event, so long pre-trigger recordings do not shift matches."
        ),
    )
    parser.add_argument(
        "--offset-min",
        type=float,
        default=1.5,
        help="Minimum absolute FBG/Infrabel offset in minutes (default: 1.5)",
    )
    parser.add_argument("--offset-max", type=float, default=7.0)
    parser.add_argument("--target-offset", type=float, default=4.5)
    parser.add_argument(
        "--direction",
        choices=("either", "infrabel_before", "infrabel_after"),
        default="either",
    )
    parser.add_argument(
        "--require-axle-match",
        action="store_true",
        help="Discard time candidates whose Infrabel axle count is different",
    )
    parser.add_argument(
        "--merge-gap-sec",
        type=float,
        default=1.0,
        help=(
            "Merge FBG events continued across consecutive frames when their "
            "absolute event intervals are at most this many seconds apart "
            "(default: 1.0; use a negative value to disable)."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("fbg_infrabel_candidates.csv"))
    parser.add_argument(
        "--best-output", type=Path, default=Path("fbg_infrabel_best_matches.csv")
    )
    parser.add_argument(
        "--organize-output",
        action="store_true",
        help="Create confidence folders with match CSVs, peak plots, and HTML galleries",
    )
    parser.add_argument(
        "--organized-dir", type=Path, default=Path("match_observation_folders")
    )
    parser.add_argument(
        "--fbg-data-dir",
        type=Path,
        default=None,
        help="Directory containing raw FBG text files; defaults to the workbook's parent data directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workbook_path = args.infrabel_workbook.expanduser().resolve()
    summary_path = args.fbg_summary.expanduser().resolve()
    infrabel = load_infrabel_workbook(workbook_path, args.sheet, args.skiprows)
    merge_gap = None if args.merge_gap_sec < 0 else args.merge_gap_sec
    fbg = load_fbg_summary(summary_path, args.time_reference, merge_gap)
    all_matches, best_matches = match_fbg_to_infrabel(fbg, infrabel, args)

    output_path = args.output.expanduser().resolve()
    best_output_path = args.best_output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_output_path.parent.mkdir(parents=True, exist_ok=True)
    all_matches.to_csv(output_path, index=False, float_format="%.4f")
    best_matches.to_csv(best_output_path, index=False, float_format="%.4f")

    matched = best_matches["match_status"].eq("candidate").sum()
    exact_values = best_matches.get(
        "exact_axle_match", pd.Series(index=best_matches.index, dtype="boolean")
    ).astype("boolean")
    exact = exact_values.fillna(False).sum()
    print(f"Reconstructed Infrabel train events: {len(infrabel)}")
    print(f"FBG recordings compared: {len(fbg)}")
    print(f"Best time-window matches: {matched}")
    print(f"Best matches with exact axle count: {exact}")
    print(f"All time candidates: {output_path}")
    print(f"Best candidate per FBG recording: {best_output_path}")
    if args.organize_output:
        raw_data_dir = (
            args.fbg_data_dir.expanduser().resolve()
            if args.fbg_data_dir
            else workbook_path.parent.parent
        )
        organized_dir = args.organized_dir.expanduser().resolve()
        organize_observation_folders(best_matches, raw_data_dir, organized_dir)
        print(f"Observation galleries: {organized_dir}")


if __name__ == "__main__":
    main()
