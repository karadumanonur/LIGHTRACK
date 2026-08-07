"""Match sensor-12 axle detections to Infrabel train records from XML.

FBG filenames and ``recording_utc`` values use UTC (the trailing ``Z``). The
script adds the detected event start time, converts the resulting passage time
to Europe/Brussels local time, and searches Infrabel records whose timestamps
are 4--6 minutes away by default. Axle-count agreement is used to rank the
time-window candidates.
"""

from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


BRUSSELS = ZoneInfo("Europe/Brussels")


def clean_tag(tag: str) -> str:
    """Remove an XML namespace from a tag."""
    return tag.rsplit("}", 1)[-1]


def flatten_element(element: ET.Element, prefix: str = "") -> dict[str, str]:
    """Flatten all leaf values below one possible XML record."""
    values: dict[str, list[str]] = defaultdict(list)

    def visit(node: ET.Element, path: str) -> None:
        children = list(node)
        if not children:
            text = (node.text or "").strip()
            if text:
                values[path].append(text)
            return
        for child in children:
            name = clean_tag(child.tag)
            child_path = f"{path}.{name}" if path else name
            visit(child, child_path)

    visit(element, prefix)
    return {key: " | ".join(items) for key, items in values.items()}


def load_xml_records(
    xml_path: Path, requested_record_tag: str | None
) -> tuple[str, pd.DataFrame]:
    """Find the repeated XML element that most likely represents a train row."""
    root = ET.parse(xml_path).getroot()
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

    for element in root.iter():
        tag = clean_tag(element.tag)
        if requested_record_tag and tag != requested_record_tag:
            continue
        flattened = flatten_element(element)
        if len(flattened) >= 2:
            grouped[tag].append(flattened)

    if requested_record_tag:
        rows = grouped.get(requested_record_tag, [])
        if not rows:
            raise ValueError(
                f"No repeated XML records found with tag {requested_record_tag!r}"
            )
        record_tag = requested_record_tag
    else:
        candidates = []
        for tag, rows in grouped.items():
            if len(rows) < 2:
                continue
            median_fields = float(np.median([len(row) for row in rows]))
            # A train record is normally both repeated and richer in fields than
            # its nested components. Logarithmic count prevents a tiny two-field
            # child node from winning only because it occurs very often.
            score = median_fields * math.log2(len(rows) + 1)
            candidates.append((score, median_fields, len(rows), tag))
        if not candidates:
            raise ValueError(
                "Could not infer repeated XML records. Use --xml-record-tag."
            )
        _, _, _, record_tag = max(candidates)
        rows = grouped[record_tag]

    return record_tag, pd.DataFrame(rows)


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def choose_column(
    columns: list[str], aliases: tuple[str, ...], required: bool = True
) -> str | None:
    """Select an XML column using exact aliases first, then suffix matches."""
    normalized_columns = {column: normalized(column.split(".")[-1]) for column in columns}
    normalized_aliases = tuple(normalized(alias) for alias in aliases)

    for alias in normalized_aliases:
        for column, name in normalized_columns.items():
            if name == alias:
                return column
    for alias in normalized_aliases:
        for column, name in normalized_columns.items():
            if alias in name or name in alias:
                return column

    if required:
        raise ValueError(
            f"Could not identify a column matching {aliases}. "
            "Run with --inspect-xml, then provide the corresponding --xml-*-field option."
        )
    return None


def get_named_column(frame: pd.DataFrame, requested: str | None) -> str | None:
    if requested is None:
        return None
    if requested in frame.columns:
        return requested
    matches = [column for column in frame.columns if normalized(column) == normalized(requested)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"XML field {requested!r} was not found. Available fields: "
        + ", ".join(frame.columns)
    )


def parse_local_datetimes(values: pd.Series, fmt: str | None) -> pd.Series:
    parsed = pd.to_datetime(values, format=fmt, errors="coerce", dayfirst=True)
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(
            BRUSSELS, ambiguous="NaT", nonexistent="shift_forward"
        )
    return parsed.dt.tz_convert(BRUSSELS)


def prepare_xml_trains(
    frame: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    columns = list(frame.columns)

    axle_column = get_named_column(frame, args.xml_axle_field) or choose_column(
        columns,
        (
            "axle_count",
            "axles",
            "number_of_axles",
            "aantal_assen",
            "aantalassen",
            "nb_essieux",
            "essieux",
        ),
    )
    train_column = get_named_column(frame, args.xml_train_field) or choose_column(
        columns,
        ("train_number", "train_no", "train_id", "treinnummer", "ritnummer"),
        required=False,
    )

    datetime_column = get_named_column(frame, args.xml_datetime_field)
    date_column = get_named_column(frame, args.xml_date_field)
    time_column = get_named_column(frame, args.xml_time_field)

    if datetime_column is None and date_column is None and time_column is None:
        datetime_column = choose_column(
            columns,
            (
                "datetime",
                "timestamp",
                "passage_datetime",
                "passage_time",
                "passing_time",
                "event_time",
                "measurement_time",
            ),
            required=False,
        )
        if datetime_column is None:
            date_column = choose_column(
                columns, ("date", "datum", "traffic_date"), required=False
            )
            time_column = choose_column(
                columns, ("time", "tijd", "heure"), required=False
            )

    if datetime_column:
        datetime_values = frame[datetime_column].astype(str)
    elif date_column and time_column:
        datetime_values = frame[date_column].astype(str) + " " + frame[time_column].astype(str)
    else:
        raise ValueError(
            "Could not identify the Infrabel timestamp. Run with --inspect-xml and "
            "provide --xml-datetime-field, or both --xml-date-field and --xml-time-field."
        )

    prepared = frame.copy()
    prepared["_infrabel_local"] = parse_local_datetimes(
        datetime_values, args.xml_datetime_format
    )
    prepared["_xml_axles"] = pd.to_numeric(
        prepared[axle_column].astype(str).str.extract(r"(-?\d+(?:[.,]\d+)?)")[0].str.replace(
            ",", ".", regex=False
        ),
        errors="coerce",
    )
    prepared["_train_id"] = (
        prepared[train_column].astype(str) if train_column else prepared.index.astype(str)
    )
    prepared = prepared.dropna(subset=["_infrabel_local", "_xml_axles"])
    prepared["_xml_axles"] = prepared["_xml_axles"].round().astype(int)
    return prepared, {
        "datetime": datetime_column,
        "date": date_column,
        "time": time_column,
        "axles": axle_column,
        "train": train_column,
    }


def prepare_fbg_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"file", "recording_utc", "peak_count", "event_start_s"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"FBG summary is missing columns: {', '.join(missing)}")

    frame = frame.copy()
    frame["_recording_utc"] = pd.to_datetime(frame["recording_utc"], utc=True, errors="coerce")
    frame["event_start_s"] = pd.to_numeric(frame["event_start_s"], errors="coerce")
    frame["peak_count"] = pd.to_numeric(frame["peak_count"], errors="coerce")
    frame["_fbg_pass_utc"] = frame["_recording_utc"] + pd.to_timedelta(
        frame["event_start_s"], unit="s"
    )
    frame["_fbg_pass_local"] = frame["_fbg_pass_utc"].dt.tz_convert(BRUSSELS)
    return frame.dropna(subset=["_fbg_pass_local", "peak_count"])


def within_time_window(delta_min: pd.Series, args: argparse.Namespace) -> pd.Series:
    if args.direction == "infrabel_after":
        comparison = delta_min
    elif args.direction == "infrabel_before":
        comparison = -delta_min
    else:
        comparison = delta_min.abs()
    return comparison.between(args.offset_min, args.offset_max, inclusive="both")


def match_records(
    fbg: pd.DataFrame, trains: pd.DataFrame, args: argparse.Namespace
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows: list[dict] = []

    for fbg_row in fbg.itertuples(index=False):
        delta_min = (
            trains["_infrabel_local"] - fbg_row._fbg_pass_local
        ).dt.total_seconds() / 60.0
        candidates = trains.loc[within_time_window(delta_min, args)].copy()
        candidate_delta = delta_min.loc[candidates.index]

        if args.require_axle_match:
            candidates = candidates[candidates["_xml_axles"].eq(round(fbg_row.peak_count))]
            candidate_delta = candidate_delta.loc[candidates.index]

        if candidates.empty:
            all_rows.append(
                {
                    "fbg_file": fbg_row.file,
                    "fbg_pass_utc": fbg_row._fbg_pass_utc.isoformat(),
                    "fbg_pass_local": fbg_row._fbg_pass_local.isoformat(),
                    "detected_axles": int(round(fbg_row.peak_count)),
                    "match_status": "no_candidate",
                }
            )
            continue

        for index, train in candidates.iterrows():
            signed_delta = float(candidate_delta.loc[index])
            axle_difference = int(train["_xml_axles"] - round(fbg_row.peak_count))
            all_rows.append(
                {
                    "fbg_file": fbg_row.file,
                    "fbg_pass_utc": fbg_row._fbg_pass_utc.isoformat(),
                    "fbg_pass_local": fbg_row._fbg_pass_local.isoformat(),
                    "detected_axles": int(round(fbg_row.peak_count)),
                    "infrabel_train_id": train["_train_id"],
                    "infrabel_time_local": train["_infrabel_local"].isoformat(),
                    "infrabel_axles": int(train["_xml_axles"]),
                    "signed_time_difference_min": signed_delta,
                    "absolute_time_difference_min": abs(signed_delta),
                    "axle_difference": axle_difference,
                    "exact_axle_match": axle_difference == 0,
                    "time_error_from_5_min": abs(abs(signed_delta) - args.target_offset),
                    "match_status": "candidate",
                }
            )

    all_matches = pd.DataFrame(all_rows)
    candidates = all_matches[all_matches["match_status"].eq("candidate")].copy()
    if len(candidates):
        candidates["_abs_axle_difference"] = candidates["axle_difference"].abs()
        candidates = candidates.sort_values(
            [
                "fbg_file",
                "exact_axle_match",
                "_abs_axle_difference",
                "time_error_from_5_min",
            ],
            ascending=[True, False, True, True],
        )
        candidates["candidate_rank"] = candidates.groupby("fbg_file").cumcount() + 1
        all_matches = all_matches.merge(
            candidates[["fbg_file", "infrabel_train_id", "candidate_rank"]],
            on=["fbg_file", "infrabel_train_id"],
            how="left",
        )

    best_candidates = candidates[candidates["candidate_rank"].eq(1)].drop(
        columns=["_abs_axle_difference"], errors="ignore"
    )
    no_candidates = all_matches[all_matches["match_status"].eq("no_candidate")]
    best = pd.concat([best_candidates, no_candidates], ignore_index=True, sort=False)
    return all_matches, best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match FBG axle detections with Infrabel XML train records."
    )
    parser.add_argument("xml_file", type=Path, help="Infrabel axle-count XML")
    parser.add_argument(
        "--fbg-summary", type=Path, default=Path("new_axle_results.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("fbg_infrabel_matches.csv"))
    parser.add_argument(
        "--best-output", type=Path, default=Path("fbg_infrabel_best_matches.csv")
    )
    parser.add_argument("--offset-min", type=float, default=4.0)
    parser.add_argument("--offset-max", type=float, default=6.0)
    parser.add_argument("--target-offset", type=float, default=5.0)
    parser.add_argument(
        "--direction",
        choices=("either", "infrabel_before", "infrabel_after"),
        default="either",
        help="Direction of the 4--6 minute difference (default: either)",
    )
    parser.add_argument("--require-axle-match", action="store_true")
    parser.add_argument("--inspect-xml", action="store_true")
    parser.add_argument("--xml-record-tag")
    parser.add_argument("--xml-datetime-field")
    parser.add_argument("--xml-date-field")
    parser.add_argument("--xml-time-field")
    parser.add_argument("--xml-datetime-format")
    parser.add_argument("--xml-axle-field")
    parser.add_argument("--xml-train-field")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xml_path = args.xml_file.expanduser().resolve()
    record_tag, xml_frame = load_xml_records(xml_path, args.xml_record_tag)
    print(f"XML record tag: {record_tag!r}; records: {len(xml_frame)}")
    print("XML fields:")
    for column in xml_frame.columns:
        print(f"  {column}")

    if args.inspect_xml:
        print("\nFirst XML records:")
        print(xml_frame.head().to_string(index=False))
        return

    trains, selected = prepare_xml_trains(xml_frame, args)
    print("Selected XML fields:")
    for purpose, column in selected.items():
        print(f"  {purpose}: {column}")
    print(f"Usable Infrabel records: {len(trains)}")

    fbg = prepare_fbg_summary(args.fbg_summary.expanduser().resolve())
    all_matches, best_matches = match_records(fbg, trains, args)

    output_path = args.output.expanduser().resolve()
    best_output_path = args.best_output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_output_path.parent.mkdir(parents=True, exist_ok=True)
    all_matches.to_csv(output_path, index=False, float_format="%.4f")
    best_matches.to_csv(best_output_path, index=False, float_format="%.4f")

    matched = best_matches["match_status"].eq("candidate").sum()
    exact = best_matches.get("exact_axle_match", pd.Series(dtype=bool)).fillna(False).sum()
    print(f"FBG recordings compared: {len(fbg)}")
    print(f"Best time-window matches: {matched}")
    print(f"Best matches with exact axle count: {exact}")
    print(f"All candidates: {output_path}")
    print(f"Best candidate per FBG recording: {best_output_path}")


if __name__ == "__main__":
    main()
