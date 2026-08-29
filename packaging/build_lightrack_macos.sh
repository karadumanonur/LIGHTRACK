#!/bin/bash
set -euo pipefail

PACKAGING_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$PACKAGING_DIR/.." && pwd)"
DIST_DIR="${LIGHTRACK_MAC_DIST:-$PROJECT_DIR/dist/macos}"
WORK_DIR="${LIGHTRACK_MAC_WORK:-$PROJECT_DIR/build/pyinstaller-macos}"
SUMMARY_NAME="two_period_peak_change_mean_std_pm.csv"
SUMMARY_REPO="$PROJECT_DIR/assets/$SUMMARY_NAME"
SUMMARY_LOCAL="$(cd "$PROJECT_DIR/.." && pwd)/outputs/fbg1_4_5_9_10_two_periods_detrended_axle_peak_pm/$SUMMARY_NAME"

mkdir -p "$DIST_DIR" "$WORK_DIR"

PYINSTALLER_ARGS=(
  --noconfirm
  --clean
  --onedir
  --windowed
  --name LIGHTRACK
  --osx-bundle-identifier be.lightrack.fbg-analysis
  --distpath "$DIST_DIR"
  --workpath "$WORK_DIR"
  --specpath "$WORK_DIR"
  --paths "$PROJECT_DIR"
  --collect-data streamlit
  --collect-submodules streamlit
  --copy-metadata streamlit
  --hidden-import vmdpy
  --hidden-import scipy._external.array_api_compat.numpy.fft
  --add-data "$PROJECT_DIR/fbg_analysis_dashboard.py:."
  --add-data "$PROJECT_DIR/.streamlit/config.toml:.streamlit"
)

if [[ -f "$SUMMARY_REPO" ]]; then
  PYINSTALLER_ARGS+=(--add-data "$SUMMARY_REPO:assets")
elif [[ -f "$SUMMARY_LOCAL" ]]; then
  PYINSTALLER_ARGS+=(--add-data "$SUMMARY_LOCAL:assets")
else
  echo "Note: the optional two-period summary was not bundled."
fi

python -m PyInstaller "${PYINSTALLER_ARGS[@]}" "$PROJECT_DIR/lightrack_exe.py"

APP="$DIST_DIR/LIGHTRACK.app"
"$APP/Contents/MacOS/LIGHTRACK" --self-test
echo "Created $APP"
