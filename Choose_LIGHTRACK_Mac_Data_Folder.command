#!/bin/bash
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
"$HERE/LIGHTRACK.app/Contents/MacOS/LIGHTRACK" --choose-data-folder
