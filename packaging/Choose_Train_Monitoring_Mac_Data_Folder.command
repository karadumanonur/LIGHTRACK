#!/bin/bash
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
"$HERE/Train Monitoring.app/Contents/MacOS/Train Monitoring" --choose-data-folder
