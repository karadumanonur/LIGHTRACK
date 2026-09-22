"""Start the local dashboard, wait for readiness, and open the browser."""

import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import webbrowser


DIRECTORY = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8767/"


def ready():
    try:
        with urllib.request.urlopen(URL + "_stcore/health", timeout=1) as response:
            return response.status == 200
    except Exception:
        return False


def main():
    process = None
    if not ready():
        print("Starting Train Monitoring...")
        with (DIRECTORY / "streamlit_latest.out.log").open("w") as output, (
            DIRECTORY / "streamlit_latest.err.log"
        ).open("w") as errors:
            process = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run",
                 str(DIRECTORY / "fbg_analysis_dashboard_strict.py"),
                 "--server.port", "8767", "--server.address", "127.0.0.1",
                 "--server.headless", "true"],
                cwd=DIRECTORY,
                stdout=output,
                stderr=errors,
                stdin=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        deadline = time.monotonic() + 30
        while not ready():
            if process.poll() is not None or time.monotonic() >= deadline:
                details = (DIRECTORY / "streamlit_latest.err.log").read_text(errors="replace")
                raise RuntimeError("Dashboard startup failed.\n" + details[-4000:])
            time.sleep(0.5)
    print("Train Monitoring is ready: " + URL)
    if "--no-browser" not in sys.argv:
        try:
            os.startfile(URL)
        except OSError:
            if not webbrowser.open(URL):
                raise RuntimeError("Open this address in your browser: " + URL)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(str(error))
        sys.exit(1)
