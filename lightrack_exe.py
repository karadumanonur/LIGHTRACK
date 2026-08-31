"""Desktop entry point for the local Train Monitoring Streamlit app."""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def user_data_root() -> Path:
    """Return a writable, platform-appropriate folder for local app state."""
    override = os.environ.get("LIGHTRACK_USER_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "LIGHTRACK"
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "LIGHTRACK"
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "LIGHTRACK"


# A frozen application cannot write Matplotlib's cache beside its bundled
# files. Configure a stable per-user cache before importing the dashboard.
_matplotlib_cache = user_data_root() / "matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_matplotlib_cache)

# This import lets PyInstaller discover the dashboard's scientific imports.
import fbg_analysis_dashboard  # noqa: F401
from streamlit import config as streamlit_config
from streamlit.web import bootstrap


TIMESTAMP_PATTERN = __import__("re").compile(r"\d{8}T\d{6},\d{3}Z\.txt$")


def resource_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def settings_path() -> Path:
    return user_data_root() / "settings.json"


def valid_data_root(path: Path | None) -> bool:
    if path is None or not path.is_dir():
        return False
    try:
        return any(TIMESTAMP_PATTERN.search(item.name) for item in path.rglob("*.txt"))
    except OSError:
        return False


def saved_data_root() -> Path | None:
    path = settings_path()
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig")).get("data_dir")
        return Path(value).expanduser() if value else None
    except (OSError, ValueError, TypeError):
        return None


def choose_data_root(initial: Path | None = None) -> Path:
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askdirectory(
        title="Select the Train Monitoring raw_data folder",
        initialdir=str(initial) if initial and initial.is_dir() else str(Path.home()),
        mustexist=True,
        parent=root,
    )
    if not selected:
        root.destroy()
        raise SystemExit("No dataset folder was selected.")
    chosen = Path(selected).resolve()
    if not valid_data_root(chosen):
        messagebox.showerror(
            "Train Monitoring dataset",
            "The selected folder does not contain timestamped FBG .txt recordings.",
            parent=root,
        )
        root.destroy()
        raise SystemExit(1)
    root.destroy()
    destination = settings_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"data_dir": str(chosen)}, indent=2), encoding="utf-8"
    )
    return chosen


def resolve_data_root(force_choice: bool) -> Path:
    configured = os.environ.get("LIGHTRACK_DATA_DIR")
    candidates = [
        Path(configured).expanduser() if configured else None,
        saved_data_root(),
        Path.home()
        / "OneDrive"
        / "Desktop"
        / "Railway Monitoring"
        / "data"
        / "data"
        / "raw_data",
    ]
    existing = next((path.resolve() for path in candidates if valid_data_root(path)), None)
    return choose_data_root(existing) if force_choice or existing is None else existing


def free_port() -> int:
    for port in range(8501, 8511):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            try:
                candidate.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("No free Train Monitoring port was found between 8501 and 8510.")


def open_when_ready(url: str, port: int) -> None:
    for _ in range(120):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                webbrowser.open(url)
                return
        except OSError:
            time.sleep(0.25)


def main() -> None:
    if "--self-test" in sys.argv[1:]:
        root = resource_root()
        dashboard = root / "fbg_analysis_dashboard.py"
        summary = root / "assets" / "two_period_peak_change_mean_std_pm.csv"
        if not dashboard.is_file():
            raise FileNotFoundError(f"Bundled dashboard was not found: {dashboard}")
        if not summary.is_file():
            raise FileNotFoundError(
                f"Bundled two-period mean/std summary was not found: {summary}"
            )
        print("Train Monitoring desktop package self-test passed.")
        return

    force_choice = "--choose-data-folder" in sys.argv[1:]
    data_root = resolve_data_root(force_choice)
    root = resource_root()
    dashboard = root / "fbg_analysis_dashboard.py"
    summary = root / "assets" / "two_period_peak_change_mean_std_pm.csv"
    if not dashboard.is_file():
        raise FileNotFoundError(f"Bundled dashboard was not found: {dashboard}")

    os.environ["LIGHTRACK_DATA_DIR"] = str(data_root)
    if summary.is_file():
        os.environ["LIGHTRACK_TWO_PERIOD_SUMMARY"] = str(summary)
    os.chdir(root)

    # A frozen Streamlit package is not located under site-packages, which
    # makes Streamlit's automatic check assume development mode. Production
    # mode is required so the packaged frontend is served by the local server.
    streamlit_config.set_option("global.developmentMode", False)

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    print("Train Monitoring — FBG Analysis")
    print(f"Dataset: {data_root}")
    print(f"Address: {url}")
    print("Close this window or press Ctrl+C to stop Train Monitoring.")
    threading.Thread(target=open_when_ready, args=(url, port), daemon=True).start()

    bootstrap.run(
        str(dashboard),
        False,
        [],
        {
            "server.address": "127.0.0.1",
            "server.port": port,
            "server.headless": True,
            "browser.gatherUsageStats": False,
            "global.developmentMode": False,
        },
    )


if __name__ == "__main__":
    main()
