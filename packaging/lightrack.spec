"""How LIGHTRACK is built into one file.

    python -m PyInstaller --noconfirm packaging/lightrack.spec

The counting engines are carried as SOURCE, not as compiled modules, because
the window shows the algorithm it ran and a byte-compiled copy has nothing to
show. That is also why `analysis` is added as data and put on the path at run
time rather than imported here: the built program reads the same files the
repository does.

The checkout has to be called `lightrack`, because that is the name the code
imports itself by.

PyInstaller does not cross-compile: a Windows build has to run on Windows and a
Mac bundle on a Mac. The same spec makes both — `.github/workflows/macos.yml`
runs it on a Mac runner and wraps the result in a disk image.
"""

import shutil
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

REPO = Path(SPECPATH).parent
ROOT = REPO.parent

for junk in REPO.rglob("__pycache__"):
    shutil.rmtree(junk, ignore_errors=True)

datas = [
    (str(REPO / "app" / "static"), "lightrack/app/static"),
    (str(REPO / "analysis"), "lightrack/analysis"),
    (str(REPO / "config.toml"), "lightrack"),
]

asude = REPO / ".sides" / "asude" / "sensor12_peak_tracker.py"
if asude.exists():
    datas.append((str(asude), "lightrack/.sides/asude"))

datas += collect_data_files("tzdata")

hidden = ["scipy.signal", "scipy.interpolate", "scipy.ndimage", "scipy.stats",
          "zoneinfo", "tzdata"]

a = Analysis(
    [str(REPO / "packaging" / "run_lightrack.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6",
              "IPython", "notebook", "jupyter", "pytest", "sphinx",
              "cv2", "sklearn",
              # pandas offers a numba engine nobody here asks for, and the
              # hook collects its LLVM backend anyway: 115 MB of compiler in
              # a program that never compiles anything.
              "numba", "llvmlite", "plotly"],
    noarchive=False,
)

pyz = PYZ(a.pure)

MAC = sys.platform == "darwin"

exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name="LIGHTRACK",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

if MAC:
    app = BUNDLE(
        exe,
        name="LIGHTRACK.app",
        icon=None,
        bundle_identifier="be.umons.lightrack",
        info_plist={
            "CFBundleName": "LIGHTRACK",
            "CFBundleDisplayName": "LIGHTRACK",
            "CFBundleShortVersionString": "1.2",
            "CFBundleVersion": "1.2",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
        },
    )
