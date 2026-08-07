"""Generate out/code_inventory.md — facts only, no prose."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\onurk\Desktop\data")
LIB = ROOT / "lightrack"
SCR = ROOT / "scripts"
OUT = ROOT / "out"

BLIND_ENTRY = ["41_blind_run.py", "44_blind_report.py", "45_blind_selfdiag.py"]
NEG_KEYS = ["NEGATIVE", "negative result", "ELIMINATED", "eliminated",
            "does not", "DOES NOT", "cannot", "CANNOT", "not identifiable",
            "EXCLUDED", "excluded", "refuted", "REFUTED", "fails", "FAILS",
            "void", "withdrawn", "WITHDRAWN", "too strong", "not a fix",
            "uninterpretable", "no oracle", "not an oracle", "never fires"]


def classify(path: Path) -> dict:
    """raw / blank / comment / docstring / code line counts."""
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    raw = len(lines)
    blank = sum(1 for ln in lines if not ln.strip())
    comment = sum(1 for ln in lines if ln.strip().startswith("#"))
    doc = 0
    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
                b = node.body
                if b and isinstance(b[0], ast.Expr) and \
                        isinstance(b[0].value, ast.Constant) and \
                        isinstance(b[0].value.value, str):
                    doc += b[0].end_lineno - b[0].lineno + 1
    except SyntaxError:
        pass
    return {"raw": raw, "blank": blank, "comment": comment, "docstring": doc,
            "code": raw - blank - comment - doc}


def defs(path: Path) -> dict:
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {"func": 0, "method": 0, "cls": 0}
    top = sum(1 for n in tree.body
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    cls = sum(1 for n in tree.body if isinstance(n, ast.ClassDef))
    meth = 0
    for n in tree.body:
        if isinstance(n, ast.ClassDef):
            meth += sum(1 for m in n.body
                        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)))
    return {"func": top, "method": meth, "cls": cls}


def internal_imports(path: Path) -> set[str]:
    src = path.read_text(encoding="utf-8")
    mods = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return mods
    names = {p.stem for p in LIB.glob("*.py")}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            if n.level and n.module in names:            # from .x import
                mods.add(n.module)
            elif n.module and n.module.startswith("lightrack."):
                m = n.module.split(".", 1)[1]
                if m in names:
                    mods.add(m)
            elif n.module == "lightrack":
                mods.add("__init__")
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("lightrack."):
                    m = a.name.split(".", 1)[1]
                    if m in names:
                        mods.add(m)
    return mods - {path.stem}


DESC = {
    "__init__": "package surface: re-exports the public API",
    "config": "loads config.toml into CONFIG",
    "loader": "file discovery, decimal/column sniffing, raw load, timestamps",
    "baseline": "rolling-median baseline removal, quiet noise floor",
    "speed": "geometry-constrained time-of-flight speed and direction",
    "detect": "DIAGNOSTIC chain: passage window, prominence axle peaks",
    "diagnostic": "per-sensor amplitude via transferred axle timing",
    "safety": "SAFETY chain: X-crossing, D-crossing, count_axles, lowpass",
    "health": "Phase 4 health features and flags",
    "occupancy": "slow baseline, stationary-train detection",
}
MARK = {
    "safety": "**SAFETY CHAIN** (`count_axles`, X/D)",
    "detect": "**DIAGNOSTIC CHAIN** (`axle_peaks`)",
    "diagnostic": "diagnostic chain (amplitudes)",
    "config": "**FROZEN CONFIG** loader (`config.toml`)",
}

lib_files = sorted(LIB.glob("*.py"))
scr_files = sorted(SCR.glob("*.py"))
L = pd.DataFrame([{"module": p.stem, **classify(p), **defs(p)} for p in lib_files])
S = pd.DataFrame([{"script": p.name, **classify(p)} for p in scr_files])
edges = {p.stem: internal_imports(p) for p in lib_files}

# layer ordering by longest path to a leaf
def depth(m, seen=None):
    seen = seen or set()
    if m in seen:
        return 0
    ds = [depth(x, seen | {m}) for x in edges.get(m, set()) if x in edges]
    return 1 + max(ds) if ds else 0


layers = {}
for m in edges:
    layers.setdefault(depth(m), []).append(m)

# frozen set: modules reachable from the blind-run entry points
def reach(start: set[str]) -> set[str]:
    seen, stack = set(), list(start)
    while stack:
        m = stack.pop()
        if m in seen or m not in edges:
            continue
        seen.add(m)
        stack.extend(edges[m])
    return seen


seed = set()
for f in BLIND_ENTRY:
    p = SCR / f
    if p.exists():
        seed |= internal_imports(p)
frozen_mods = sorted(reach(seed))
FZ = L[L["module"].isin(frozen_mods)]
cfg = LIB / "config_frozen_poc1.toml"
cfg_lines = len(cfg.read_text(encoding="utf-8").splitlines()) if cfg.exists() else 0

# assumptions
a_txt = (ROOT / "ASSUMPTIONS.md").read_text(encoding="utf-8")
heads = re.findall(r"^## (A-\d+) — (.+)$", a_txt, re.M)
blocks = re.split(r"^## A-\d+ — .+$", a_txt, flags=re.M)[1:]
neg = [(t, ti) for (t, ti), b in zip(heads, blocks)
       if any(k in (ti + b[:1200]) for k in NEG_KEYS)]

csvs = sorted(OUT.glob("*.csv"))
pngs = sorted(OUT.glob("*.png"))
numbered = [p for p in scr_files if re.match(r"^\d\d_", p.name)]


def tot(df, col):
    return int(df[col].sum())


md = []
w = md.append
w("# Code inventory\n")
w(f"Generated by `scripts/{Path(__file__).name}`.\n")

w("\n## 1. Line counts\n")
w("\n### 1.1 Totals\n")
w("| population | files | code | +docstrings | +comments | +blank = raw |")
w("|---|---|---|---|---|---|")
for lab, df in (("library `lightrack/`", L), ("scripts `scripts/`", S),
                ("**all Python**", pd.concat([L, S]))):
    w(f"| {lab} | {len(df)} | **{tot(df,'code')}** | "
      f"{tot(df,'code')+tot(df,'docstring')} | "
      f"{tot(df,'code')+tot(df,'docstring')+tot(df,'comment')} | "
      f"**{tot(df,'raw')}** |")
w("")
w("| metric | value |")
w("|---|---|")
w(f"| lines excluding blank and comments | **{tot(pd.concat([L,S]),'code')+tot(pd.concat([L,S]),'docstring')}** |")
w(f"| lines including blank and comments | **{tot(pd.concat([L,S]),'raw')}** |")
w(f"| of which docstring | {tot(pd.concat([L,S]),'docstring')} |")
w(f"| of which comment | {tot(pd.concat([L,S]),'comment')} |")
w(f"| of which blank | {tot(pd.concat([L,S]),'blank')} |")

w("\n### 1.2 Library, per module\n")
w("| module | code | docstring | comment | blank | raw |")
w("|---|---|---|---|---|---|")
for _, r in L.sort_values("raw", ascending=False).iterrows():
    w(f"| `{r['module']}.py` | {r['code']} | {r['docstring']} | "
      f"{r['comment']} | {r['blank']} | {r['raw']} |")
w(f"| **total** | **{tot(L,'code')}** | {tot(L,'docstring')} | "
  f"{tot(L,'comment')} | {tot(L,'blank')} | **{tot(L,'raw')}** |")

w("\n### 1.3 Scripts\n")
w("| metric | value |")
w("|---|---|")
w(f"| files in `scripts/` | **{len(S)}** |")
w(f"| numbered analysis scripts (`NN_*.py`) | **{len(numbered)}** |")
w(f"| code lines | **{tot(S,'code')}** |")
w(f"| raw lines | **{tot(S,'raw')}** |")
w(f"| docstring lines | {tot(S,'docstring')} |")
w(f"| mean raw lines per script | {tot(S,'raw')//max(len(S),1)} |")
w(f"| largest script | `{S.loc[S['raw'].idxmax(),'script']}` ({S['raw'].max()} raw) |")

w("\n### 1.4 Reusable library vs one-off scripts\n")
w("| | files | code | raw | share of code |")
w("|---|---|---|---|---|")
tc = tot(L, "code") + tot(S, "code")
for lab, df in (("reusable library", L), ("one-off analysis scripts", S)):
    w(f"| {lab} | {len(df)} | {tot(df,'code')} | {tot(df,'raw')} | "
      f"{100*tot(df,'code')/tc:.1f} % |")

w("\n## 2. Structure\n")
w("\n### 2.1 Modules\n")
w("| module | role | imports (internal) | marker |")
w("|---|---|---|---|")
for m in sorted(edges):
    imp = ", ".join(f"`{x}`" for x in sorted(edges[m])) or "—"
    w(f"| `{m}.py` | {DESC.get(m,'')} | {imp} | {MARK.get(m,'')} |")

w("\n### 2.2 Edge list (module → module it imports)\n")
w("```")
for m in sorted(edges):
    for d in sorted(edges[m]):
        w(f"{m} -> {d}")
w("```")

w("\n### 2.3 Layer ordering (leaves at the bottom)\n")
w("| layer | modules |")
w("|---|---|")
for k in sorted(layers, reverse=True):
    w(f"| {k} | {', '.join(f'`{x}`' for x in sorted(layers[k]))} |")

w("\n### 2.4 Chain markers\n")
w("| chain | module | key symbols |")
w("|---|---|---|")
w("| **safety (SIL4 candidate)** | `safety.py` | `count_axles`, `x_crossing`, `d_crossing`, `levels`, `lowpass` |")
w("| **diagnostic** | `detect.py`, `diagnostic.py` | `axle_peaks`, `passage_window`, `has_train`, `channel_amplitudes` |")
w("| **frozen config** | `config.py` → `config.toml` | `CONFIG`; snapshot `config_frozen_poc1.toml` |")

w("\n## 3. Scale markers\n")
w("| metric | value |")
w("|---|---|")
w(f"| library top-level functions | **{tot(L,'func')}** |")
w(f"| library classes | **{tot(L,'cls')}** |")
w(f"| library methods/properties | {tot(L,'method')} |")
w(f"| library functions + classes | **{tot(L,'func')+tot(L,'cls')}** |")
w(f"| numbered analysis scripts | **{len(numbered)}** |")
w(f"| `out/*.csv` tables | **{len(csvs)}** |")
w(f"| `out/*.png` figures | **{len(pngs)}** |")
w(f"| ASSUMPTIONS entries (A-nn) | **{len(heads)}** |")
w(f"| — carrying a negative/null result | **{len(neg)}** ({100*len(neg)/max(len(heads),1):.0f} %) |")
w("")
w(f"Negative/null classification: entry title or first 1200 characters contains "
  f"one of {len(NEG_KEYS)} markers — "
  f"{', '.join(sorted(set(k.lower() for k in NEG_KEYS)))}.\n")
w("| A-nn | title |")
w("|---|---|")
for t, ti in neg:
    w(f"| {t} | {ti} |")

w("\n## 4. Frozen vs exploratory\n")
w("\n### 4.1 The frozen configuration the July blind test ran against\n")
w(f"Entry points: {', '.join('`'+b+'`' for b in BLIND_ENTRY)}. "
  f"Modules below are the transitive closure of their internal imports.\n")
w("| file | code | raw |")
w("|---|---|---|")
for _, r in FZ.sort_values("module").iterrows():
    w(f"| `lightrack/{r['module']}.py` | {r['code']} | {r['raw']} |")
w(f"| `lightrack/config.toml` | — | {len((LIB/'config.toml').read_text(encoding='utf-8').splitlines())} |")
w(f"| `lightrack/config_frozen_poc1.toml` (snapshot) | — | {cfg_lines} |")
w(f"| **total, code only** | **{tot(FZ,'code')}** | |")
w(f"| **total, raw incl. config.toml** | | **{tot(FZ,'raw')+len((LIB/'config.toml').read_text(encoding='utf-8').splitlines())}** |")
w("")
w("| metric | value |")
w("|---|---|")
w(f"| modules in the frozen set | **{len(FZ)}** of {len(L)} |")
w(f"| frozen code lines | **{tot(FZ,'code')}** |")
w(f"| frozen code + docstring lines | **{tot(FZ,'code')+tot(FZ,'docstring')}** |")
w(f"| frozen raw lines (Python only) | **{tot(FZ,'raw')}** |")
w(f"| frozen share of library code | {100*tot(FZ,'code')/max(tot(L,'code'),1):.1f} % |")
w(f"| frozen share of all project code | {100*tot(FZ,'code')/max(tc,1):.1f} % |")
w("")
w("| fact | value |")
w("|---|---|")
w("| `config_frozen_poc1.toml` is a byte-identical snapshot of `config.toml` at blind-run time | yes |")
w("| `detect.py` gained `has_train` / `absolute_gate_pm` **after** the blind run | yes (A-53) |")
w("| library modules outside the frozen set | "
  f"{', '.join('`'+m+'`' for m in sorted(set(L['module'])-set(frozen_mods))) or '—'} |")

w("\n### 4.2 Exploratory\n")
w("| | files | code | raw |")
w("|---|---|---|---|")
w(f"| scripts (all one-off) | {len(S)} | {tot(S,'code')} | {tot(S,'raw')} |")
w(f"| library modules not in the frozen set | {len(L)-len(FZ)} | "
  f"{tot(L,'code')-tot(FZ,'code')} | {tot(L,'raw')-tot(FZ,'raw')} |")

(OUT / "code_inventory.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(f"wrote out/code_inventory.md ({len(md)} lines)")
print(f"library {len(L)} modules / {tot(L,'raw')} raw; "
      f"scripts {len(S)} / {tot(S,'raw')} raw; frozen {len(FZ)} modules "
      f"/ {tot(FZ,'code')} code")
print(f"assumptions {len(heads)}, negative {len(neg)}; "
      f"csv {len(csvs)}, png {len(pngs)}")
