#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║            DataClean Pro — by Eduxellence Analytics              ║
║              https://eduxellence.org | Free Tool v1.0            ║
╚══════════════════════════════════════════════════════════════════╝

Automates the entire CSV data cleaning & validation pipeline.
Handles: Duplicates · Missing Values · Outliers · Type Inference ·
         Date Standardisation · Currency Parsing · HTML Report Output

Usage:
    python dataclean_pro.py                        # interactive mode
    python dataclean_pro.py mydata.csv             # direct file
    python dataclean_pro.py mydata.csv --silent    # no prompts
    python dataclean_pro.py --batch data/          # clean entire folder
"""

import os
import sys
import csv
import re
import json
import time
import shutil
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
from tabulate import tabulate

warnings.filterwarnings("ignore")

# ─── Colour helpers (no tqdm dependency) ────────────────────────────────────
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    C = {
        "header": Fore.CYAN + Style.BRIGHT,
        "ok":     Fore.GREEN + Style.BRIGHT,
        "warn":   Fore.YELLOW,
        "err":    Fore.RED + Style.BRIGHT,
        "dim":    Style.DIM,
        "reset":  Style.RESET_ALL,
        "blue":   Fore.BLUE + Style.BRIGHT,
        "magenta":Fore.MAGENTA + Style.BRIGHT,
    }
except ImportError:
    C = {k: "" for k in ("header","ok","warn","err","dim","reset","blue","magenta")}


# ─── Branding ────────────────────────────────────────────────────────────────
BANNER = f"""
{C['header']}╔══════════════════════════════════════════════════════════════════╗
║            DataClean Pro — by Eduxellence Analytics              ║
║              https://eduxellence.org | Free Tool v1.0            ║
╚══════════════════════════════════════════════════════════════════╝{C['reset']}
"""

BRAND = "Eduxellence Analytics · https://eduxellence.org"


# ─── Progress bar (pure stdlib) ──────────────────────────────────────────────
def progress_bar(label: str, total: int, current: int, width: int = 30) -> None:
    filled = int(width * current / max(total, 1))
    bar = "█" * filled + "░" * (width - filled)
    pct = int(100 * current / max(total, 1))
    print(f"\r  {C['dim']}{label}{C['reset']} [{C['ok']}{bar}{C['reset']}] {pct}%", end="", flush=True)
    if current >= total:
        print()


def step_banner(text: str) -> None:
    print(f"\n{C['blue']}▶  {text}{C['reset']}")


def ok(text: str) -> None:
    print(f"  {C['ok']}✓{C['reset']}  {text}")


def warn(text: str) -> None:
    print(f"  {C['warn']}⚠{C['reset']}  {text}")


def err(text: str) -> None:
    print(f"  {C['err']}✗{C['reset']}  {text}")


# ─── Currency / Numeric Parser ────────────────────────────────────────────────
_CURRENCY_RE = re.compile(r"[\$€£¥₦,\s]")

def parse_numeric(val) -> float | None:
    """Strip currency symbols, commas, and whitespace; return float or None."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s in ("", "N/A", "NA", "n/a", "na", "None", "null", "NULL", "-", "#N/A"):
        return None
    cleaned = _CURRENCY_RE.sub("", s)
    try:
        return float(cleaned)
    except ValueError:
        return None


# ─── Date Parser ─────────────────────────────────────────────────────────────
_DATE_FMTS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
    "%d-%m-%Y", "%Y/%m/%d",
    "%b %d %Y", "%B %d %Y",
    "%d %b %Y", "%d %B %Y",
    "%Y%m%d",
]

def parse_date(val) -> pd.Timestamp | None:
    if pd.isna(val):
        return None
    s = str(val).strip()
    for fmt in _DATE_FMTS:
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            continue
    try:
        return pd.Timestamp(val)
    except Exception:
        return None


# ─── Smart Type Detector ──────────────────────────────────────────────────────
def detect_column_types(df: pd.DataFrame) -> dict:
    """
    Returns a dict: col_name -> inferred type
    Types: numeric | date | categorical | text | identifier | boolean
    """
    types = {}
    for col in df.columns:
        series = df[col].dropna().astype(str).str.strip()
        if series.empty:
            types[col] = "empty"
            continue

        # Boolean check
        bool_vals = {"true","false","yes","no","1","0","y","n"}
        if series.str.lower().isin(bool_vals).mean() > 0.85:
            types[col] = "boolean"
            continue

        # Numeric check (allow currency symbols)
        numeric_hits = series.apply(lambda x: parse_numeric(x) is not None).mean()
        if numeric_hits > 0.70:
            types[col] = "numeric"
            continue

        # Date check
        sample = series.sample(min(30, len(series)), random_state=42)
        date_hits = sample.apply(lambda x: parse_date(x) is not None).mean()
        if date_hits > 0.60:
            types[col] = "date"
            continue

        # Identifier / email / phone check
        col_lower = col.lower()
        id_keywords = {"id","code","ref","number","no","num","email","phone","mobile","tel"}
        if any(kw in col_lower for kw in id_keywords):
            types[col] = "identifier"
            continue

        # Categorical vs text (low cardinality = categorical)
        nunique = series.nunique()
        if nunique <= max(10, len(series) * 0.15):
            types[col] = "categorical"
        else:
            types[col] = "text"

    return types


# ─── Outlier Detector (IQR method) ──────────────────────────────────────────
def detect_outliers(series: pd.Series, factor: float = 3.0) -> pd.Series:
    """Return boolean mask: True = outlier. Uses IQR with generous factor."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=series.index)
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (series < lower) | (series > upper)


# ─── Core Cleaner ────────────────────────────────────────────────────────────
class DataCleanPro:
    """
    Full data cleaning pipeline for messy CSV files.
    Produces a cleaned Excel file + a standalone HTML audit report.
    """

    def __init__(self, filepath: str, output_dir: str = "output",
                 outlier_factor: float = 3.0, silent: bool = False):
        self.filepath      = Path(filepath)
        self.output_dir    = Path(output_dir)
        self.outlier_factor = outlier_factor
        self.silent        = silent
        self.log           = []          # list of dicts for the HTML report
        self.stats         = {}          # summary stats

    # ── Public entry point ──────────────────────────────────────────────────
    def run(self) -> dict:
        """Execute the full cleaning pipeline. Returns a summary dict."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = self.filepath.stem
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not self.silent:
            step_banner(f"Loading  →  {self.filepath.name}")

        df_raw = self._load()
        df     = df_raw.copy()
        n_rows_original = len(df)
        n_cols_original = len(df.columns)

        # ── Run cleaning stages ────────────────────────────────────────────
        stages = [
            ("Stripping whitespace",           self._strip_whitespace),
            ("Standardising null markers",     self._standardise_nulls),
            ("Removing blank rows/cols",       self._remove_blank),
            ("Detecting column types",         self._detect_and_cast),
            ("Standardising category case",    self._standardise_categories),
            ("Parsing & standardising dates",  self._parse_dates),
            ("Parsing currency / numerics",    self._parse_currencies),
            ("Removing duplicates",            self._remove_duplicates),
            ("Flagging outliers",              self._flag_outliers),
            ("Imputing missing values",        self._impute_missing),
            ("Generating column summary",      self._column_summary),
        ]

        total = len(stages)
        for i, (label, fn) in enumerate(stages, 1):
            if not self.silent:
                progress_bar("Cleaning", total, i - 1)
                time.sleep(0.05)
            try:
                df = fn(df)
            except Exception as exc:
                self._log("error", label, str(exc))
            if not self.silent:
                progress_bar("Cleaning", total, i)

        # ── Build summary stats ────────────────────────────────────────────
        self.stats = {
            "file":            self.filepath.name,
            "rows_in":         n_rows_original,
            "cols_in":         n_cols_original,
            "rows_out":        len(df),
            "cols_out":        len(df.columns),
            "rows_removed":    n_rows_original - len(df),
            "missing_before":  int(df_raw.isna().sum().sum()),
            "missing_after":   int(df.isna().sum().sum()),
            "processed_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # ── Save outputs ───────────────────────────────────────────────────
        excel_path  = self.output_dir / f"{stem}_cleaned_{ts}.xlsx"
        report_path = self.output_dir / f"{stem}_report_{ts}.html"

        self._save_excel(df, excel_path)
        self._save_report(df, report_path)

        if not self.silent:
            print()
            self._print_summary(excel_path, report_path)

        return {
            "cleaned_df":  df,
            "excel":       str(excel_path),
            "report":      str(report_path),
            "stats":       self.stats,
            "log":         self.log,
        }

    # ── Stage implementations ───────────────────────────────────────────────
    def _load(self) -> pd.DataFrame:
        encodings = ["utf-8", "latin-1", "cp1252", "utf-8-sig"]
        for enc in encodings:
            try:
                df = pd.read_csv(self.filepath, encoding=enc, low_memory=False)
                self._log("ok", "File loaded",
                          f"{len(df)} rows · {len(df.columns)} columns · encoding={enc}")
                return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        raise ValueError(f"Cannot decode {self.filepath}. Try saving as UTF-8.")

    def _strip_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("nan", np.nan)
        self._log("ok", "Whitespace stripped", "Column names + string cells cleaned")
        return df

    def _standardise_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        null_markers = ["N/A", "NA", "n/a", "na", "None", "none", "null",
                        "NULL", "NaN", "-", "#N/A", "#VALUE!", "?", ".",
                        "", " ", "  "]
        before = int(df.isna().sum().sum())
        df = df.replace(null_markers, np.nan)
        after = int(df.isna().sum().sum())
        self._log("ok", "Null markers standardised",
                  f"{after - before} additional nulls identified")
        return df

    def _remove_blank(self, df: pd.DataFrame) -> pd.DataFrame:
        # Rows that are entirely NaN
        blank_rows = df.isna().all(axis=1).sum()
        df = df.dropna(how="all")
        # Columns that are entirely NaN
        blank_cols = df.isna().all(axis=0).sum()
        df = df.dropna(axis=1, how="all")
        self._log("ok", "Blank rows / columns removed",
                  f"{blank_rows} empty rows · {blank_cols} empty columns dropped")
        return df

    def _detect_and_cast(self, df: pd.DataFrame) -> pd.DataFrame:
        self.col_types = detect_column_types(df)
        summary = ", ".join(f"{k}:{v}" for k, v in
                            pd.Series(self.col_types).value_counts().items())
        self._log("ok", "Column types detected", summary)
        return df

    def _standardise_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        cat_cols = [c for c, t in self.col_types.items()
                    if t == "categorical" and c in df.columns]
        changed = 0
        for col in cat_cols:
            original = df[col].copy()
            df[col] = df[col].astype(str).str.strip().str.title()
            df[col] = df[col].replace("Nan", np.nan)
            if not original.equals(df[col]):
                changed += 1
        self._log("ok", "Categories standardised",
                  f"{changed} columns → Title Case applied")
        return df

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        date_cols = [c for c, t in self.col_types.items()
                     if t == "date" and c in df.columns]
        fixed = 0
        for col in date_cols:
            df[col] = df[col].apply(parse_date)
            # Keep as YYYY-MM-DD string for Excel compatibility
            df[col] = df[col].apply(
                lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else np.nan
            )
            fixed += 1
        self._log("ok", "Dates standardised",
                  f"{fixed} date columns → YYYY-MM-DD format")
        return df

    def _parse_currencies(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = [c for c, t in self.col_types.items()
                    if t == "numeric" and c in df.columns]
        converted = 0
        for col in num_cols:
            new_vals = df[col].apply(parse_numeric)
            if new_vals.notna().sum() > 0:
                df[col] = new_vals
                converted += 1
        self._log("ok", "Currency / numerics parsed",
                  f"{converted} columns stripped of symbols → float")
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        removed = before - after
        if removed:
            self._log("warn", "Duplicate rows removed", f"{removed} exact duplicates dropped")
        else:
            self._log("ok", "Duplicate check", "No duplicate rows found")
        return df

    def _flag_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = [c for c, t in self.col_types.items()
                    if t == "numeric" and c in df.columns]
        outlier_report = []
        for col in num_cols:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) < 4:
                continue
            mask = detect_outliers(series, self.outlier_factor)
            n_out = int(mask.sum())
            if n_out > 0:
                outlier_report.append(f"{col}: {n_out} outlier(s)")
                # Flag in new column rather than deleting
                flag_col = f"_outlier_{col}"
                df[flag_col] = False
                df.loc[mask.index[mask], flag_col] = True

        if outlier_report:
            self._log("warn", "Outliers flagged",
                      "; ".join(outlier_report) + " — flagged in _outlier_* columns")
        else:
            self._log("ok", "Outlier check", "No statistical outliers detected")
        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        report = []
        for col in df.columns:
            if col.startswith("_outlier_"):
                continue
            n_missing = int(df[col].isna().sum())
            if n_missing == 0:
                continue
            col_type = self.col_types.get(col, "text")
            if col_type == "numeric":
                median_val = pd.to_numeric(df[col], errors="coerce").median()
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median_val)
                report.append(f"{col}(median={median_val:.2f})")
            elif col_type == "categorical":
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val.iloc[0])
                    report.append(f"{col}(mode='{mode_val.iloc[0]}')")
            # text / date / identifier: leave as NaN (too risky to impute)

        if report:
            self._log("ok", "Missing values imputed", "; ".join(report))
        else:
            self._log("ok", "Missing value check",
                      "No imputation needed (or columns are non-numeric)")
        return df

    def _column_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        self.column_stats = []
        for col in df.columns:
            if col.startswith("_outlier_"):
                continue
            col_data = df[col]
            inferred = self.col_types.get(col, "?")
            missing_n = int(col_data.isna().sum())
            missing_pct = round(100 * missing_n / max(len(df), 1), 1)
            unique_n = int(col_data.nunique())
            if inferred == "numeric":
                numeric_s = pd.to_numeric(col_data, errors="coerce")
                sample = f"min={numeric_s.min():.2f} · max={numeric_s.max():.2f} · mean={numeric_s.mean():.2f}"
            elif inferred == "categorical":
                top = col_data.value_counts().head(3).index.tolist()
                sample = "Top: " + ", ".join(str(v) for v in top)
            else:
                sample = str(col_data.dropna().iloc[0]) if col_data.dropna().any() else "—"
            self.column_stats.append({
                "Column": col,
                "Type": inferred,
                "Missing": f"{missing_n} ({missing_pct}%)",
                "Unique": unique_n,
                "Sample / Stats": sample[:80],
            })
        self._log("ok", "Column summary generated", f"{len(self.column_stats)} columns analysed")
        return df

    # ── Output helpers ──────────────────────────────────────────────────────
    def _save_excel(self, df: pd.DataFrame, path: Path) -> None:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            # Sheet 1: Cleaned Data
            df.to_excel(writer, sheet_name="Cleaned Data", index=False)

            # Sheet 2: Cleaning Log
            log_df = pd.DataFrame(self.log)
            log_df.to_excel(writer, sheet_name="Cleaning Log", index=False)

            # Sheet 3: Column Summary
            if hasattr(self, "column_stats"):
                pd.DataFrame(self.column_stats).to_excel(
                    writer, sheet_name="Column Summary", index=False)

            # Sheet 4: Stats
            stats_df = pd.DataFrame([self.stats]).T.reset_index()
            stats_df.columns = ["Metric", "Value"]
            stats_df.to_excel(writer, sheet_name="Run Stats", index=False)

        self._log("ok", "Excel saved", str(path))

    def _save_report(self, df: pd.DataFrame, path: Path) -> None:
        html = self._build_html_report(df)
        path.write_text(html, encoding="utf-8")
        self._log("ok", "HTML report saved", str(path))

    def _print_summary(self, excel: Path, report: Path) -> None:
        s = self.stats
        print(f"\n{C['header']}{'═'*62}")
        print(f"  CLEANING COMPLETE — {self.filepath.name}")
        print(f"{'═'*62}{C['reset']}")
        table = [
            ["Rows processed",    s['rows_in']],
            ["Rows output",       s['rows_out']],
            ["Rows removed",      s['rows_removed']],
            ["Missing (before)",  s['missing_before']],
            ["Missing (after)",   s['missing_after']],
            ["Missing fixed",     s['missing_before'] - s['missing_after']],
        ]
        print(tabulate(table, headers=["Metric", "Value"],
                       tablefmt="rounded_outline"))
        print(f"\n  {C['ok']}📊 Excel  →{C['reset']} {excel}")
        print(f"  {C['ok']}📋 Report →{C['reset']} {report}")
        print(f"\n  {C['dim']}{BRAND}{C['reset']}\n")

    # ── HTML Report ─────────────────────────────────────────────────────────
    def _build_html_report(self, df: pd.DataFrame) -> str:
        s = self.stats
        now = s['processed_at']

        # Log rows
        log_rows = ""
        for entry in self.log:
            icon = {"ok":"✓","warn":"⚠","error":"✗"}.get(entry["level"], "·")
            colour = {"ok":"#22c55e","warn":"#f59e0b","error":"#ef4444"}.get(entry["level"],"#888")
            log_rows += (
                f'<tr><td style="color:{colour};font-weight:600">{icon} {entry["stage"]}</td>'
                f'<td>{entry["detail"]}</td></tr>'
            )

        # Column summary rows
        col_rows = ""
        if hasattr(self, "column_stats"):
            for row in self.column_stats:
                col_rows += (
                    f'<tr><td><code>{row["Column"]}</code></td>'
                    f'<td><span class="badge">{row["Type"]}</span></td>'
                    f'<td>{row["Missing"]}</td>'
                    f'<td>{row["Unique"]}</td>'
                    f'<td style="color:#64748b;font-size:13px">{row["Sample / Stats"]}</td></tr>'
                )

        # Data preview (first 10 rows)
        preview_df = df[[c for c in df.columns if not c.startswith("_outlier_")]].head(10)
        th = "".join(f"<th>{c}</th>" for c in preview_df.columns)
        td_rows = ""
        for _, row in preview_df.iterrows():
            tds = "".join(f'<td>{str(v) if pd.notna(v) else "<em style=\'color:#cbd5e1\'>null</em>"}</td>'
                          for v in row)
            td_rows += f"<tr>{tds}</tr>"

        efficiency = round(100 * (1 - s['missing_after'] / max(s['missing_before'], 1)), 1)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DataClean Pro Report — {s['file']}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:#f8fafc;color:#1e293b;line-height:1.6}}
  a{{color:#2563eb;text-decoration:none}}
  a:hover{{text-decoration:underline}}

  /* Header */
  .header{{background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);color:#fff;padding:2.5rem 3rem}}
  .header h1{{font-size:2rem;font-weight:800;letter-spacing:-0.5px}}
  .header .sub{{color:#94a3b8;margin-top:4px;font-size:0.95rem}}
  .brand-link{{color:#60a5fa;font-weight:600}}
  .brand-link:hover{{color:#93c5fd}}

  /* Stats cards */
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1rem;padding:2rem 3rem 0}}
  .card{{background:#fff;border-radius:12px;padding:1.25rem 1.5rem;border:1px solid #e2e8f0;box-shadow:0 1px 3px rgba(0,0,0,.05)}}
  .card .num{{font-size:2rem;font-weight:800;color:#0f172a}}
  .card .lbl{{font-size:0.8rem;color:#64748b;margin-top:2px;text-transform:uppercase;letter-spacing:.05em}}
  .card.green .num{{color:#16a34a}}
  .card.amber .num{{color:#d97706}}
  .card.red .num{{color:#dc2626}}

  /* Sections */
  .section{{margin:2rem 3rem}}
  .section-title{{font-size:1.1rem;font-weight:700;color:#0f172a;margin-bottom:1rem;
    padding-bottom:.5rem;border-bottom:2px solid #e2e8f0;display:flex;align-items:center;gap:.5rem}}

  /* Tables */
  table{{width:100%;border-collapse:collapse;font-size:0.875rem;background:#fff;
    border-radius:10px;overflow:hidden;border:1px solid #e2e8f0;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
  th{{background:#f1f5f9;color:#475569;font-weight:600;padding:.75rem 1rem;text-align:left;
    font-size:0.78rem;text-transform:uppercase;letter-spacing:.05em}}
  td{{padding:.65rem 1rem;border-top:1px solid #f1f5f9;vertical-align:top}}
  tr:hover td{{background:#f8fafc}}
  .scroll-x{{overflow-x:auto;border-radius:10px}}

  /* Badge */
  .badge{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:0.75rem;
    font-weight:600;background:#e0f2fe;color:#0369a1}}

  /* Progress bar */
  .pbar-wrap{{background:#f1f5f9;border-radius:999px;height:10px;overflow:hidden;margin-top:8px}}
  .pbar{{height:100%;border-radius:999px;background:linear-gradient(90deg,#22c55e,#16a34a)}}

  /* Footer */
  .footer{{background:#0f172a;color:#64748b;padding:2rem 3rem;margin-top:3rem;font-size:0.875rem}}
  .footer a{{color:#60a5fa}}

  code{{background:#f1f5f9;padding:1px 6px;border-radius:4px;font-family:'Courier New',monospace;font-size:0.85em}}
</style>
</head>
<body>

<div class="header">
  <h1>📊 DataClean Pro — Audit Report</h1>
  <div class="sub">
    File: <strong>{s['file']}</strong> &nbsp;·&nbsp; Processed: {now}<br>
    Powered by <a class="brand-link" href="https://eduxellence.org" target="_blank">Eduxellence Analytics</a>
    — Elite Data Solutions for Research &amp; Enterprise
  </div>
</div>

<div class="cards">
  <div class="card">
    <div class="num">{s['rows_in']}</div>
    <div class="lbl">Rows in</div>
  </div>
  <div class="card green">
    <div class="num">{s['rows_out']}</div>
    <div class="lbl">Rows out</div>
  </div>
  <div class="card {'amber' if s['rows_removed'] > 0 else 'green'}">
    <div class="num">{s['rows_removed']}</div>
    <div class="lbl">Rows removed</div>
  </div>
  <div class="card red">
    <div class="num">{s['missing_before']}</div>
    <div class="lbl">Missing (before)</div>
  </div>
  <div class="card green">
    <div class="num">{s['missing_after']}</div>
    <div class="lbl">Missing (after)</div>
  </div>
  <div class="card green">
    <div class="num">{efficiency}%</div>
    <div class="lbl">Missing fixed</div>
    <div class="pbar-wrap"><div class="pbar" style="width:{min(efficiency,100)}%"></div></div>
  </div>
</div>

<div class="section">
  <div class="section-title">🔧 Cleaning Log</div>
  <table>
    <thead><tr><th>Stage</th><th>Detail</th></tr></thead>
    <tbody>{log_rows}</tbody>
  </table>
</div>

<div class="section">
  <div class="section-title">📋 Column Intelligence Report</div>
  <div class="scroll-x">
  <table>
    <thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th><th>Sample / Stats</th></tr></thead>
    <tbody>{col_rows}</tbody>
  </table>
  </div>
</div>

<div class="section">
  <div class="section-title">👁 Cleaned Data Preview (first 10 rows)</div>
  <div class="scroll-x">
  <table>
    <thead><tr>{th}</tr></thead>
    <tbody>{td_rows}</tbody>
  </table>
  </div>
</div>

<div class="footer">
  <p>Generated by <strong>DataClean Pro v1.0</strong> — a free tool by
     <a href="https://eduxellence.org" target="_blank">Eduxellence Analytics</a></p>
  <p style="margin-top:.5rem">
    Need advanced analytics, predictive modelling, or a custom data pipeline?
    <a href="https://eduxellence.org/#contact" target="_blank">Book a free 15-minute Data Strategy Audit</a>
  </p>
</div>

</body>
</html>"""

    # ── Internal log ────────────────────────────────────────────────────────
    def _log(self, level: str, stage: str, detail: str) -> None:
        self.log.append({"level": level, "stage": stage, "detail": detail})


# ─── Batch runner ────────────────────────────────────────────────────────────
def run_batch(folder: str, output_dir: str, silent: bool) -> None:
    folder = Path(folder)
    csvs = list(folder.rglob("*.csv"))
    if not csvs:
        err(f"No CSV files found in {folder}")
        return
    print(f"\n{C['header']}  Batch mode: {len(csvs)} file(s) found{C['reset']}\n")
    results = []
    for i, csv_path in enumerate(csvs, 1):
        print(f"{C['blue']}  [{i}/{len(csvs)}] {csv_path.name}{C['reset']}")
        cleaner = DataCleanPro(csv_path, output_dir=output_dir, silent=silent)
        r = cleaner.run()
        results.append({"file": csv_path.name, **r["stats"]})

    print(f"\n{C['header']}  BATCH SUMMARY{C['reset']}")
    table = [[r["file"], r["rows_in"], r["rows_out"],
              r["missing_before"], r["missing_after"]] for r in results]
    print(tabulate(table,
                   headers=["File","Rows In","Rows Out","Missing Before","Missing After"],
                   tablefmt="rounded_outline"))
    print(f"\n  {C['dim']}{BRAND}{C['reset']}\n")


# ─── Interactive file picker ──────────────────────────────────────────────────
def interactive_mode(output_dir: str) -> None:
    print(BANNER)
    print(f"  {C['dim']}No file specified. Starting interactive mode.{C['reset']}\n")

    # List CSV files in current directory and subdirs (max depth 2)
    csvs = sorted(Path(".").glob("**/*.csv"))[:20]
    if csvs:
        print(f"  {C['blue']}CSV files found:{C['reset']}")
        for i, p in enumerate(csvs, 1):
            print(f"    [{i}] {p}")
        print()

    filepath = input(f"  {C['ok']}Enter CSV path (or number from list above): {C['reset']}").strip()
    if filepath.isdigit():
        idx = int(filepath) - 1
        if 0 <= idx < len(csvs):
            filepath = str(csvs[idx])
        else:
            err("Invalid selection")
            sys.exit(1)

    if not Path(filepath).exists():
        err(f"File not found: {filepath}")
        sys.exit(1)

    cleaner = DataCleanPro(filepath, output_dir=output_dir)
    cleaner.run()


# ─── CLI entry point ─────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dataclean_pro",
        description="DataClean Pro — automated CSV cleaning by Eduxellence Analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"  {BRAND}\n  Free tool · MIT Licence",
    )
    parser.add_argument("file",          nargs="?",  help="Path to CSV file")
    parser.add_argument("--output", "-o", default="output", help="Output directory (default: output)")
    parser.add_argument("--batch",  "-b", metavar="FOLDER",  help="Clean all CSVs in a folder")
    parser.add_argument("--silent", "-s", action="store_true", help="Suppress progress output")
    parser.add_argument("--outlier-factor", type=float, default=3.0,
                        help="IQR multiplier for outlier detection (default: 3.0)")

    args = parser.parse_args()

    if not args.silent:
        print(BANNER)

    if args.batch:
        run_batch(args.batch, args.output, args.silent)
    elif args.file:
        if not Path(args.file).exists():
            err(f"File not found: {args.file}")
            sys.exit(1)
        cleaner = DataCleanPro(args.file, output_dir=args.output,
                               outlier_factor=args.outlier_factor,
                               silent=args.silent)
        cleaner.run()
    else:
        interactive_mode(args.output)


if __name__ == "__main__":
    main()
