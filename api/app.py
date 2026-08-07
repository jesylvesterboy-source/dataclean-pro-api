"""
DataClean Pro — Web Backend with Python-Style Cleaning
CORRECTED VERSION - fixes ordering + NA-detection bugs identified in comparison
"""

import os
import re
import tempfile
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_file, after_this_request
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB max upload

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}

# Regex patterns used to decide whether a column LOOKS like it holds dates,
# before attempting pd.to_datetime on it. This mirrors the reference
# Python/R scripts exactly — no hardcoded column names.
DATE_PATTERNS = [
    r'\d{1,2}/\d{1,2}/\d{4}',   # MM/DD/YYYY or DD/MM/YYYY
    r'\d{4}-\d{1,2}-\d{1,2}',   # YYYY-MM-DD
    r'\d{1,2}-\d{1,2}-\d{4}',   # MM-DD-YYYY
]


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _is_text_col(series: pd.Series) -> bool:
    """
    True for columns holding text, regardless of pandas version.

    Pandas < 3.0 stores text as dtype 'object'. Pandas >= 3.0 (PDEP-14)
    defaults text columns to a dedicated 'str' dtype instead. Every
    cleaning step here needs to recognize BOTH, otherwise on pandas 3.x
    every `dtype == 'object'` check silently evaluates to False and the
    entire cleaning pipeline becomes a no-op — which is exactly the kind
    of silent divergence between environments (local script vs. deployed
    backend) that produces mismatched output with no visible error.
    """
    return series.dtype == 'object' or pd.api.types.is_string_dtype(series)


def clean_data_python_style(df):
    """
    Faithful port of the confirmed reference cleaning algorithm (the same
    one implemented in the client's standalone Python script and R script).
    This is column-name-agnostic: it inspects every object column and
    decides numeric/date/categorical status dynamically, exactly like the
    reference scripts do, instead of relying on a hardcoded column list.
    """
    print(f"\n📊 Starting Python-Style Cleaning...")
    print(f"   Initial shape: {df.shape}")
    print(f"   Initial missing values: {df.isnull().sum().sum()}")

    df_clean = df.copy()

    # ------------------------------------------------------------------
    # Step 1: Remove leading/trailing spaces; normalize NA-like tokens
    # ------------------------------------------------------------------
    print("\n   Step 1: Removing spaces / normalizing NA tokens...")
    for col in df_clean.columns:
        if _is_text_col(df_clean[col]):
            df_clean[col] = df_clean[col].astype(str).str.strip()
            df_clean[col] = df_clean[col].replace('nan', np.nan)
            df_clean[col] = df_clean[col].replace('None', np.nan)
            df_clean[col] = df_clean[col].replace('', np.nan)

    # ------------------------------------------------------------------
    # Step 2: Standardize text case for categorical-looking columns
    # (few unique values relative to row count -> Title Case)
    # ------------------------------------------------------------------
    print("   Step 2: Standardizing text case...")
    for col in df_clean.columns:
        if _is_text_col(df_clean[col]):
            if len(df_clean) > 0 and df_clean[col].nunique() / len(df_clean) < 0.3:
                df_clean[col] = df_clean[col].str.title()
            else:
                df_clean[col] = df_clean[col].str.strip()

    # ------------------------------------------------------------------
    # Step 3: Convert numeric columns — tries EVERY object column,
    # keeps the conversion only if >50% of values parse as numeric.
    # (No hardcoded column list — matches the reference script.)
    # ------------------------------------------------------------------
    print("   Step 3: Converting numeric columns...")
    converted_numeric = []
    for col in df_clean.columns:
        if _is_text_col(df_clean[col]):
            try:
                cleaned = df_clean[col].astype(str).str.replace('$', '', regex=False)
                cleaned = cleaned.str.replace(',', '', regex=False)
                cleaned = cleaned.str.replace('%', '', regex=False)
                cleaned = cleaned.str.strip()

                numeric = pd.to_numeric(cleaned, errors='coerce')
                if numeric.notna().sum() > len(df_clean) * 0.5:
                    df_clean[col] = numeric
                    converted_numeric.append(col)
                    print(f"      ✓ Converted '{col}' to numeric")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Step 4: Detect and convert date columns — gated by a regex check
    # on a sample of the column's values, exactly like the reference.
    # ------------------------------------------------------------------
    print("   Step 4: Detecting and converting date columns...")
    converted_dates = []
    for col in df_clean.columns:
        if _is_text_col(df_clean[col]):
            try:
                sample = df_clean[col].dropna().astype(str).head(10)
                if len(sample) > 0 and any(re.search(p, ' '.join(sample)) for p in DATE_PATTERNS):
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                    converted_dates.append(col)
                    print(f"      ✓ Converted '{col}' to datetime")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Step 5: Handle missing values (threshold-based, matches reference)
    # ------------------------------------------------------------------
    print("   Step 5: Handling missing values...")
    missing_pct = df_clean.isnull().sum() / len(df_clean) * 100
    for col in df_clean.columns:
        if missing_pct[col] < 5:
            if _is_text_col(df_clean[col]):
                df_clean[col] = df_clean[col].fillna('Unknown')
            elif df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif missing_pct[col] < 30:
            if _is_text_col(df_clean[col]):
                df_clean[col] = df_clean[col].fillna('Missing')
            elif df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

    # ------------------------------------------------------------------
    # Step 6: Remove duplicate rows
    # ------------------------------------------------------------------
    print("   Step 6: Removing duplicates...")
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed = initial_rows - len(df_clean)
    if removed > 0:
        print(f"      Removed {removed} duplicate rows")

    # ------------------------------------------------------------------
    # Step 7: Handle outliers — cap numeric columns at 1.5*IQR
    # ------------------------------------------------------------------
    print("   Step 7: Handling outliers...")
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

    print(f"\n✅ Cleaning Complete!")
    print(f"   Final shape: {df_clean.shape}")
    print(f"   Missing values: {df_clean.isnull().sum().sum()}")
    print(f"   Numeric columns: {len(df_clean.select_dtypes(include=[np.number]).columns)}")
    print(f"   Date columns: {len(df_clean.select_dtypes(include=['datetime64']).columns)}")
    print(f"   Converted numeric: {', '.join(converted_numeric) if converted_numeric else 'None'}")
    print(f"   Converted dates: {', '.join(converted_dates) if converted_dates else 'None'}")

    return df_clean


@app.route("/api/clean", methods=["POST"])
def clean_file():
    """
    POST /api/clean
    Accepts: multipart/form-data with field 'file' (CSV or Excel)
    Returns: cleaned .xlsx file as download attachment
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Please select a CSV or Excel file."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a .csv, .xlsx, or .xls file."}), 400

    tmp_dir = tempfile.mkdtemp(prefix="dcp_")
    safe_name = secure_filename(file.filename)
    input_path = os.path.join(tmp_dir, safe_name)
    output_dir = os.path.join(tmp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    file.save(input_path)
    print(f"\n📁 File uploaded: {safe_name}")

    try:
        ext = safe_name.rsplit(".", 1)[1].lower()
        if ext in ("xlsx", "xls"):
            df = pd.read_excel(input_path)
        else:
            df = pd.read_csv(input_path, low_memory=False)
        print(f"📊 Original file loaded: {df.shape}")
    except Exception as exc:
        return jsonify({"error": f"Failed to read file: {str(exc)}"}), 500

    rows_before = len(df)
    missing_before = int(df.isna().sum().sum())

    try:
        df_cleaned = clean_data_python_style(df)
    except Exception as exc:
        return jsonify({
            "error": f"Cleaning failed: {str(exc)}",
            "detail": traceback.format_exc()
        }), 500

    rows_after = len(df_cleaned)
    missing_after = int(df_cleaned.isna().sum().sum())
    rows_removed = rows_before - rows_after

    excel_path = os.path.join(output_dir, f"{Path(safe_name).stem}_cleaned.xlsx")
    try:
        df_cleaned.to_excel(excel_path, index=False)
        print(f"💾 Cleaned file saved: {excel_path}")
    except Exception as exc:
        return jsonify({"error": f"Failed to save cleaned file: {str(exc)}"}), 500

    if not os.path.exists(excel_path):
        return jsonify({"error": "Cleaned file was not generated."}), 500

    stem = Path(safe_name).stem
    download_name = f"{stem}_cleaned_by_eduxellence.xlsx"

    @after_this_request
    def cleanup(response):
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass
        return response

    response = send_file(
        excel_path,
        as_attachment=True,
        download_name=download_name,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    response.headers["X-Rows-In"] = str(rows_before)
    response.headers["X-Rows-Out"] = str(rows_after)
    response.headers["X-Rows-Removed"] = str(rows_removed)
    response.headers["X-Missing-Before"] = str(missing_before)
    response.headers["X-Missing-After"] = str(missing_after)
    response.headers["X-Numeric-Cols"] = str(len(df_cleaned.select_dtypes(include=[np.number]).columns))
    response.headers["X-Date-Cols"] = str(len(df_cleaned.select_dtypes(include=['datetime64']).columns))
    response.headers["Access-Control-Expose-Headers"] = (
        "X-Rows-In, X-Rows-Out, X-Rows-Removed, X-Missing-Before, X-Missing-After, X-Numeric-Cols, X-Date-Cols"
    )

    return response


# Keep all other routes (/, /api/upload, /api/preview, /api/health) as they were

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 70)
    print("  DataClean Pro Web — CORRECTED VERSION")
    print("  Powered by Eduxellence Analytics · https://eduxellence.org")
    print("=" * 70)
    print(f"  Running at: http://localhost:{port}")
    print("=" * 70 + "\n")
    app.run(debug=True, host="0.0.0.0", port=port)
