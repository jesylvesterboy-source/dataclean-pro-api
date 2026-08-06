"""
DataClean Pro — Web Backend
Flask API for eduxellence.org/upload
Handles CSV/Excel file upload → clean → download

Free tier compatible with Vercel serverless deployment.
STANDARDIZED TO MATCH PYTHON CLEANING OUTPUT
"""

import os
import sys
import uuid
import tempfile
import traceback
import re
from pathlib import Path
from flask import Flask, request, jsonify, send_file, after_this_request
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np

# Add parent directory so we can import dataclean_pro
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataclean_pro import DataCleanPro

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB max upload

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_data_python_style(df):
    """
    Clean data using Python-style approach (matching our test script)
    This standardizes the cleaning to match Python's output
    """
    df_clean = df.copy()
    
    # Step 1: Remove leading/trailing spaces from string columns
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str).str.strip()
            df_clean[col] = df_clean[col].replace('nan', np.nan)
            df_clean[col] = df_clean[col].replace('None', np.nan)
            df_clean[col] = df_clean[col].replace('', np.nan)
    
    # Step 2: Standardize text case for string columns
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Check if column seems categorical (few unique values)
            if df_clean[col].nunique() / len(df_clean) < 0.3:
                df_clean[col] = df_clean[col].str.title()  # Title case for categories
            else:
                df_clean[col] = df_clean[col].str.strip()
    
    # Step 3: Convert numeric columns (matching Python's approach)
    numeric_columns = ['Progress', 'Duration (in seconds)', 'Q2', 'Q6', 'Q16', 'Q18']
    for col in numeric_columns:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            try:
                # Remove currency symbols and commas
                cleaned = df_clean[col].astype(str).str.replace('$', '')
                cleaned = cleaned.str.replace(',', '')
                cleaned = cleaned.str.replace('%', '')
                cleaned = cleaned.str.strip()
                
                # Convert to numeric, coerce errors to NaN
                numeric = pd.to_numeric(cleaned, errors='coerce')
                if numeric.notna().sum() > len(df_clean) * 0.5:  # If more than 50% valid
                    df_clean[col] = numeric
            except:
                pass
    
    # Step 4: Convert date columns (matching Python's approach)
    date_columns = ['StartDate', 'EndDate', 'Finished', 'Q20']
    for col in date_columns:
        if col in df_clean.columns and df_clean[col].dtype == 'object':
            try:
                # Try multiple date formats
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            except:
                pass
    
    # Step 5: Handle missing values (matching Python's approach)
    missing_pct = df_clean.isnull().sum() / len(df_clean) * 100
    
    for col in df_clean.columns:
        if missing_pct[col] < 5:  # If less than 5% missing
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna('Unknown')
            elif df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif missing_pct[col] < 30:  # If 5-30% missing
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna('Missing')
            elif df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    
    # Step 6: Remove duplicate rows
    df_clean = df_clean.drop_duplicates()
    
    # Step 7: Handle outliers (cap at 1.5*IQR)
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_clean


@app.route("/", methods=["GET"])
def index():
    """Serve the main upload page."""
    html_path = Path(__file__).parent.parent / "public" / "index.html"
    return html_path.read_text(encoding="utf-8"), 200, {"Content-Type": "text/html"}


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """
    POST /api/upload
    Accepts: multipart/form-data with field 'file' (CSV or Excel)
    Returns: JSON with file preview, column stats, and metadata
    This is used by the frontend for the initial upload/preview.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Please select a CSV or Excel file."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a .csv, .xlsx, or .xls file."}), 400

    tmp_dir = tempfile.mkdtemp(prefix="dcp_upload_")
    safe_name = secure_filename(file.filename)
    input_path = os.path.join(tmp_dir, safe_name)
    file.save(input_path)

    try:
        ext = safe_name.rsplit(".", 1)[1].lower()
        if ext in ("xlsx", "xls"):
            df = pd.read_excel(input_path)
        else:
            df = pd.read_csv(input_path, low_memory=False)

        # Replace NaN with None for JSON serialisation
        df_preview = df.head(5).where(df.head(5).notna(), other=None)

        col_stats = []
        for col in df.columns:
            missing = int(df[col].isna().sum())
            pct = round(100 * missing / max(len(df), 1), 1)
            col_stats.append({
                "name": col,
                "missing": missing,
                "missing_pct": pct,
                "unique": int(df[col].nunique()),
                "dtype": str(df[col].dtype),
            })

        return jsonify({
            "rows": len(df),
            "cols": len(df.columns),
            "missing_total": int(df.isna().sum().sum()),
            "columns": col_stats,
            "preview": df_preview.to_dict(orient="records"),
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.route("/api/clean", methods=["POST"])
def clean_file():
    """
    POST /api/clean
    Accepts: multipart/form-data with field 'file' (CSV or Excel)
    Returns: cleaned .xlsx file as download attachment
             or JSON error on failure
    STANDARDIZED to match Python cleaning output
    """
    # ── Validate request ──────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Please select a CSV or Excel file."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a .csv, .xlsx, or .xls file."}), 400

    # ── Save upload to temp directory ─────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="dcp_")
    safe_name = secure_filename(file.filename)
    input_path = os.path.join(tmp_dir, safe_name)
    output_dir = os.path.join(tmp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    file.save(input_path)

    # ── Read the file ────────────────────────────────────────────────────
    try:
        ext = safe_name.rsplit(".", 1)[1].lower()
        if ext in ("xlsx", "xls"):
            df = pd.read_excel(input_path)
        else:
            df = pd.read_csv(input_path, low_memory=False)
    except Exception as exc:
        return jsonify({
            "error": f"Failed to read file: {str(exc)}"
        }), 500

    # ── Track stats before cleaning ──────────────────────────────────────
    rows_before = len(df)
    missing_before = int(df.isna().sum().sum())

    # ── Apply Python-style cleaning ──────────────────────────────────────
    try:
        df_cleaned = clean_data_python_style(df)
    except Exception as exc:
        return jsonify({
            "error": f"Cleaning failed: {str(exc)}",
            "detail": traceback.format_exc()
        }), 500

    # ── Track stats after cleaning ──────────────────────────────────────
    rows_after = len(df_cleaned)
    missing_after = int(df_cleaned.isna().sum().sum())
    rows_removed = rows_before - rows_after

    # ── Save cleaned file ──────────────────────────────────────────────────
    excel_path = os.path.join(output_dir, f"{Path(safe_name).stem}_cleaned.xlsx")
    try:
        df_cleaned.to_excel(excel_path, index=False)
    except Exception as exc:
        return jsonify({
            "error": f"Failed to save cleaned file: {str(exc)}"
        }), 500

    if not os.path.exists(excel_path):
        return jsonify({"error": "Cleaned file was not generated. Please check your input."}), 500

    # ── Stream the cleaned file back ──────────────────────────────────────
    stem = Path(safe_name).stem
    download_name = f"{stem}_cleaned_by_eduxellence.xlsx"

    # Attach stats as response headers so the frontend can read them
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

    # Pass stats back via headers (frontend reads these)
    response.headers["X-Rows-In"]         = str(rows_before)
    response.headers["X-Rows-Out"]        = str(rows_after)
    response.headers["X-Rows-Removed"]    = str(rows_removed)
    response.headers["X-Missing-Before"]  = str(missing_before)
    response.headers["X-Missing-After"]   = str(missing_after)
    response.headers["X-Numeric-Cols"]    = str(len(df_cleaned.select_dtypes(include=[np.number]).columns))
    response.headers["X-Date-Cols"]       = str(len(df_cleaned.select_dtypes(include=['datetime64']).columns))
    response.headers["Access-Control-Expose-Headers"] = (
        "X-Rows-In, X-Rows-Out, X-Rows-Removed, X-Missing-Before, X-Missing-After, X-Numeric-Cols, X-Date-Cols"
    )

    return response


@app.route("/api/preview", methods=["POST"])
def preview_file():
    """
    POST /api/preview
    Returns a JSON preview of the first 5 rows + column stats
    before full cleaning — so the user can see what they uploaded.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type."}), 400

    tmp_dir = tempfile.mkdtemp(prefix="dcp_prev_")
    safe_name = secure_filename(file.filename)
    input_path = os.path.join(tmp_dir, safe_name)
    file.save(input_path)

    try:
        ext = safe_name.rsplit(".", 1)[1].lower()
        if ext in ("xlsx", "xls"):
            df = pd.read_excel(input_path)
        else:
            df = pd.read_csv(input_path, low_memory=False)

        # Replace NaN with None for JSON serialisation
        df_preview = df.head(5).where(df.head(5).notna(), other=None)

        col_stats = []
        for col in df.columns:
            missing = int(df[col].isna().sum())
            pct = round(100 * missing / max(len(df), 1), 1)
            col_stats.append({
                "name": col,
                "missing": missing,
                "missing_pct": pct,
                "unique": int(df[col].nunique()),
                "dtype": str(df[col].dtype),
            })

        return jsonify({
            "rows": len(df),
            "cols": len(df.columns),
            "missing_total": int(df.isna().sum().sum()),
            "columns": col_stats,
            "preview": df_preview.to_dict(orient="records"),
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "DataClean Pro", "site": "eduxellence.org"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n  DataClean Pro Web — running at http://localhost:5000")
    print("  Powered by Eduxellence Analytics · https://eduxellence.org")
    print("  STANDARDIZED to match Python cleaning output\n")
    app.run(debug=True, host="0.0.0.0", port=port)
