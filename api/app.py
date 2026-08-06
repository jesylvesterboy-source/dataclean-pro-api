"""
DataClean Pro — Web Backend with Python-Style Cleaning
FIXED VERSION - Ensures proper data type conversion
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
from datetime import datetime

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
    print(f"\n📊 Starting Python-Style Cleaning...")
    print(f"   Initial shape: {df.shape}")
    print(f"   Initial columns: {len(df.columns)}")
    print(f"   Initial missing values: {df.isnull().sum().sum()}")
    
    df_clean = df.copy()
    
    # Step 1: Remove leading/trailing spaces from string columns
    print("\n   Step 1: Removing spaces...")
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str).str.strip()
            df_clean[col] = df_clean[col].replace('nan', np.nan)
            df_clean[col] = df_clean[col].replace('None', np.nan)
            df_clean[col] = df_clean[col].replace('', np.nan)
    
    # Step 2: Standardize text case for string columns
    print("   Step 2: Standardizing text case...")
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Check if column seems categorical (few unique values)
            if df_clean[col].nunique() / len(df_clean) < 0.3:
                df_clean[col] = df_clean[col].str.title()
            else:
                df_clean[col] = df_clean[col].str.strip()
    
    # Step 3: Convert numeric columns (matching Python's approach)
    print("   Step 3: Converting numeric columns...")
    numeric_columns = ['Progress', 'Duration (in seconds)', 'Q2', 'Q6', 'Q16', 'Q18']
    converted_numeric = []
    
    for col in numeric_columns:
        if col in df_clean.columns:
            original_type = df_clean[col].dtype
            try:
                # Remove currency symbols and commas
                cleaned = df_clean[col].astype(str).str.replace('$', '')
                cleaned = cleaned.str.replace(',', '')
                cleaned = cleaned.str.replace('%', '')
                cleaned = cleaned.str.strip()
                
                # Convert to numeric, coerce errors to NaN
                numeric = pd.to_numeric(cleaned, errors='coerce')
                if numeric.notna().sum() > len(df_clean) * 0.5:
                    df_clean[col] = numeric
                    converted_numeric.append(col)
                    print(f"      ✓ Converted '{col}': {original_type} → {df_clean[col].dtype}")
            except Exception as e:
                print(f"      ✗ Failed to convert '{col}': {str(e)}")
    
    # Step 4: Convert date columns (matching Python's approach)
    print("   Step 4: Converting date columns...")
    date_columns = ['StartDate', 'EndDate', 'Finished', 'Q20']
    converted_dates = []
    
    for col in date_columns:
        if col in df_clean.columns:
            original_type = df_clean[col].dtype
            try:
                # Try multiple date formats
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                if df_clean[col].notna().sum() > 0:
                    converted_dates.append(col)
                    print(f"      ✓ Converted '{col}': {original_type} → {df_clean[col].dtype}")
            except Exception as e:
                print(f"      ✗ Failed to convert '{col}': {str(e)}")
    
    # Step 5: Handle missing values
    print("   Step 5: Handling missing values...")
    missing_pct = df_clean.isnull().sum() / len(df_clean) * 100
    
    for col in df_clean.columns:
        if missing_pct[col] < 5:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna('Unknown')
            elif df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif missing_pct[col] < 30:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna('Missing')
            elif df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    
    # Step 6: Remove duplicate rows
    print("   Step 6: Removing duplicates...")
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed = initial_rows - len(df_clean)
    if removed > 0:
        print(f"      Removed {removed} duplicate rows")
    
    # Step 7: Handle outliers (cap at 1.5*IQR)
    print("   Step 7: Handling outliers...")
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Final summary
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
    # Validate request
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Please select a CSV or Excel file."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a .csv, .xlsx, or .xls file."}), 500

    # Save upload to temp directory
    tmp_dir = tempfile.mkdtemp(prefix="dcp_")
    safe_name = secure_filename(file.filename)
    input_path = os.path.join(tmp_dir, safe_name)
    output_dir = os.path.join(tmp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    file.save(input_path)
    print(f"\n📁 File uploaded: {safe_name}")

    # Read the file
    try:
        ext = safe_name.rsplit(".", 1)[1].lower()
        if ext in ("xlsx", "xls"):
            df = pd.read_excel(input_path)
        else:
            df = pd.read_csv(input_path, low_memory=False)
        print(f"📊 Original file loaded: {df.shape}")
    except Exception as exc:
        return jsonify({
            "error": f"Failed to read file: {str(exc)}"
        }), 500

    # Track stats before cleaning
    rows_before = len(df)
    missing_before = int(df.isna().sum().sum())

    # Apply Python-style cleaning
    try:
        df_cleaned = clean_data_python_style(df)
    except Exception as exc:
        return jsonify({
            "error": f"Cleaning failed: {str(exc)}",
            "detail": traceback.format_exc()
        }), 500

    # Track stats after cleaning
    rows_after = len(df_cleaned)
    missing_after = int(df_cleaned.isna().sum().sum())
    rows_removed = rows_before - rows_after

    # Save cleaned file
    excel_path = os.path.join(output_dir, f"{Path(safe_name).stem}_cleaned.xlsx")
    try:
        df_cleaned.to_excel(excel_path, index=False)
        print(f"💾 Cleaned file saved: {excel_path}")
    except Exception as exc:
        return jsonify({
            "error": f"Failed to save cleaned file: {str(exc)}"
        }), 500

    if not os.path.exists(excel_path):
        return jsonify({"error": "Cleaned file was not generated."}), 500

    # Stream the cleaned file back
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

    # Pass stats back via headers
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
    print("\n" + "="*70)
    print("  DataClean Pro Web — STANDARDIZED VERSION")
    print("  Using Python-Style Cleaning")
    print("  Powered by Eduxellence Analytics · https://eduxellence.org")
    print("="*70)
    print(f"  Running at: http://localhost:{port}")
    print("="*70 + "\n")
    app.run(debug=True, host="0.0.0.0", port=port)
