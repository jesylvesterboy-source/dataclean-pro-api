"""
DataClean Pro — Web Backend
Flask API for eduxellence.org/upload
Handles CSV/Excel file upload → clean → download

Free tier compatible with Vercel serverless deployment.
"""

import os
import sys
import uuid
import tempfile
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_file, after_this_request
from werkzeug.utils import secure_filename

# Add parent directory so we can import dataclean_pro
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataclean_pro import DataCleanPro

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB max upload

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    """Serve the main upload page."""
    html_path = Path(__file__).parent.parent / "public" / "index.html"
    return html_path.read_text(encoding="utf-8"), 200, {"Content-Type": "text/html"}


# ─── ADDED: /api/upload endpoint ──────────────────────────────────────────
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
        import pandas as pd
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

    # ── If Excel, convert to CSV first ────────────────────────────────────
    ext = safe_name.rsplit(".", 1)[1].lower()
    if ext in ("xlsx", "xls"):
        import pandas as pd
        df_raw = pd.read_excel(input_path)
        csv_path = os.path.join(tmp_dir, safe_name.rsplit(".", 1)[0] + ".csv")
        df_raw.to_csv(csv_path, index=False)
        input_path = csv_path

    # ── Run the cleaner ───────────────────────────────────────────────────
    try:
        cleaner = DataCleanPro(
            filepath=input_path,
            output_dir=output_dir,
            silent=True,
        )
        result = cleaner.run()
    except Exception as exc:
        return jsonify({
            "error": f"Cleaning failed: {str(exc)}",
            "detail": traceback.format_exc()
        }), 500

    excel_path = result["excel"]
    stats = result["stats"]

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
    response.headers["X-Rows-In"]         = str(stats.get("rows_in", ""))
    response.headers["X-Rows-Out"]        = str(stats.get("rows_out", ""))
    response.headers["X-Rows-Removed"]    = str(stats.get("rows_removed", ""))
    response.headers["X-Missing-Before"]  = str(stats.get("missing_before", ""))
    response.headers["X-Missing-After"]   = str(stats.get("missing_after", ""))
    response.headers["Access-Control-Expose-Headers"] = (
        "X-Rows-In, X-Rows-Out, X-Rows-Removed, X-Missing-Before, X-Missing-After"
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
        import pandas as pd
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
    print("  Powered by Eduxellence Analytics · https://eduxellence.org\n")
    app.run(debug=True, host="0.0.0.0", port=port)
