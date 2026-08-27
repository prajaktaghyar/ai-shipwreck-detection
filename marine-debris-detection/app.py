import os
import json
import csv
import time
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from preprocessing.preprocess import preprocess_image
from detection.detector import MarineDebrisDetector


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "results")
REPORT_FOLDER = os.path.join(BASE_DIR, "reports")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")

ALLOWED_EXTENSIONS = {
    "jpg", "jpeg", "png", "bmp", "tif", "tiff"
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024


# ============================================================
# DETECTOR
# ============================================================

detector = MarineDebrisDetector(
    model_path=MODEL_PATH,
    confidence_threshold=0.25,
    iou_threshold=0.45
)


# ============================================================
# HELPERS
# ============================================================

def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def save_report(result):
    csv_path = os.path.join(REPORT_FOLDER, "report.csv")
    json_path = os.path.join(REPORT_FOLDER, "report.json")

    timestamp = datetime.now().isoformat()

    row = {
        "timestamp": timestamp,
        "filename": result.get("filename", ""),
        "status": result.get("status", ""),
        "detections": len(result.get("detections", [])),
        "best_class": result.get("best_class", ""),
        "best_confidence": result.get("best_confidence", 0)
    }

    # ---------------- CSV ----------------

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=row.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    # ---------------- JSON ----------------

    records = []

    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                records = json.load(f)

            if not isinstance(records, list):
                records = []

        except Exception:
            records = []

    records.append({
        "timestamp": timestamp,
        **result
    })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            records,
            f,
            indent=4
        )


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def dashboard():
    return render_template(
        "dashboard.html",
        model_loaded=detector.model_loaded,
        model_path=MODEL_PATH
    )


# ============================================================
# MODEL STATUS
# ============================================================

@app.route("/api/status")
def status():

    return jsonify({
        "model_loaded": detector.model_loaded,
        "model_path": MODEL_PATH,
        "confidence_threshold": detector.confidence_threshold,
        "iou_threshold": detector.iou_threshold,
        "time": datetime.now().isoformat()
    })


# ============================================================
# IMAGE UPLOAD + DETECTION
# ============================================================

@app.route("/api/detect", methods=["POST"])
def detect():

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "No file selected"
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported image format"
        }), 400

    filename = secure_filename(file.filename)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    original_name = f"{timestamp}_{filename}"

    input_path = os.path.join(
        UPLOAD_FOLDER,
        original_name
    )

    file.save(input_path)

    try:

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        processed_path = preprocess_image(
            input_path,
            UPLOAD_FOLDER
        )

        # ----------------------------------------------------
        # DETECTION
        # ----------------------------------------------------

        result = detector.detect(
            processed_path,
            RESULT_FOLDER
        )

        result["filename"] = filename

        result["input_image"] = (
            "/uploads/" + original_name
        )

        if result.get("result_image"):
            result["result_image"] = (
                "/results/" +
                os.path.basename(result["result_image"])
            )

        save_report(result)

        return jsonify({
            "success": True,
            **result
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# SERVE UPLOADS
# ============================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ============================================================
# SERVE RESULTS
# ============================================================

@app.route("/results/<filename>")
def result_file(filename):

    return send_from_directory(
        RESULT_FOLDER,
        filename
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("MARINE DEBRIS DETECTION SYSTEM")
    print("=" * 60)

    print(
        "Model:",
        "LOADED" if detector.model_loaded else "NOT LOADED"
    )

    print("Model path:", MODEL_PATH)

    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )