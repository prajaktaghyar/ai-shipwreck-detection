import os

import json

import csv

from datetime import datetime



from flask import (

    Flask,

    render_template,

    request,

    jsonify,

    send_from_directory

)

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



# ============================================================

# TRAINED YOLO MODEL

# ============================================================

# Your actual trained model is located at:

#

# runs/

# └── marine_debris_training/

#     └── weights/

#         └── best.pt

#

# ============================================================



MODEL_PATH = os.path.join(

    BASE_DIR,

    "runs",

    "marine_debris_training",

    "weights",

    "best.pt"

)





# ============================================================

# ALLOWED IMAGE TYPES

# ============================================================



ALLOWED_EXTENSIONS = {

    "jpg",

    "jpeg",

    "png",

    "bmp",

    "tif",

    "tiff"

}





# ============================================================

# CREATE REQUIRED DIRECTORIES

# ============================================================



os.makedirs(

    UPLOAD_FOLDER,

    exist_ok=True

)



os.makedirs(

    RESULT_FOLDER,

    exist_ok=True

)



os.makedirs(

    REPORT_FOLDER,

    exist_ok=True

)





# ============================================================

# FLASK APP

# ============================================================



app = Flask(__name__)



app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["RESULT_FOLDER"] = RESULT_FOLDER



# Maximum upload size = 50 MB

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024





# ============================================================

# MODEL PATH CHECK

# ============================================================



print("=" * 60)

print("MARINE DEBRIS DETECTION SYSTEM")

print("=" * 60)



print("Base directory:")

print(BASE_DIR)



print()



print("YOLO model path:")

print(MODEL_PATH)



print()



if os.path.exists(MODEL_PATH):



    print("Model file: FOUND")



    try:

        model_size = os.path.getsize(MODEL_PATH)



        print(

            "Model size:",

            model_size,

            "bytes"

        )



    except Exception:

        pass



else:



    print("Model file: NOT FOUND")



    print()

    print("Expected model location:")

    print(MODEL_PATH)



print("=" * 60)





# ============================================================

# YOLO DETECTOR

# ============================================================



try:



    detector = MarineDebrisDetector(

        model_path=MODEL_PATH,

        confidence_threshold=0.50,

        iou_threshold=0.35

    )



except Exception as e:



    print()

    print("=" * 60)

    print("ERROR WHILE LOADING YOLO MODEL")

    print("=" * 60)



    print(str(e))



    print("=" * 60)



    detector = None





# ============================================================

# FILE VALIDATION

# ============================================================



def allowed_file(filename):



    return (

        "." in filename

        and

        filename.rsplit(

            ".",

            1

        )[1].lower()

        in ALLOWED_EXTENSIONS

    )





# ============================================================

# SAVE DETECTION REPORT

# ============================================================



def save_report(result):



    csv_path = os.path.join(

        REPORT_FOLDER,

        "report.csv"

    )



    json_path = os.path.join(

        REPORT_FOLDER,

        "report.json"

    )



    timestamp = datetime.now().isoformat()



    # --------------------------------------------------------

    # CSV DATA

    # --------------------------------------------------------



    row = {

        "timestamp": timestamp,



        "filename": result.get(

            "filename",

            ""

        ),



        "status": result.get(

            "status",

            ""

        ),



        "detections": len(

            result.get(

                "detections",

                []

            )

        ),



        "best_class": result.get(

            "best_class",

            ""

        ),



        "best_confidence": result.get(

            "best_confidence",

            0

        )

    }



    # --------------------------------------------------------

    # SAVE CSV

    # --------------------------------------------------------



    file_exists = os.path.exists(

        csv_path

    )



    with open(

        csv_path,

        "a",

        newline="",

        encoding="utf-8"

    ) as f:



        writer = csv.DictWriter(

            f,

            fieldnames=row.keys()

        )



        if not file_exists:



            writer.writeheader()



        writer.writerow(row)



    # --------------------------------------------------------

    # SAVE JSON

    # --------------------------------------------------------



    records = []



    if os.path.exists(json_path):



        try:



            with open(

                json_path,

                "r",

                encoding="utf-8"

            ) as f:



                records = json.load(f)



            if not isinstance(

                records,

                list

            ):



                records = []



        except Exception:



            records = []



    records.append({

        "timestamp": timestamp,

        **result

    })



    with open(

        json_path,

        "w",

        encoding="utf-8"

    ) as f:



        json.dump(

            records,

            f,

            indent=4

        )





# ============================================================

# DASHBOARD

# ============================================================



@app.route("/")

def dashboard():



    if detector is not None:



        model_loaded = detector.model_loaded



    else:



        model_loaded = False



    return render_template(

        "dashboard.html",



        model_loaded=model_loaded,



        model_path=MODEL_PATH

    )





# ============================================================

# MODEL STATUS API

# ============================================================



@app.route("/api/status")

def status():



    if detector is not None:



        model_loaded = detector.model_loaded



        confidence_threshold = (

            detector.confidence_threshold

        )



        iou_threshold = (

            detector.iou_threshold

        )



    else:



        model_loaded = False



        confidence_threshold = 0.50



        iou_threshold = 0.35



    return jsonify({



        "model_loaded": model_loaded,



        "model_exists": os.path.exists(

            MODEL_PATH

        ),



        "model_path": MODEL_PATH,



        "confidence_threshold":

            confidence_threshold,



        "iou_threshold":

            iou_threshold,



        "time":

            datetime.now().isoformat()



    })





# ============================================================

# IMAGE UPLOAD + DETECTION

# ============================================================



@app.route(

    "/api/detect",

    methods=["POST"]

)

def detect():



    # --------------------------------------------------------

    # CHECK MODEL

    # --------------------------------------------------------



    if detector is None:



        return jsonify({



            "success": False,



            "error":

                "YOLO detector could not be initialized."



        }), 500





    if not detector.model_loaded:



        return jsonify({



            "success": False,



            "error":

                "YOLO model is not loaded. "

                "Check the model path: "

                + MODEL_PATH



        }), 500





    # --------------------------------------------------------

    # CHECK IMAGE

    # --------------------------------------------------------



    if "image" not in request.files:



        return jsonify({



            "success": False,



            "error":

                "No image uploaded"



        }), 400





    file = request.files["image"]





    if file.filename == "":



        return jsonify({



            "success": False,



            "error":

                "No file selected"



        }), 400





    # --------------------------------------------------------

    # CHECK IMAGE FORMAT

    # --------------------------------------------------------



    if not allowed_file(

        file.filename

    ):



        return jsonify({



            "success": False,



            "error":

                "Unsupported image format"



        }), 400





    # --------------------------------------------------------

    # SECURE FILE NAME

    # --------------------------------------------------------



    filename = secure_filename(

        file.filename

    )





    # --------------------------------------------------------

    # CREATE UNIQUE FILE NAME

    # --------------------------------------------------------



    timestamp = datetime.now().strftime(

        "%Y%m%d_%H%M%S_%f"

    )



    original_name = (

        f"{timestamp}_{filename}"

    )





    # --------------------------------------------------------

    # INPUT IMAGE PATH

    # --------------------------------------------------------



    input_path = os.path.join(

        UPLOAD_FOLDER,

        original_name

    )





    # --------------------------------------------------------

    # SAVE UPLOADED IMAGE

    # --------------------------------------------------------



    file.save(input_path)





    try:



        # ====================================================

        # PREPROCESS IMAGE

        # ====================================================



        processed_path = preprocess_image(

            input_path,

            UPLOAD_FOLDER

        )





        # ====================================================

        # YOLO DETECTION

        # ====================================================



        result = detector.detect(

            processed_path,

            RESULT_FOLDER

        )





        # ====================================================

        # ADD FILE INFORMATION

        # ====================================================



        result["filename"] = filename



        result["input_image"] = (

            "/uploads/"

            + original_name

        )





        # ====================================================

        # RESULT IMAGE URL

        # ====================================================



        if result.get(

            "result_image"

        ):



            result["result_image"] = (

                "/results/"

                +

                os.path.basename(

                    result["result_image"]

                )

            )





        # ====================================================

        # SAVE REPORT

        # ====================================================



        save_report(result)





        # ====================================================

        # RETURN RESULT

        # ====================================================



        return jsonify({



            "success": True,



            **result



        })





    except Exception as e:



        print()

        print("Detection error:")

        print(str(e))





        return jsonify({



            "success": False,



            "error": str(e)



        }), 500





# ============================================================

# SERVE UPLOADED IMAGES

# ============================================================



@app.route(

    "/uploads/<filename>"

)

def uploaded_file(filename):



    return send_from_directory(

        UPLOAD_FOLDER,

        filename

    )





# ============================================================

# SERVE RESULT IMAGES

# ============================================================



@app.route(

    "/results/<filename>"

)

def result_file(filename):



    return send_from_directory(

        RESULT_FOLDER,

        filename

    )





# ============================================================

# RUN FLASK SERVER

# ============================================================



if __name__ == "__main__":



    print()

    print("=" * 60)

    print("STARTING MARINE DEBRIS DETECTION SERVER")

    print("=" * 60)





    # --------------------------------------------------------

    # MODEL STATUS

    # --------------------------------------------------------



    if detector is not None:



        if detector.model_loaded:



            print(

                "YOLO MODEL: LOADED"

            )



        else:



            print(

                "YOLO MODEL: NOT LOADED"

            )



    else:



        print(

            "YOLO MODEL: NOT LOADED"

        )





    print()



    print(

        "Model path:"

    )



    print(

        MODEL_PATH

    )



    print()



    print(

        "Server:"

    )



    print(

        "http://127.0.0.1:5000"

    )



    print("=" * 60)





    # --------------------------------------------------------

    # START SERVER

    # --------------------------------------------------------



    app.run(

        host="0.0.0.0",

        port=5000,

        debug=False

    )