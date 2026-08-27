import os
import sys
import cv2

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# ============================================================
# IMPORT YOLO
# ============================================================

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ============================================================
# CONFIDENCE FUNCTIONS
# ============================================================

def calculate_confidence_score(confidence):
    return round(float(confidence) * 100, 2)


def classify_confidence(confidence):
    confidence = float(confidence)

    if confidence >= 0.75:
        return "HIGH"
    elif confidence >= 0.50:
        return "MEDIUM"
    elif confidence >= 0.25:
        return "LOW"
    else:
        return "VERY_LOW"


def detection_status(confidence, threshold):
    if confidence >= threshold:
        return "DETECTED"

    return "BELOW_THRESHOLD"


# ============================================================
# MARINE DEBRIS / SHIPWRECK DETECTOR
# ============================================================

class MarineDebrisDetector:

    def __init__(
        self,
        model_path,
        confidence_threshold=0.15,
        iou_threshold=0.45
    ):

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        self.model = None
        self.model_loaded = False

        self._load_model()

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def _load_model(self):

        if YOLO is None:
            print("[ERROR] ultralytics is not installed.")
            return

        if not os.path.exists(self.model_path):
            print("[ERROR] Model not found:")
            print(self.model_path)
            return

        try:

            print("[INFO] Loading YOLO model...")
            print("[MODEL PATH]", self.model_path)

            self.model = YOLO(self.model_path)

            self.model_loaded = True

            print("[MODEL] YOLO model loaded successfully")

            try:
                print("[CLASSES]", self.model.names)
            except Exception:
                pass

        except Exception as e:

            self.model_loaded = False

            print("[MODEL ERROR]", str(e))

    # ========================================================
    # DETECT IMAGE
    # ========================================================

    def detect(
        self,
        image_path,
        result_directory
    ):

        # ----------------------------------------------------
        # MODEL CHECK
        # ----------------------------------------------------

        if not self.model_loaded:

            return {
                "status": "MODEL_NOT_LOADED",
                "message": "YOLO model is not loaded",
                "detections": [],
                "best_class": "",
                "best_confidence": 0,
                "result_image": None
            }

        # ----------------------------------------------------
        # RESULT DIRECTORY
        # ----------------------------------------------------

        os.makedirs(
            result_directory,
            exist_ok=True
        )

        # ----------------------------------------------------
        # IMAGE CHECK
        # ----------------------------------------------------

        if not os.path.exists(image_path):

            return {
                "status": "IMAGE_NOT_FOUND",
                "message": f"Image not found: {image_path}",
                "detections": [],
                "best_class": "",
                "best_confidence": 0,
                "result_image": None
            }

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image = cv2.imread(image_path)

        if image is None:

            return {
                "status": "IMAGE_ERROR",
                "message": "Unable to read image",
                "detections": [],
                "best_class": "",
                "best_confidence": 0,
                "result_image": None
            }

        print()
        print("[OK] Image loaded:")
        print(image_path)

        print()
        print("Image size:")
        print(f"{image.shape[1]} x {image.shape[0]}")

        # ====================================================
        # YOLO INFERENCE
        # ====================================================

        try:

            results = self.model.predict(
                source=image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=10,
                imgsz=640,
                verbose=False
            )

        except Exception as e:

            print("[YOLO ERROR]", str(e))

            return {
                "status": "INFERENCE_ERROR",
                "message": str(e),
                "detections": [],
                "best_class": "",
                "best_confidence": 0,
                "result_image": None
            }

        # ====================================================
        # PREPARE
        # ====================================================

        detections = []

        result_image = image.copy()

        best_detection = None

        # ====================================================
        # PROCESS RESULTS
        # ====================================================

        for result in results:

            if result.boxes is None:
                continue

            names = result.names

            for box in result.boxes:

                # ------------------------------------------------
                # CONFIDENCE
                # ------------------------------------------------

                confidence = float(
                    box.conf[0]
                )

                # ------------------------------------------------
                # CLASS ID
                # ------------------------------------------------

                class_id = int(
                    box.cls[0]
                )

                # ------------------------------------------------
                # CLASS NAME
                # ------------------------------------------------

                if isinstance(names, dict):

                    class_name = names.get(
                        class_id,
                        f"class_{class_id}"
                    )

                else:

                    class_name = str(
                        class_id
                    )

                # ------------------------------------------------
                # BOUNDING BOX
                # ------------------------------------------------

                coordinates = box.xyxy[0].tolist()

                x1, y1, x2, y2 = [
                    int(v)
                    for v in coordinates
                ]

                # ------------------------------------------------
                # CONFIDENCE
                # ------------------------------------------------

                confidence_percent = (
                    calculate_confidence_score(
                        confidence
                    )
                )

                confidence_level = (
                    classify_confidence(
                        confidence
                    )
                )

                status = (
                    detection_status(
                        confidence,
                        self.confidence_threshold
                    )
                )

                # ------------------------------------------------
                # DETECTION OBJECT
                # ------------------------------------------------

                item = {

                    "class_id": class_id,

                    "class_name": class_name,

                    "confidence": round(
                        confidence,
                        4
                    ),

                    "confidence_percent":
                        confidence_percent,

                    "confidence_level":
                        confidence_level,

                    "status":
                        status,

                    "bbox": {

                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2

                    }
                }

                detections.append(item)

                # =================================================
                # DRAW BOUNDING BOX
                # =================================================

                cv2.rectangle(
                    result_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # ------------------------------------------------
                # LABEL
                # ------------------------------------------------

                label = (
                    f"{class_name} "
                    f"{confidence_percent:.1f}%"
                )

                cv2.putText(
                    result_image,
                    label,
                    (
                        x1,
                        max(
                            25,
                            y1 - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                # =================================================
                # BEST DETECTION
                # =================================================

                if (
                    best_detection is None
                    or confidence >
                    best_detection["confidence"]
                ):

                    best_detection = item

        # ====================================================
        # SAVE RESULT IMAGE
        # ====================================================

        filename = os.path.basename(
            image_path
        )

        name, _ = os.path.splitext(
            filename
        )

        output_path = os.path.join(
            result_directory,
            f"detected_{name}.jpg"
        )

        cv2.imwrite(
            output_path,
            result_image
        )

        # ====================================================
        # NO DETECTION
        # ====================================================

        if best_detection is None:

            print()
            print("=" * 70)
            print("[YOLO] NO OBJECT DETECTED")
            print("=" * 70)

            print()
            print(
                "No object was detected above confidence:",
                f"{self.confidence_threshold * 100:.1f}%"
            )

            print()
            print("Result image:")
            print(output_path)

            return {

                "status":
                    "NO_OBJECT_DETECTED",

                "message":
                    "No object detected above threshold",

                "detections":
                    [],

                "best_class":
                    "",

                "best_confidence":
                    0,

                "result_image":
                    output_path
            }

        # ====================================================
        # PRINT DETECTIONS
        # ====================================================

        print()
        print("=" * 70)
        print("YOLO DETECTIONS")
        print("=" * 70)

        for detection in detections:

            print()

            print(
                "Class:",
                detection["class_name"]
            )

            print(
                "Confidence:",
                f"{detection['confidence_percent']:.2f}%"
            )

            print(
                "Level:",
                detection["confidence_level"]
            )

            print(
                "Status:",
                detection["status"]
            )

            print(
                "Bounding Box:",
                detection["bbox"]
            )

        # ====================================================
        # BEST DETECTION
        # ====================================================

        print()
        print("=" * 70)
        print("BEST DETECTION")
        print("=" * 70)

        print()

        print(
            "Class:",
            best_detection["class_name"]
        )

        print(
            "Confidence:",
            f"{best_detection['confidence_percent']:.2f}%"
        )

        print(
            "Status:",
            best_detection["status"]
        )

        # ====================================================
        # RESULT IMAGE
        # ====================================================

        print()
        print("Result image:")
        print(output_path)

        # ====================================================
        # FINAL RESULT
        # ====================================================

        return {

            "status":
                best_detection["status"],

            "message":
                "Object detected",

            "detections":
                detections,

            "best_class":
                best_detection["class_name"],

            "best_confidence":
                best_detection["confidence_percent"],

            "result_image":
                output_path
        }


# ============================================================
# STANDALONE IMAGE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("MARINE DEBRIS YOLO IMAGE DETECTOR")
    print("=" * 70)

    # ========================================================
    # PROJECT
    # ========================================================

    print()
    print("Project:")
    print(PROJECT_ROOT)

    # ========================================================
    # MODEL PATH
    # ========================================================

    MODEL_PATH = os.path.join(
        PROJECT_ROOT,
        "models",
        "best.pt"
    )

    # ========================================================
    # RESULT DIRECTORY
    # ========================================================

    RESULT_DIR = os.path.join(
        PROJECT_ROOT,
        "results"
    )

    # ========================================================
    # TEST PNG IMAGE
    # ========================================================

    IMAGE_PATH = os.path.join(
        PROJECT_ROOT,
        "uploads",
        "test.png"
    )

    # ========================================================
    # PRINT PATHS
    # ========================================================

    print()
    print("Model:")
    print(MODEL_PATH)

    print()
    print("Input PNG:")
    print(IMAGE_PATH)

    print()
    print("Result directory:")
    print(RESULT_DIR)

    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not os.path.exists(MODEL_PATH):

        print()
        print("[ERROR] best.pt does not exist!")
        print(MODEL_PATH)

        sys.exit(1)

    print()
    print("[OK] best.pt found")

    # ========================================================
    # CHECK PNG IMAGE
    # ========================================================

    if not os.path.exists(IMAGE_PATH):

        print()
        print("[ERROR] PNG test image not found!")
        print(IMAGE_PATH)

        print()
        print("Put your PNG image here:")

        print(
            os.path.join(
                PROJECT_ROOT,
                "uploads"
            )
        )

        print()
        print("Expected filename:")
        print("test.png")

        sys.exit(1)

    print()
    print("[OK] test.png found")

    # ========================================================
    # CREATE DETECTOR
    # ========================================================

    detector = MarineDebrisDetector(

        model_path=MODEL_PATH,

        confidence_threshold=0.15,

        iou_threshold=0.45
    )

    # ========================================================
    # CHECK MODEL
    # ========================================================

    if not detector.model_loaded:

        print()
        print("[ERROR] YOLO MODEL FAILED TO LOAD")

        sys.exit(1)

    # ========================================================
    # MODEL READY
    # ========================================================

    print()
    print("=" * 70)
    print("[SUCCESS] YOLO MODEL READY")
    print("=" * 70)

    try:

        print()
        print("Classes:")
        print(detector.model.names)

    except Exception:
        pass

    # ========================================================
    # RUN DETECTION
    # ========================================================

    print()
    print("=" * 70)
    print("RUNNING YOLO DETECTION")
    print("=" * 70)

    result = detector.detect(

        image_path=IMAGE_PATH,

        result_directory=RESULT_DIR
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL DETECTION RESULT")
    print("=" * 70)

    print()
    print("Status:")
    print(result["status"])

    print()
    print("Message:")
    print(result["message"])

    print()
    print("Best class:")
    print(result["best_class"])

    print()
    print("Best confidence:")
    print(
        f"{result['best_confidence']:.2f}%"
    )

    print()
    print("Number of detections:")
    print(
        len(result["detections"])
    )

    print()
    print("Result image:")
    print(result["result_image"])

    print()
    print("=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)