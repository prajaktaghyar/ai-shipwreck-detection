import os
import cv2
import numpy as np


# ============================================================
# SONAR IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    input_path,
    output_directory,
    width=1024,
    height=1024
):

    os.makedirs(output_directory, exist_ok=True)

    image = cv2.imread(
        input_path,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {input_path}"
        )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Noise reduction
    # --------------------------------------------------------

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # --------------------------------------------------------
    # CLAHE contrast enhancement
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # --------------------------------------------------------
    # Convert back to 3-channel
    # --------------------------------------------------------

    output = cv2.cvtColor(
        enhanced,
        cv2.COLOR_GRAY2BGR
    )

    filename = os.path.basename(input_path)

    name, ext = os.path.splitext(filename)

    output_path = os.path.join(
        output_directory,
        f"processed_{name}.png"
    )

    cv2.imwrite(
        output_path,
        output
    )

    return output_path


# ============================================================
# OPTIONAL SIMPLE PREPROCESSOR
# ============================================================

def enhance_sonar_image(image):

    if image is None:
        return None

    if len(image.shape) == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    else:
        gray = image.copy()

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return cv2.cvtColor(
        enhanced,
        cv2.COLOR_GRAY2BGR
    )