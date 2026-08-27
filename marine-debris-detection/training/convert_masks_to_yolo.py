import os
import cv2
import shutil
import numpy as np


# ============================================================
# AI4SHIPWRECKS -> YOLO CONVERTER
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SOURCE_DATASET = os.path.join(
    BASE_DIR,
    "dataset",
    "AI4Shipwrecks"
)

OUTPUT_DATASET = os.path.join(
    BASE_DIR,
    "dataset",
    "yolo_dataset"
)


# ============================================================
# SOURCE DIRECTORIES
# ============================================================

TRAIN_IMAGES = os.path.join(
    SOURCE_DATASET,
    "train",
    "images"
)

TRAIN_MASKS = os.path.join(
    SOURCE_DATASET,
    "train",
    "labels"
)

TEST_IMAGES = os.path.join(
    SOURCE_DATASET,
    "test",
    "images"
)

TEST_MASKS = os.path.join(
    SOURCE_DATASET,
    "test",
    "labels"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

YOLO_TRAIN_IMAGES = os.path.join(
    OUTPUT_DATASET,
    "train",
    "images"
)

YOLO_TRAIN_LABELS = os.path.join(
    OUTPUT_DATASET,
    "train",
    "labels"
)

YOLO_VAL_IMAGES = os.path.join(
    OUTPUT_DATASET,
    "val",
    "images"
)

YOLO_VAL_LABELS = os.path.join(
    OUTPUT_DATASET,
    "val",
    "labels"
)


# ============================================================
# YOLO CONFIG
# ============================================================

SHIPWRECK_CLASS_ID = 0

# Ignore extremely tiny regions
MIN_AREA = 50


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():

    directories = [
        YOLO_TRAIN_IMAGES,
        YOLO_TRAIN_LABELS,
        YOLO_VAL_IMAGES,
        YOLO_VAL_LABELS
    ]

    for directory in directories:
        os.makedirs(
            directory,
            exist_ok=True
        )


# ============================================================
# CLEAR OLD OUTPUT DATA
# ============================================================

def clear_old_output():

    print()
    print("[INFO] Cleaning old YOLO dataset...")

    for directory in [
        YOLO_TRAIN_IMAGES,
        YOLO_TRAIN_LABELS,
        YOLO_VAL_IMAGES,
        YOLO_VAL_LABELS
    ]:

        if not os.path.exists(directory):
            continue

        for filename in os.listdir(directory):

            path = os.path.join(
                directory,
                filename
            )

            if os.path.isfile(path):

                try:
                    os.remove(path)

                except Exception as e:

                    print(
                        "[WARNING] Could not remove:",
                        path,
                        e
                    )

    print("[OK] Old YOLO data removed.")


# ============================================================
# REMOVE CACHE
# ============================================================

def remove_cache_files():

    print("[INFO] Removing cache files...")

    if not os.path.exists(OUTPUT_DATASET):
        return

    for root, dirs, files in os.walk(
        OUTPUT_DATASET
    ):

        for filename in files:

            if filename.endswith(".cache"):

                path = os.path.join(
                    root,
                    filename
                )

                try:

                    os.remove(path)

                    print(
                        "[REMOVED CACHE]",
                        path
                    )

                except Exception as e:

                    print(
                        "[WARNING] Could not remove cache:",
                        e
                    )


# ============================================================
# FIND MATCHING MASK
# ============================================================

def find_mask(
    mask_directory,
    image_filename
):

    image_name, _ = os.path.splitext(
        image_filename
    )

    possible_extensions = [
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff"
    ]

    for extension in possible_extensions:

        mask_path = os.path.join(
            mask_directory,
            image_name + extension
        )

        if os.path.exists(mask_path):

            return mask_path

    return None


# ============================================================
# CREATE BINARY MASK
# ============================================================

def create_binary_mask(mask):

    # Any non-zero pixel = foreground
    binary = np.zeros_like(
        mask,
        dtype=np.uint8
    )

    binary[mask > 0] = 255

    return binary


# ============================================================
# CONVERT MASK TO YOLO
# ============================================================

def convert_mask_to_yolo(
    mask_path,
    output_label_path
):

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    # --------------------------------------------------------
    # MASK READ ERROR
    # --------------------------------------------------------

    if mask is None:

        print(
            "[ERROR] Cannot read mask:",
            mask_path
        )

        return False, 0, "error"


    height, width = mask.shape

    unique_values = np.unique(mask)

    foreground_pixels = np.count_nonzero(mask)


    print()
    print(
        "[MASK]",
        os.path.basename(mask_path)
    )

    print(
        "       Size:",
        width,
        "x",
        height
    )

    print(
        "       Unique:",
        unique_values[:20]
    )

    print(
        "       Nonzero:",
        foreground_pixels
    )


    # ========================================================
    # EMPTY MASK
    # ========================================================

    if foreground_pixels == 0:

        print(
            "       -> NEGATIVE / NO OBJECT"
        )

        # Empty YOLO label is valid
        with open(
            output_label_path,
            "w",
            encoding="utf-8"
        ):
            pass

        return True, 0, "negative"


    # ========================================================
    # BINARY MASK
    # ========================================================

    binary = create_binary_mask(mask)


    # ========================================================
    # MORPHOLOGICAL CLEANUP
    # ========================================================

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )


    # ========================================================
    # FIND OBJECTS
    # ========================================================

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    annotations = []


    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        # Ignore tiny noise
        if area < MIN_AREA:
            continue


        x, y, w, h = cv2.boundingRect(
            contour
        )


        if w <= 0 or h <= 0:
            continue


        # ====================================================
        # YOLO NORMALIZED COORDINATES
        # ====================================================

        center_x = (
            x + w / 2.0
        ) / width

        center_y = (
            y + h / 2.0
        ) / height

        box_width = w / width

        box_height = h / height


        # ====================================================
        # CLAMP
        # ====================================================

        center_x = np.clip(
            center_x,
            0.0,
            1.0
        )

        center_y = np.clip(
            center_y,
            0.0,
            1.0
        )

        box_width = np.clip(
            box_width,
            0.0,
            1.0
        )

        box_height = np.clip(
            box_height,
            0.0,
            1.0
        )


        # ====================================================
        # YOLO LABEL
        # ====================================================

        annotation = (
            f"{SHIPWRECK_CLASS_ID} "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

        annotations.append(
            annotation
        )


    # ========================================================
    # NO VALID CONTOUR
    # ========================================================

    if len(annotations) == 0:

        print(
            "       -> FOREGROUND EXISTS "
            "BUT NO VALID OBJECT"
        )

        with open(
            output_label_path,
            "w",
            encoding="utf-8"
        ):
            pass

        return True, 0, "empty_after_filter"


    # ========================================================
    # WRITE LABEL
    # ========================================================

    with open(
        output_label_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(
                annotations
            )
        )


    print(
        "       -> OBJECTS:",
        len(annotations)
    )


    return (
        True,
        len(annotations),
        "positive"
    )


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    split_name,
    image_directory,
    mask_directory,
    output_image_directory,
    output_label_directory
):

    print()
    print("=" * 70)
    print(
        f"PROCESSING {split_name.upper()}"
    )
    print("=" * 70)


    if not os.path.exists(
        image_directory
    ):

        print(
            "[ERROR] Image directory not found:"
        )

        print(
            image_directory
        )

        return {}


    if not os.path.exists(
        mask_directory
    ):

        print(
            "[ERROR] Mask directory not found:"
        )

        print(
            mask_directory
        )

        return {}


    os.makedirs(
        output_image_directory,
        exist_ok=True
    )

    os.makedirs(
        output_label_directory,
        exist_ok=True
    )


    image_files = sorted(
        os.listdir(
            image_directory
        )
    )


    total = 0
    converted = 0

    positive_images = 0
    negative_images = 0

    failed_images = 0

    total_objects = 0


    for filename in image_files:

        if not filename.lower().endswith(
            (
                ".png",
                ".jpg",
                ".jpeg",
                ".bmp",
                ".tif",
                ".tiff"
            )
        ):
            continue


        total += 1


        image_path = os.path.join(
            image_directory,
            filename
        )


        mask_path = find_mask(
            mask_directory,
            filename
        )


        # ----------------------------------------------------
        # NO MASK
        # ----------------------------------------------------

        if mask_path is None:

            print(
                "[WARNING] No matching mask:",
                filename
            )

            failed_images += 1

            continue


        name, extension = os.path.splitext(
            filename
        )


        output_image_path = os.path.join(
            output_image_directory,
            filename
        )


        output_label_path = os.path.join(
            output_label_directory,
            name + ".txt"
        )


        # ----------------------------------------------------
        # COPY IMAGE
        # ----------------------------------------------------

        try:

            shutil.copy2(
                image_path,
                output_image_path
            )

        except Exception as e:

            print(
                "[ERROR] Could not copy image:",
                filename,
                e
            )

            failed_images += 1

            continue


        # ----------------------------------------------------
        # CONVERT MASK
        # ----------------------------------------------------

        success, object_count, status = (
            convert_mask_to_yolo(
                mask_path,
                output_label_path
            )
        )


        if not success:

            failed_images += 1
            continue


        converted += 1

        total_objects += object_count


        if status == "positive":

            positive_images += 1

        elif status == "negative":

            negative_images += 1


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        f"{split_name} images found      : {total}"
    )

    print(
        f"{split_name} converted          : {converted}"
    )

    print(
        f"{split_name} positive images    : "
        f"{positive_images}"
    )

    print(
        f"{split_name} negative images    : "
        f"{negative_images}"
    )

    print(
        f"{split_name} failed             : "
        f"{failed_images}"
    )

    print(
        f"{split_name} annotations        : "
        f"{total_objects}"
    )


    return {
        "total": total,
        "converted": converted,
        "positive": positive_images,
        "negative": negative_images,
        "failed": failed_images,
        "objects": total_objects
    }


# ============================================================
# CREATE DATASET YAML
# ============================================================

def create_dataset_yaml():

    yaml_path = os.path.join(
        OUTPUT_DATASET,
        "dataset.yaml"
    )


    content = """path: .

train: train/images
val: val/images

names:
  0: shipwreck
"""


    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(content)


    print()
    print(
        "[OK] dataset.yaml created:"
    )

    print(
        yaml_path
    )


# ============================================================
# VERIFY YOLO LABELS
# ============================================================

def verify_labels(
    label_directory,
    image_directory,
    split_name
):

    print()
    print("=" * 70)
    print(
        f"{split_name.upper()} LABEL VERIFICATION"
    )
    print("=" * 70)


    label_files = [
        f
        for f in os.listdir(label_directory)
        if f.lower().endswith(".txt")
    ]


    empty_labels = 0
    non_empty_labels = 0

    total_annotations = 0

    invalid_labels = 0


    for filename in label_files:

        path = os.path.join(
            label_directory,
            filename
        )


        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            lines = [
                line.strip()
                for line in f.readlines()
                if line.strip()
            ]


        # ----------------------------------------------------
        # EMPTY LABEL
        # ----------------------------------------------------

        if len(lines) == 0:

            empty_labels += 1

            continue


        non_empty_labels += 1

        total_annotations += len(lines)


        # ----------------------------------------------------
        # VALIDATE
        # ----------------------------------------------------

        for line_number, line in enumerate(
            lines,
            start=1
        ):

            parts = line.split()


            if len(parts) != 5:

                print(
                    "[ERROR] Invalid YOLO label:",
                    filename,
                    "line",
                    line_number
                )

                print(
                    "       ",
                    line
                )

                invalid_labels += 1

                continue


            try:

                class_id = int(parts[0])

                values = [
                    float(x)
                    for x in parts[1:]
                ]


                if class_id != 0:

                    print(
                        "[ERROR] Invalid class:",
                        filename
                    )

                    invalid_labels += 1


                for value in values:

                    if value < 0 or value > 1:

                        print(
                            "[ERROR] Value outside 0-1:",
                            filename,
                            line_number
                        )

                        invalid_labels += 1


            except ValueError:

                print(
                    "[ERROR] Non-numeric label:",
                    filename,
                    line_number
                )

                invalid_labels += 1


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print(
        f"Total label files : {len(label_files)}"
    )

    print(
        f"Empty labels      : {empty_labels}"
    )

    print(
        f"Non-empty labels  : {non_empty_labels}"
    )

    print(
        f"Total annotations  : {total_annotations}"
    )

    print(
        f"Invalid labels     : {invalid_labels}"
    )


    if invalid_labels == 0:

        print(
            "[SUCCESS] YOLO label format is valid."
        )

    else:

        print(
            "[ERROR] Invalid YOLO labels found."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "AI4SHIPWRECKS -> YOLO CONVERTER"
    )
    print("=" * 70)


    print()
    print(
        "Source dataset:"
    )

    print(
        SOURCE_DATASET
    )


    print()
    print(
        "Output dataset:"
    )

    print(
        OUTPUT_DATASET
    )


    # ========================================================
    # CHECK SOURCE
    # ========================================================

    if not os.path.exists(
        SOURCE_DATASET
    ):

        print()
        print(
            "[ERROR] AI4Shipwrecks dataset not found."
        )

        print(
            SOURCE_DATASET
        )

        return


    # ========================================================
    # CREATE DIRECTORIES
    # ========================================================

    create_directories()


    # ========================================================
    # CLEAN OUTPUT
    # ========================================================

    clear_old_output()

    remove_cache_files()


    # ========================================================
    # TRAIN
    # ========================================================

    train_stats = process_split(
        "train",
        TRAIN_IMAGES,
        TRAIN_MASKS,
        YOLO_TRAIN_IMAGES,
        YOLO_TRAIN_LABELS
    )


    # ========================================================
    # TEST -> VALIDATION
    # ========================================================

    val_stats = process_split(
        "validation",
        TEST_IMAGES,
        TEST_MASKS,
        YOLO_VAL_IMAGES,
        YOLO_VAL_LABELS
    )


    # ========================================================
    # YAML
    # ========================================================

    create_dataset_yaml()


    # ========================================================
    # VERIFY
    # ========================================================

    verify_labels(
        YOLO_TRAIN_LABELS,
        YOLO_TRAIN_IMAGES,
        "train"
    )


    verify_labels(
        YOLO_VAL_LABELS,
        YOLO_VAL_IMAGES,
        "validation"
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "CONVERSION COMPLETED"
    )
    print("=" * 70)


    print()

    print(
        "TRAIN"
    )

    print(
        f"  Images       : "
        f"{train_stats.get('converted', 0)}"
    )

    print(
        f"  Positive     : "
        f"{train_stats.get('positive', 0)}"
    )

    print(
        f"  Negative     : "
        f"{train_stats.get('negative', 0)}"
    )

    print(
        f"  Objects      : "
        f"{train_stats.get('objects', 0)}"
    )


    print()

    print(
        "VALIDATION"
    )

    print(
        f"  Images       : "
        f"{val_stats.get('converted', 0)}"
    )

    print(
        f"  Positive     : "
        f"{val_stats.get('positive', 0)}"
    )

    print(
        f"  Negative     : "
        f"{val_stats.get('negative', 0)}"
    )

    print(
        f"  Objects      : "
        f"{val_stats.get('objects', 0)}"
    )


    print()
    print(
        "YOLO DATASET:"
    )

    print(
        OUTPUT_DATASET
    )


    print()
    print(
        "DATASET YAML:"
    )

    print(
        os.path.join(
            OUTPUT_DATASET,
            "dataset.yaml"
        )
    )


    print()
    print(
        "Next step:"
    )

    print(
        "Run training/train.py"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()