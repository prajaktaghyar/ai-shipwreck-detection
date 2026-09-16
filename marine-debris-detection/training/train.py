import os
import shutil
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "yolo_dataset"
)

DATASET_YAML = os.path.join(
    DATASET_DIR,
    "dataset.yaml"
)

BASE_MODEL = "yolov8n.pt"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "runs"
)

EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 8
WORKERS = 4


# ============================================================
# PRINT DIRECTORY STRUCTURE
# ============================================================

def print_config():

    print("=" * 70)
    print("MARINE OBJECT YOLO TRAINING")
    print("=" * 70)

    print("Project directory:")
    print(BASE_DIR)

    print()
    print("Dataset directory:")
    print(DATASET_DIR)

    print()
    print("Dataset YAML:")
    print(DATASET_YAML)

    print()
    print("Base model:")
    print(BASE_MODEL)

    print()
    print("Epochs:")
    print(EPOCHS)

    print()
    print("Image size:")
    print(IMAGE_SIZE)

    print()
    print("Batch size:")
    print(BATCH_SIZE)

    print()
    print("Output directory:")
    print(OUTPUT_DIR)

    print("=" * 70)


# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset():

    print()
    print("=" * 70)
    print("CHECKING DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # YAML
    # --------------------------------------------------------

    if not os.path.isfile(DATASET_YAML):

        print()
        print("[ERROR] dataset.yaml not found!")
        print()
        print("Expected:")
        print(DATASET_YAML)
        print()

        return False

    print("[OK] dataset.yaml found")
    print(DATASET_YAML)

    # --------------------------------------------------------
    # REQUIRED DIRECTORIES
    # --------------------------------------------------------

    required_directories = {

        "Train images": os.path.join(
            DATASET_DIR,
            "train",
            "images"
        ),

        "Train labels": os.path.join(
            DATASET_DIR,
            "train",
            "labels"
        ),

        "Validation images": os.path.join(
            DATASET_DIR,
            "val",
            "images"
        ),

        "Validation labels": os.path.join(
            DATASET_DIR,
            "val",
            "labels"
        )
    }

    print()

    all_valid = True

    for name, path in required_directories.items():

        if os.path.isdir(path):

            print(f"[OK] {name}:")
            print(f"     {path}")

        else:

            print(f"[MISSING] {name}:")
            print(f"          {path}")

            all_valid = False

    # --------------------------------------------------------
    # COUNT FILES
    # --------------------------------------------------------

    train_images = os.path.join(
        DATASET_DIR,
        "train",
        "images"
    )

    train_labels = os.path.join(
        DATASET_DIR,
        "train",
        "labels"
    )

    val_images = os.path.join(
        DATASET_DIR,
        "val",
        "images"
    )

    val_labels = os.path.join(
        DATASET_DIR,
        "val",
        "labels"
    )

    if os.path.isdir(train_images):

        image_count = len([
            f for f in os.listdir(train_images)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            )
        ])

        print()
        print("Training images:", image_count)

    if os.path.isdir(train_labels):

        label_count = len([
            f for f in os.listdir(train_labels)
            if f.lower().endswith(".txt")
        ])

        print("Training labels:", label_count)

    if os.path.isdir(val_images):

        image_count = len([
            f for f in os.listdir(val_images)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            )
        ])

        print("Validation images:", image_count)

    if os.path.isdir(val_labels):

        label_count = len([
            f for f in os.listdir(val_labels)
            if f.lower().endswith(".txt")
        ])

        print("Validation labels:", label_count)

    print()

    if not all_valid:

        print("=" * 70)
        print("[ERROR] DATASET STRUCTURE IS INVALID")
        print("=" * 70)

        print()
        print("Expected structure:")
        print()

        print(
            os.path.join(
                DATASET_DIR,
                "train",
                "images"
            )
        )

        print(
            os.path.join(
                DATASET_DIR,
                "train",
                "labels"
            )
        )

        print(
            os.path.join(
                DATASET_DIR,
                "val",
                "images"
            )
        )

        print(
            os.path.join(
                DATASET_DIR,
                "val",
                "labels"
            )
        )

        print()

        return False

    print("[SUCCESS] Dataset structure looks correct.")

    return True


# ============================================================
# CHECK LABELS
# ============================================================

def check_labels():

    print()
    print("=" * 70)
    print("CHECKING YOLO LABELS")
    print("=" * 70)

    label_directories = [
        os.path.join(
            DATASET_DIR,
            "train",
            "labels"
        ),
        os.path.join(
            DATASET_DIR,
            "val",
            "labels"
        )
    ]

    valid_classes = {
        0,
        1,
        2
    }

    total_labels = 0
    invalid_labels = 0

    for label_directory in label_directories:

        if not os.path.isdir(label_directory):
            continue

        for filename in os.listdir(label_directory):

            if not filename.endswith(".txt"):
                continue

            filepath = os.path.join(
                label_directory,
                filename
            )

            try:

                with open(
                    filepath,
                    "r",
                    encoding="utf-8"
                ) as file:

                    lines = file.readlines()

                for line_number, line in enumerate(
                    lines,
                    start=1
                ):

                    line = line.strip()

                    if not line:
                        continue

                    parts = line.split()

                    if len(parts) != 5:

                        print(
                            f"[WARNING] Invalid label: "
                            f"{filepath}:{line_number}"
                        )

                        invalid_labels += 1
                        continue

                    class_id = int(parts[0])

                    if class_id not in valid_classes:

                        print(
                            f"[WARNING] Invalid class "
                            f"{class_id} in {filepath}"
                        )

                        invalid_labels += 1

                    total_labels += 1

            except Exception as e:

                print(
                    f"[WARNING] Could not read {filepath}: {e}"
                )

                invalid_labels += 1

    print()
    print("Total label entries:", total_labels)
    print("Invalid labels:", invalid_labels)

    if invalid_labels == 0:

        print("[OK] All checked labels use classes 0, 1, 2.")

    else:

        print(
            "[WARNING] Some labels have problems. "
            "Training may fail or produce poor results."
        )


# ============================================================
# TRAIN
# ============================================================

def train():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print_config()

    # --------------------------------------------------------
    # DATASET CHECK
    # --------------------------------------------------------

    if not check_dataset():

        print()
        print("[STOP] Fix the dataset structure first.")
        return

    # --------------------------------------------------------
    # LABEL CHECK
    # --------------------------------------------------------

    check_labels()

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING YOLO MODEL")
    print("=" * 70)

    print()
    print("Model:", BASE_MODEL)

    model = YOLO(BASE_MODEL)

    print("[OK] YOLO model loaded.")

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    print()

    results = model.train(

        data=DATASET_YAML,

        epochs=EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        workers=WORKERS,

        project=OUTPUT_DIR,

        name="marine_debris_training",

        patience=20,

        pretrained=True,

        save=True,

        plots=True,

        verbose=True,

        device="cpu"
    )

    # --------------------------------------------------------
    # FIND ACTUAL SAVE DIRECTORY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINDING TRAINING OUTPUT")
    print("=" * 70)

    save_dir = None

    try:

        save_dir = str(results.save_dir)

        print()
        print("Ultralytics save directory:")
        print(save_dir)

    except Exception:

        print(
            "[WARNING] Could not determine save directory "
            "from results."
        )

    # --------------------------------------------------------
    # FIND BEST MODEL
    # --------------------------------------------------------

    best_model = None

    if save_dir:

        possible_best = os.path.join(
            save_dir,
            "weights",
            "best.pt"
        )

        if os.path.isfile(possible_best):

            best_model = possible_best

    # Fallback search
    if best_model is None:

        for root, dirs, files in os.walk(OUTPUT_DIR):

            if "best.pt" in files:

                candidate = os.path.join(
                    root,
                    "best.pt"
                )

                if os.path.basename(root) == "weights":

                    best_model = candidate
                    break

    # --------------------------------------------------------
    # COPY BEST MODEL
    # --------------------------------------------------------

    if best_model and os.path.isfile(best_model):

        destination = os.path.join(
            OUTPUT_DIR,
            "best.pt"
        )

        shutil.copy2(
            best_model,
            destination
        )

        print()
        print("=" * 70)
        print("BEST MODEL FOUND")
        print("=" * 70)

        print()
        print("Training best.pt:")
        print(best_model)

        print()
        print("Final model:")
        print(destination)

        print()
        print("[SUCCESS] best.pt copied successfully.")

    else:

        print()
        print("=" * 70)
        print("[WARNING] best.pt NOT FOUND")
        print("=" * 70)

        print()
        print(
            "Check the models directory manually:"
        )

        print(OUTPUT_DIR)

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print()
    print("Your model should be here:")

    print(
        os.path.join(
            OUTPUT_DIR,
            "best.pt"
        )
    )

    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()