"""
ML Mirchi — Dataset Preparation Script
Organizes raw chilli images into train/val/test splits with proper structure.
Supports custom datasets or generates synthetic samples for development.
"""

import os
import shutil
import random
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"

CLASSES = ["Grade_A", "Grade_B", "Grade_C", "Reject"]
CLASS_DESCRIPTIONS = {
    "Grade_A": "Premium: Deep red, uniform color, no defects, no cracks, proper shape",
    "Grade_B": "Standard: Minor color variation, slight surface imperfections, acceptable shape",
    "Grade_C": "Sub-standard: Visible discoloration, wrinkles, green spots, minor damage",
    "Reject": "Rejected: Severe rot, mold, deep cracks, pest damage, contamination",
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# Test = remaining 0.15

SEED = 42
IMAGE_SIZE = (224, 224)  # Standard for transfer learning


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def create_directory_structure():
    """Create the processed directory structure."""
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            dir_path = PROCESSED_DIR / split / cls
            dir_path.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Directory structure created at {PROCESSED_DIR}")


def split_existing_dataset():
    """
    If you already have images organized in raw/Grade_A/, raw/Grade_B/, etc.
    this splits them into train/val/test.
    """
    if not RAW_DIR.exists():
        print("[SKIP] No raw/ directory found. Run generate_synthetic_dataset() first.")
        return False

    total_moved = 0
    for cls in CLASSES:
        cls_dir = RAW_DIR / cls
        if not cls_dir.exists():
            print(f"[WARN] {cls_dir} not found, skipping.")
            continue

        images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))
        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }

        for split_name, split_images in splits.items():
            for img_path in split_images:
                dest = PROCESSED_DIR / split_name / cls / img_path.name
                shutil.copy2(img_path, dest)
                total_moved += 1

        print(f"  {cls}: {n} images -> train={n_train}, val={n_val}, test={n - n_train - n_val}")

    print(f"[OK] Moved {total_moved} images to processed/")
    return total_moved > 0


# ============================================================
# SYNTHETIC DATA GENERATION (for development/testing)
# ============================================================
def generate_synthetic_dataset(images_per_class=500):
    """
    Generate synthetic chilli images for development and testing.
    NOT a replacement for real data — use this to validate your pipeline
    before collecting real chilli photographs.
    """
    print(f"[GEN] Generating {images_per_class} synthetic images per class...")
    set_seed(SEED)

    grade_params = {
        "Grade_A": {
            "hue_range": (0, 15),       # Deep red hues
            "sat_range": (0.7, 1.0),
            "val_range": (0.5, 0.8),
            "defect_prob": 0.0,
            "shape_variance": 0.1,
            "size_range": (0.3, 0.5),
        },
        "Grade_B": {
            "hue_range": (0, 30),       # Red to orange-red
            "sat_range": (0.5, 0.85),
            "val_range": (0.4, 0.75),
            "defect_prob": 0.2,
            "shape_variance": 0.2,
            "size_range": (0.25, 0.5),
        },
        "Grade_C": {
            "hue_range": (0, 60),       # Red to yellow-green
            "sat_range": (0.3, 0.7),
            "val_range": (0.3, 0.65),
            "defect_prob": 0.6,
            "shape_variance": 0.35,
            "size_range": (0.2, 0.45),
        },
        "Reject": {
            "hue_range": (0, 120),      # Wide range including brown/green
            "sat_range": (0.1, 0.5),
            "val_range": (0.15, 0.5),
            "defect_prob": 0.9,
            "shape_variance": 0.5,
            "size_range": (0.15, 0.4),
        },
    }

    for cls, params in grade_params.items():
        for i in range(images_per_class):
            img = generate_single_chilli(params, seed=SEED + hash(f"{cls}_{i}"))
            img = img.resize(IMAGE_SIZE, Image.LANCZOS)

            # Save to raw directory first
            raw_cls_dir = RAW_DIR / cls
            raw_cls_dir.mkdir(parents=True, exist_ok=True)
            img.save(raw_cls_dir / f"{cls}_{i:05d}.jpg", quality=90)

    print(f"[OK] Generated {images_per_class * len(CLASSES)} synthetic images in raw/")


def generate_single_chilli(params, seed=0):
    """Generate a single synthetic chilli image."""
    rng = np.random.RandomState(seed)
    w, h = 256, 256
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)

    # Background — conveyor belt / sorting surface
    bg_val = rng.randint(20, 45)
    draw.rectangle([0, 0, w, h], fill=(bg_val, bg_val - 5, bg_val - 10))

    # Add subtle texture to background
    for _ in range(200):
        x, y = rng.randint(0, w), rng.randint(0, h)
        c = bg_val + rng.randint(-10, 10)
        draw.ellipse([x, y, x + 2, y + 2], fill=(max(0, c), max(0, c - 5), max(0, c - 10)))

    # Chilli parameters
    num_chillies = rng.randint(1, 4)
    for _ in range(num_chillies):
        cx = rng.randint(60, w - 60)
        cy = rng.randint(60, h - 60)
        angle = rng.uniform(-0.6, 0.6)
        length = rng.randint(60, 130)
        width = rng.randint(15, 30)

        hue = rng.uniform(*params["hue_range"])
        sat = rng.uniform(*params["sat_range"])
        val = rng.uniform(*params["val_range"])

        # HSV to RGB
        r, g, b = hsv_to_rgb(hue / 360.0, sat, val)
        base_color = (int(r * 255), int(g * 255), int(b * 255))
        dark_color = (max(0, int(r * 200)), max(0, int(g * 180)), max(0, int(b * 160)))
        light_color = (min(255, int(r * 255) + 30), min(255, int(g * 255) + 15), min(255, int(b * 255) + 10))

        # Draw chilli body using polygon
        points = []
        n_points = 20
        for j in range(n_points + 1):
            t = j / n_points
            # Tapered shape: wide in middle, narrow at ends
            taper = np.sin(t * np.pi) ** 0.7
            radius = (width / 2) * taper * (1 + rng.uniform(-params["shape_variance"], params["shape_variance"]))
            px = cx + int(t * length * np.cos(angle) - radius * np.sin(angle))
            py = cy + int(t * length * np.sin(angle) + radius * np.cos(angle))
            points.append((px, py))
        for j in range(n_points, -1, -1):
            t = j / n_points
            taper = np.sin(t * np.pi) ** 0.7
            radius = (width / 2) * taper * (1 + rng.uniform(-params["shape_variance"], params["shape_variance"]))
            px = cx + int(t * length * np.cos(angle) + radius * np.sin(angle))
            py = cy + int(t * length * np.sin(angle) - radius * np.cos(angle))
            points.append((px, py))

        if len(points) > 2:
            draw.polygon(points, fill=base_color)
            # Highlight stripe
            stripe_points = []
            for j in range(n_points + 1):
                t = j / n_points
                taper = np.sin(t * np.pi) ** 0.7 * 0.3
                px = cx + int(t * length * np.cos(angle) - taper * width * np.sin(angle))
                py = cy + int(t * length * np.sin(angle) + taper * width * np.cos(angle))
                stripe_points.append((px, py))
            if len(stripe_points) > 1:
                for j in range(len(stripe_points) - 1):
                    draw.line([stripe_points[j], stripe_points[j + 1]], fill=light_color, width=2)

            # Stem
            stem_start_x = cx - int(15 * np.cos(angle))
            stem_start_y = cy - int(15 * np.sin(angle))
            stem_end_x = stem_start_x - int(rng.randint(10, 20) * np.cos(angle + 0.3))
            stem_end_y = stem_start_y - int(rng.randint(10, 20) * np.sin(angle + 0.3))
            draw.line([stem_start_x, stem_start_y, stem_end_x, stem_end_y], fill=(45, 80, 22), width=3)

    # Add defects
    if rng.random() < params["defect_prob"]:
        num_defects = rng.randint(1, 6)
        for _ in range(num_defects):
            dx, dy = rng.randint(40, w - 40), rng.randint(40, h - 40)
            dr = rng.randint(3, 12)
            defect_type = rng.choice(["dark_spot", "green_patch", "crack"])
            if defect_type == "dark_spot":
                draw.ellipse([dx - dr, dy - dr, dx + dr, dy + dr], fill=(20, 18, 15))
            elif defect_type == "green_patch":
                draw.ellipse([dx - dr, dy - dr, dx + dr, dy + dr], fill=(40 + rng.randint(0, 40), 70 + rng.randint(0, 40), 20))
            elif defect_type == "crack":
                for seg in range(rng.randint(2, 5)):
                    x1 = dx + rng.randint(-dr, dr)
                    y1 = dy + rng.randint(-dr, dr)
                    x2 = x1 + rng.randint(-15, 15)
                    y2 = y1 + rng.randint(-15, 15)
                    draw.line([(x1, y1), (x2, y2)], fill=(30, 25, 20), width=1)

    # Add noise and blur for realism
    arr = np.array(img, dtype=np.float32)
    noise = rng.normal(0, 5, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Random brightness/contrast variation (simulates camera conditions)
    from PIL import ImageEnhance
    img = ImageEnhance.Brightness(img).enhance(0.85 + rng.random() * 0.3)
    img = ImageEnhance.Contrast(img).enhance(0.9 + rng.random() * 0.2)

    return img


def hsv_to_rgb(h, s, v):
    """Convert HSV to RGB (h in 0-1, s in 0-1, v in 0-1)."""
    if s == 0:
        return v, v, v
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i %= 6
    if i == 0: return v, t, p
    if i == 1: return q, v, p
    if i == 2: return p, v, t
    if i == 3: return p, q, v
    if i == 4: return t, p, v
    return v, p, q


def print_dataset_statistics():
    """Print statistics about the processed dataset."""
    stats = {}
    for split in ["train", "val", "test"]:
        stats[split] = {}
        for cls in CLASSES:
            cls_dir = PROCESSED_DIR / split / cls
            count = len(list(cls_dir.glob("*.jpg"))) + len(list(cls_dir.glob("*.png")))
            stats[split][cls] = count

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    header = f"{'Class':<12} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}"
    print(header)
    print("-" * 60)
    for cls in CLASSES:
        train = stats["train"][cls]
        val = stats["val"][cls]
        test = stats["test"][cls]
        total = train + val + test
        print(f"{cls:<12} {train:>8} {val:>8} {test:>8} {total:>8}")
    print("-" * 60)
    totals = {s: sum(stats[s].values()) for s in ["train", "val", "test"]}
    print(f"{'TOTAL':<12} {totals['train']:>8} {totals['val']:>8} {totals['test']:>8} {sum(totals.values()):>8}")
    print("=" * 60)

    # Save stats
    with open(PROCESSED_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


# ============================================================
# WHERE TO GET REAL DATA (replace synthetic with these)
# ============================================================
REAL_DATA_SOURCES = """
============================================================
REAL DATASET SOURCES — Replace synthetic data with these
============================================================

1. KAGGLE DATASETS:
   - Search: "chilli pepper quality", "chili classification", "pepper grading"
   - https://www.kaggle.com/datasets (search for chilli/pepper datasets)
   - Example: "Chilli Quality Dataset", "Pepper Grade Classification"

2. COLLECT YOUR OWN (RECOMMENDED for production):
   Steps:
   a) Set up a controlled lighting station:
      - White LED panel (5000K color temperature)
      - Black background board
      - Camera mount at 30cm height, fixed angle
      - Color calibration card in frame for first 10 shots

   b) From your sorting line:
      - Capture 500+ images per grade from actual production
      - Include variety: Guntur, Byadgi, Kashmiri, etc.
      - Vary conditions: different times of day, different batches
      - Capture edge cases deliberately (mixed grades, odd shapes)

   c) Naming convention:
      Grade_A_001.jpg, Grade_A_002.jpg, ...
      Grade_B_001.jpg, ...
      Reject_001.jpg, ...

3. DATA AUGMENTATION will multiply your effective dataset:
   - The training script below applies 15+ augmentations per image
   - 500 real images → ~7500+ augmented training samples per class

4. MINIMUM VIABLE DATASET:
   - Development: 200 images/class (synthetic OK)
   - Pilot:       500 images/class (real required)
   - Production:  2000+ images/class (real, diverse conditions)
============================================================
"""


def main():
    print("=" * 60)
    print("ML MIRCHI — Dataset Preparation")
    print("=" * 60)

    create_directory_structure()

    # Check if real data exists in raw/
    real_data_found = split_existing_dataset()

    if not real_data_found:
        print("\n[INFO] No real dataset found. Generating synthetic data for pipeline development.")
        print("[INFO] Replace synthetic data with real chilli images before production training.\n")
        print(REAL_DATA_SOURCES)
        generate_synthetic_dataset(images_per_class=500)
        split_existing_dataset()

    print_dataset_statistics()
    print("\n[DONE] Dataset is ready for training. Run: python train.py")


if __name__ == "__main__":
    main()