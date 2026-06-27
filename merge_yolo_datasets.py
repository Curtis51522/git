# Merge YOLO datasets with class name normalization
# Run from: bakery-ai-system/

import os, sys, shutil
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "merged_yolo"

# --- Class name mapping: old_name → new_name (snake_case, lowercase) ---
CLASS_MAP = {
    # Bakery Nude
    "EggTart": "eggtart",
    "cream horn": "cream_horn",
    "croissant": "croissant",
    "doughnut": "donut",
    "melon_bread": "melon_bread",
    "pizza_bread": "pizza_bread",
    "soboru_bread": "soboru_bread",
    # bakery v1
    "chocoPie": "chocopie",
    "pizzaBread": "pizza_bread",
    "stickBread": "stickbread",
    # Bread Detector
    "baguette": "baguette",
    "pandesal": "pandesal",
    "sourdough": "sourdough",
    # Bread Classification
    "soboro": "soboru_bread",
    "pizzabread": "pizza_bread",
    # Others (keep what we want, skip the rest)
    "bagel": "bagel",  # extra, keep if we want
    "donut": "donut",
}

# Classes to include (skip Philippino-only types and generic breads)
SKIP_CLASSES = {
    "binangkal", "bonete", "cornbread", "ensaymada", "flatbread",
    "kalihim", "monay", "spanish-bread", "wheat-bread", "white-bread",
    "whole-grain-bread", "bagel",
}

# Datasets to merge
DATASET_DIRS = [
    "Bakery Nude.v1i.yolov11",
    "bakery.v1i.yolov11",
    "Bread Detector.v2i.yolov11",
    "Bread_Classification.v7i.yolov11",
]

def remap_labels(label_dir, old_to_new_idx, class_mapping):
    """Rewrite YOLO label files with new class IDs."""
    for txt_file in Path(label_dir).glob("*.txt"):
        lines = []
        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                old_id = int(parts[0])
                if old_id in old_to_new_idx:
                    new_id = old_to_new_idx[old_id]
                    parts[0] = str(new_id)
                    lines.append(" ".join(parts) + "\n")
        with open(txt_file, "w") as f:
            f.writelines(lines)

def main():
    # Step 1: Build global class list
    all_names = set()
    for ds_dir in DATASET_DIRS:
        yaml_path = DATA_DIR / ds_dir / "data.yaml"
        if not yaml_path.exists():
            continue
        # Parse yaml simply
        with open(yaml_path) as f:
            content = f.read()
        # Extract names
        import re
        match = re.search(r"names:\s*\[(.*?)\]", content, re.DOTALL)
        if match:
            names_str = match.group(1)
            names = [n.strip().strip("'\"") for n in names_str.split(",")]
            all_names.update(names)

    # Build final class list
    final_classes = []
    old_to_new = {}  # (dataset_name, old_name) → new_name
    seen_new = {}    # new_name → index

    for old_name in sorted(all_names):
        new_name = CLASS_MAP.get(old_name, old_name.lower().replace(" ", "_"))
        if new_name in SKIP_CLASSES:
            continue
        if new_name not in seen_new:
            seen_new[new_name] = len(final_classes)
            final_classes.append(new_name)
        old_to_new[old_name] = (new_name, seen_new[new_name])

    print(f"Final class list ({len(final_classes)}):")
    for i, c in enumerate(final_classes):
        print(f"  {i}: {c}")

    # Step 2: Create output directory structure
    for split in ["train", "valid"]:
        (OUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Step 3: Copy images + remap labels
    stats = defaultdict(lambda: defaultdict(int))  # {class: {train: N, valid: N}}
    img_counter = 0

    for ds_dir_name in DATASET_DIRS:
        ds_path = DATA_DIR / ds_dir_name
        if not ds_path.exists():
            continue

        # Parse data.yaml for this dataset
        yaml_path = ds_path / "data.yaml"
        with open(yaml_path) as f:
            yaml_text = f.read()

        # Get original class list for this dataset
        import re
        match = re.search(r"names:\s*\[(.*?)\]", yaml_text, re.DOTALL)
        if not match:
            continue
        names_str = match.group(1)
        ds_names = [n.strip().strip("'\"") for n in names_str.split(",")]

        # Build old_id → new_id mapping
        old_to_new_idx = {}
        for old_id, old_name in enumerate(ds_names):
            new_name = CLASS_MAP.get(old_name, old_name.lower().replace(" ", "_"))
            if new_name in SKIP_CLASSES:
                continue
            if new_name in seen_new:
                old_to_new_idx[old_id] = seen_new[new_name]

        # Process train and valid splits
        for split in ["train", "valid", "val"]:
            src_img = ds_path / split / "images"
            src_lbl = ds_path / split / "labels"
            if not src_img.exists():
                # Try the flat structure (Bread_Classification)
                src_img = ds_path / "images"
                src_lbl = ds_path / "labels"

            if not src_img.exists():
                continue

            dst_split = "train" if split in ["train", "valid"] else "valid"
            if split == "val":
                dst_split = "valid"

            # Copy images
            for img_file in src_img.iterdir():
                if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue

                # Unique name to avoid collisions
                new_name_img = f"{ds_dir_name[:10]}_{img_counter:06d}{img_file.suffix}"
                dst_img = OUT_DIR / dst_split / "images" / new_name_img
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_file, dst_img)

                # Copy and remap label
                label_file = src_lbl / (img_file.stem + ".txt")
                if label_file.exists():
                    dst_lbl = OUT_DIR / dst_split / "labels" / (Path(new_name_img).stem + ".txt")
                    new_lines = []
                    with open(label_file) as f:
                        for line in f:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            old_id = int(parts[0])
                            if old_id in old_to_new_idx:
                                parts[0] = str(old_to_new_idx[old_id])
                                new_lines.append(" ".join(parts) + "\n")
                    with open(dst_lbl, "w") as f:
                        f.writelines(new_lines)

                img_counter += 1

    # Step 4: Write data.yaml
    yaml_content = f"""# Merged bakery detection dataset
# Auto-generated by merge_yolo_datasets.py

path: .
train: train/images
val: valid/images

nc: {len(final_classes)}
names: {final_classes}
"""
    with open(OUT_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)

    # Stats
    print(f"\nTotal images: {img_counter}")
    print(f"Train dir: {(OUT_DIR / 'train' / 'images').absolute()}")
    print(f"Valid dir: {(OUT_DIR / 'valid' / 'images').absolute()}")

if __name__ == "__main__":
    main()
