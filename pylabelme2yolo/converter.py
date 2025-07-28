import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Literal
import numpy as np
from PIL import Image
from tqdm import tqdm
import collections

def read_classes(json_dir: Path) -> Dict[int, str]:
    """Read classes.txt file and return a dictionary mapping class IDs to names."""
    classes_file = json_dir / "classes.txt"
    if not classes_file.exists():
        raise FileNotFoundError(f"classes.txt not found in {json_dir}")
    
    classes = {}
    with open(classes_file, "r") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                class_id, class_name = line.split(":", 1)
                classes[int(class_id.strip())] = class_name.strip()
            else:
                class_id, class_name = line.split(maxsplit=1)
                classes[int(class_id)] = class_name.strip()
    return classes

def validate_files(json_dir: Path) -> List[Tuple[Path, Path]]:
    """Validate that each JSON file has a corresponding image file."""
    pairs = []
    for json_file in json_dir.glob("*.json"):
        image_file = json_dir / f"{json_file.stem}.jpg"
        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found for {json_file}")
        pairs.append((json_file, image_file))
    return pairs

def convert_labelme_to_yolo(
    json_dir: Path,
    val_size: float = 0.2,
    test_size: float = 0.0,
    output_format: Literal["polygon", "bbox"] = "bbox",
    seed: int = 42,
):
    """Convert LabelMe annotations to YOLO format and organize dataset structure."""
    random.seed(seed)
    
    # Read classes
    classes = read_classes(json_dir)
    class_names = {v: k for k, v in classes.items()}  # Reverse mapping
    
    # Validate files
    file_pairs = validate_files(json_dir)

    # Group by main label
    class_to_files = collections.defaultdict(list)
    for json_file, image_file in file_pairs:
        with open(json_file, "r") as f:
            data = json.load(f)
        if not data["shapes"]:
            continue
        main_label = data["shapes"][0]["label"]
        class_to_files[main_label].append((json_file, image_file))

    train_files, val_files, test_files = [], [], []
    for files in class_to_files.values():
        random.shuffle(files)
        n = len(files)
        n_val = int(n * val_size)
        n_test = int(n * test_size)
        n_train = n - n_val - n_test
        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:n_train + n_val])
        test_files.extend(files[n_train + n_val:])

    # Shuffle to mix classes
    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)
    
    # Create output directories
    output_dir = json_dir / "data"
    (output_dir / "images/train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images/val").mkdir(parents=True, exist_ok=True)
    (output_dir / "images/test").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels/train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels/val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels/test").mkdir(parents=True, exist_ok=True)

    # Helper to process and save
    def process_split(split_files, split):
        for json_file, image_file in tqdm(split_files, desc=f"Processing {split}"):
            dest_image = output_dir / f"images/{split}/{image_file.name}"
            shutil.copy2(image_file, dest_image)
            with open(json_file, "r") as f:
                data = json.load(f)
            img_width = data["imageWidth"]
            img_height = data["imageHeight"]
            yolo_lines = []
            for shape in data["shapes"]:
                label = shape["label"]
                points = np.array(shape["points"])
                if output_format == "bbox":
                    x_coords = points[:, 0]
                    y_coords = points[:, 1]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    x_center = ((x_min + x_max) / 2) / img_width
                    y_center = ((y_min + y_max) / 2) / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height
                    yolo_line = f"{class_names[label]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    yolo_lines.append(yolo_line)
                else:
                    normalized_points = []
                    for x, y in points:
                        normalized_points.append(f"{x / img_width:.6f}")
                        normalized_points.append(f"{y / img_height:.6f}")
                    yolo_line = f"{class_names[label]} {' '.join(normalized_points)}"
                    yolo_lines.append(yolo_line)
            dest_label = output_dir / f"labels/{split}/{json_file.stem}.txt"
            with open(dest_label, "w") as f:
                f.write("\n".join(yolo_lines))

    process_split(train_files, "train")
    process_split(val_files, "val")
    process_split(test_files, "test")
    generate_dataset_yaml(output_dir, classes)


def generate_dataset_yaml(output_dir: Path, classes: Dict[int, str]):
    """Generate YOLO dataset description file."""
    yaml_content = f"""path: {output_dir.absolute()}
train: images/train
val: images/val
test: images/test

names:
"""
    for class_id, class_name in sorted(classes.items()):
        yaml_content += f"    {class_id}: {class_name}\n"
    
    with open(output_dir / "dataset.yaml", "w") as f:
        f.write(yaml_content)