import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Literal
import numpy as np
from PIL import Image
from tqdm import tqdm

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
    random.shuffle(file_pairs)
    
    # Calculate split sizes
    total_files = len(file_pairs)
    val_count = int(total_files * val_size)
    test_count = int(total_files * test_size)
    train_count = total_files - val_count - test_count
    
    # Create output directories
    output_dir = json_dir / "data"
    (output_dir / "images/train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images/val").mkdir(parents=True, exist_ok=True)
    (output_dir / "images/test").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels/train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels/val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels/test").mkdir(parents=True, exist_ok=True)
    
    # Process files
    for i, (json_file, image_file) in enumerate(tqdm(file_pairs, desc="Processing files")):
        # Determine split
        if i < train_count:
            split = "train"
        elif i < train_count + val_count:
            split = "val"
        else:
            split = "test"
        
        # Copy image
        dest_image = output_dir / f"images/{split}/{image_file.name}"
        shutil.copy2(image_file, dest_image)
        
        # Convert annotations
        with open(json_file, "r") as f:
            data = json.load(f)
        
        img_width = data["imageWidth"]
        img_height = data["imageHeight"]
        
        yolo_lines = []
        for shape in data["shapes"]:
            label = shape["label"]
            points = np.array(shape["points"])
            
            if output_format == "bbox":
                # Convert to YOLO bbox format
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
                # Convert to YOLO polygon format
                normalized_points = []
                for x, y in points:
                    normalized_points.append(f"{x/img_width:.6f}")
                    normalized_points.append(f"{y/img_height:.6f}")
                
                yolo_line = f"{class_names[label]} {' '.join(normalized_points)}"
                yolo_lines.append(yolo_line)
        
        # Write YOLO annotations
        dest_label = output_dir / f"labels/{split}/{json_file.stem}.txt"
        with open(dest_label, "w") as f:
            f.write("\n".join(yolo_lines))
    
    # Generate dataset.yaml
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