import pytest
from pathlib import Path
from pylabelme2yolo.converter import read_classes, validate_files

def test_read_classes(tmp_path):
    classes_file = tmp_path / "classes.txt"
    classes_file.write_text("0: cat\n1: dog\n2: person")
    classes = read_classes(tmp_path)
    assert classes == {0: "cat", 1: "dog", 2: "person"}

def test_validate_files(tmp_path):
    json_file = tmp_path / "test1.json"
    json_file.write_text("{}")
    image_file = tmp_path / "test1.jpg"
    image_file.write_text("")  # Empty file for testing
    
    pairs = validate_files(tmp_path)
    assert len(pairs) == 1
    assert pairs[0][0].name == "test1.json"
    assert pairs[0][1].name == "test1.jpg"
    
    with pytest.raises(FileNotFoundError):
        (tmp_path / "test1.jpg").unlink()
        validate_files(tmp_path)