# pylabelme2yolo
convert labelme to yolo segment format with stable class id

## features
1. class id will be dependes on the classes.txt file, and it never change in multiple run
2. generate yolo segment data folder in `{json_dir}\data`

## Installation & Usage
```shell
pip install pylabelme2yolo
```

```shell
pylabelme2yolo --output_format polygon --json_dir images
```

