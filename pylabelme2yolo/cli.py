import typer
from pathlib import Path
from typing import Optional, Literal
from pylabelme2yolo.converter import convert_labelme_to_yolo

app = typer.Typer(help="Convert LabelMe annotations to YOLO format")

@app.command()
def main(
    json_dir: Path = typer.Option(
        ...,
        "--json_dir", "-d",
        help="Directory containing LabelMe JSON files and images",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    val_size: float = typer.Option(
        0.2,
        "--val_size",
        help="Validation set size ratio (0-1)",
        min=0.0,
        max=1.0,
    ),
    test_size: float = typer.Option(
        0.0,
        "--test_size",
        help="Test set size ratio (0-1)",
        min=0.0,
        max=1.0,
    ),
    output_format: str = typer.Option(
        "bbox",
        "--output_format", "--format",
        help="Output format (polygon or bbox)",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed for dataset splitting",
    ),
    version: bool = typer.Option(
        False,
        "--version", "-V",
        help="Print version and exit",
        callback=lambda _: typer.echo("pylabelme2yolo v0.1.0") and typer.Exit(),
    ),
):
    if output_format not in ["polygon", "bbox"]:
        raise typer.BadParameter("Invalid format. Choose 'polygon' or 'bbox'")

    """
    Convert LabelMe annotations to YOLO format and organize dataset structure.
    """
    if val_size + test_size >= 1.0:
        typer.echo("Error: val_size + test_size must be less than 1.0", err=True)
        raise typer.Exit(code=1)
    
    convert_labelme_to_yolo(
        json_dir=json_dir,
        val_size=val_size,
        test_size=test_size,
        output_format=output_format,
        seed=seed,
    )

if __name__ == "__main__":
    app()