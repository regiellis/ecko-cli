import os
import typer
import errno
import cv2
from pathlib import Path
from importlib import resources


from typing import Dict, Any
from rich.console import Console
from rich.table import Table

console = Console(soft_wrap=True)

__all__ = [
    "feedback_message",
    "create_output_directory",
    "generate_caption_file",
    "create_table",
    "add_rows_to_table",
    "get_models_dir",
    "make_square",
    "smart_resize",
]


def feedback_message(message: str, type: str = "info") -> None:
    options = {
        "types": {
            "info": "white",
            "success": "green",
            "warning": "yellow",
            "error": "red",
            "exception": "red",
        },
        "titles": {
            "info": "Information",
            "success": "Success",
            "warning": "Warning",
            "error": "Error Message",
            "exception": "Exception Message",
        },
    }

    feedback_message_table = Table(style=options["types"][type])
    feedback_message_table.add_column(options["titles"][type])
    feedback_message_table.add_row(message)

    if type == "exception":
        console.print_exception(feedback_message_table)
        raise typer.Exit()
    console.print(feedback_message_table)
    return None


def create_table(title: str, columns: list) -> Table:
    table = Table(title=title, title_justify="left")
    for col_name, style in columns:
        table.add_column(col_name, style=style)
    return table


def add_rows_to_table(table: Table, data: Dict[str, Any]) -> None:
    for key, value in data.items():
        if isinstance(value, list):
            value = ", ".join(map(str, value))
        table.add_row(key, str(value))


def create_output_directory(base_dir: str) -> str:
    output_dir = Path(base_dir) / "training_data"

    # Check if we have read access to the parent directory
    if not os.access(base_dir, os.R_OK):
        console.print(f"[bold red]Error: No read access to {base_dir}[/bold red]")
        raise PermissionError(f"No read access to {base_dir}")

    # Check if we have write access to the parent directory
    if not os.access(base_dir, os.W_OK):
        console.print(f"[bold red]Error: No write access to {base_dir}[/bold red]")
        raise PermissionError(f"No write access to {base_dir}")

    # Try to create the output directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            console.print(
                f"[bold red]Error: Failed to create output directory {output_dir}[/bold red]"
            )
            raise

    # Verify we can write to the output directory
    if not os.access(output_dir, os.W_OK):
        console.print(
            f"[bold red]Error: No write access to output directory {output_dir}[/bold red]"
        )
        raise PermissionError(f"No write access to output directory {output_dir}")

    console.print(f"[green]Output directory created successfully: {output_dir}[/green]")
    return str(output_dir)


def generate_caption_file(caption_path: str, caption: str) -> bool:
    sizes = [1024]
    # Write the caption to the caption file for all images sizes (1024, 768, 512)
    # should be in the format: "caption_path_[size].txt"  - may bring this back
    try:
        for size in sizes:
            caption_file = f"{caption_path[:-4]}_{size}.txt"
            with open(caption_file, "w") as file:
                file.write(caption)
        return True
    except Exception as e:
        feedback_message(
            f"Error writing caption to file: {str(e)}", type="exception"
        )
        return False


def get_models_dir() -> str:
    # Use the package name directly
    package_name = "ecko_cli"

    try:
        # For Python 3.9+
        with resources.files(package_name) as package_path:
            models_dir = os.path.join(package_path, "models")
    except AttributeError:
        # Fallback for earlier Python versions
        package_path = resources.files(package_name)
        models_dir = os.path.join(package_path, "models")

    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found at {models_dir}")

    return models_dir


def make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    return cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )


def smart_resize(img, size):
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img
