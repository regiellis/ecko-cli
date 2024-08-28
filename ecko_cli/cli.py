from . import __version__

import os
import json
import typer
from pathlib import Path
from typing import List, Dict, Any

from .helpers import (
    create_output_directory,
    generate_caption_file,
    create_table,
)

from .images import ImageProcessor
from .analyze import analyze_image


from rich.progress import Progress
from rich.console import Console

from rich.traceback import install

install()

"""
====================================================================
Ecko CLI - Simplified Image Processing and Caption Generation Tool
====================================================================

Ecko is a simple CLI tool that streamlines the process of processing
images in a directory, generating captions, and saving them as text files.
Additionally, it provides functionalities to create a JSONL file from images
in the directory you specify. Images will be captioned using the Microsoft
Florence-2-large model and the ONNX Runtime engine. Images are resized to
multiple sizes for better captioning results. [1024, 768, 672, 512]. The
WD14 model is used for captioning all images based on a modified version of
the selected tags it was trained on.


Usage:
$ pipx install ecko-cli (recommended)
$ pipx install . (if you want to install it globally)
$ pip install -e . (if you want to install it locally and poke around, 
make sure to create a virtual environment)
$ ecko-cli [OPTIONS] [COMMAND] [ARGS]

Options:
    process-images DIRECTORY BATCH_IMAGE_NAME [--trigger WORD] [--is_object True/False] [--padding PADDING]  
    Process images in a directory and generate captions.
    
    create-jsonl DIRECTORY OUTPUT_NAME                 Create a JSONL file from images in a directory.
    --help                                             Show this message and exit.

Examples:

$ ecko-cli process-images /path/to/images watercolors --padding 4
$ ecko-cli process-images /path/to/images doors --is_object True
$ ecko-cli create-jsonl /path/to/images [dataset]

"""

__all_ = ["ecko_cli"]

__version__ = __version__

ecko_cli = typer.Typer()
console = Console()

DEFAULT_PADDING = os.getenv("DEFAULT_PADDING", 4)


def create_jsonl_from_images(directory: str, output_name: str = "dataset") -> None:
    """
    Create a JSONL file from images in a directory, reading captions from corresponding text files.

    Args:
    directory (str): Path to the directory containing images and caption text files.
    output_name (str): Name for the output JSONL file (without extension).

    Returns:
    None
    """
    # Ensure the directory path is absolute
    directory = os.path.abspath(directory)

    # List of common image file extensions
    image_extensions: List[str] = [".jpg", ".jpeg", ".png", ".gif"]

    # Construct the full path for the output JSONL file
    jsonl_filename = f"{output_name}.jsonl"
    jsonl_path = os.path.join(directory, jsonl_filename)

    # Open the JSONL file for writing
    with open(jsonl_path, "w") as jsonl_file:
        # Iterate through all files in the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)

            # Check if the file is an image
            if (
                os.path.isfile(file_path)
                and Path(file_path).suffix.lower() in image_extensions
            ):
                # Construct the path for the corresponding text file
                text_file_path = os.path.splitext(file_path)[0] + ".txt"

                # Read the caption from the text file if it exists
                if os.path.exists(text_file_path):
                    with open(text_file_path, "r") as text_file:
                        caption = text_file.read().strip()
                else:
                    caption = "No caption available"

                # Create a dictionary with the required information
                image_data: Dict[str, Any] = {
                    "image": file_path,
                    "text": caption,
                    "mask_path": None,
                }

                # Write the JSON object to the file, followed by a newline
                json.dump(image_data, jsonl_file)
                jsonl_file.write("\n")

    print(f"JSONL file created: {jsonl_path}")


def display_results_table(results: List[Dict[str, str]]) -> None:
    """
    Display a table of processing results using Rich.

    Args:
        results (List[Dict[str, str]]): List of dictionaries containing processing results.
    """
    results_table = create_table(
        title="Image Processing Results",
        columns=[("Filename", "cyan"), ("Path", "magenta"), ("Status", "green")],
    )
    for result in results:
        status_style = "green" if result["status"] == "Success" else "red"
        results_table.add_row(
            result["filename"],
            result["path"],
            f"[{status_style}]{result['status']}[/{status_style}]",
        )

    console.print(results_table)


@ecko_cli.command(
    "create-jsonl", help="Create a JSONL file from images in a directory."
)
def create_jsonl(
    directory: str = typer.Argument(
        ..., help="Path to the directory containing images."
    ),
    output_name: str = typer.Argument(..., help="Name for the output JSONL file."),
):
    """
    Create a JSONL file from images in a directory, reading captions from corresponding text files.

    Args:
        directory (str): Path to the directory containing images and caption text files.
        output_name (str): Name for the output JSONL file (without extension).
    """
    create_jsonl_from_images(directory, output_name)


@ecko_cli.command(
    "process-images", help="Process images in a directory and generate captions."
)
def process_directory(
    directory_path: str,
    name: str,
    trigger: str = None,
    padding: int = DEFAULT_PADDING,
    is_object: bool = False,
) -> None:
    """
    Process all images in a directory, generate captions, and save them as text files.

    Args:
        directory_path (str): Path to the directory containing images.
        name (str): Base name for output files.
        is_object (bool): Whether the images contain objects or scenes.
        padding (int): Number of digits for output file numbering.
    """
    output_dir = create_output_directory(directory_path)
    image_files = [
        f
        for f in os.listdir(directory_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
    ]

    results: List[Dict[str, str]] = []

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing images...", total=len(image_files))

        for index, filename in enumerate(image_files, start=1):
            input_path = os.path.join(directory_path, filename)
            output_filename = f"{name}_{index:0{padding}d}{Path(filename).suffix}"
            output_image_path = os.path.join(output_dir, output_filename)
            output_caption_path = os.path.join(
                output_dir, f"{name}_{index:0{padding}d}.txt"
            )

            # Process and save the image
            image_sizes = [1024, 672]
            images = ImageProcessor()
            images.process_image(image_sizes, input_path, output_image_path, output_dir)

            # Generate and save the caption
            caption_image = f"{output_dir}/{name}_{index:0{padding}d}_{image_sizes[1]}{Path(filename).suffix}"
            caption = analyze_image(caption_image, task, progress, trigger, is_object)
            if caption:
                if generate_caption_file(output_caption_path, caption):
                    results.append(
                        {
                            "filename": output_filename,
                            "path": output_image_path,
                            "status": "Success",
                        }
                    )
                else:
                    results.append(
                        {
                            "filename": output_filename,
                            "path": output_image_path,
                            "status": "Failed to save caption",
                        }
                    )
            else:
                results.append(
                    {
                        "filename": output_filename,
                        "path": output_image_path,
                        "status": "Failed to generate caption",
                    }
                )

    display_results_table(results)
