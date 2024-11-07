from . import __version__

import os
import json
import csv
import typer
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any
import subprocess

from .helpers import (
    create_output_directory,
    generate_caption_file,
    create_table,
    feedback_message,
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
multiple sizes for better captioning results. [1024, 672, 512]. The
WD14 model is used for captioning all images based on a modified version of
the selected tags it was trained on.


Usage:
$ pipx install ecko-cli (recommended)
$ pipx install . (if you want to install it globally)
$ pip install -e . (if you want to install it locally and poke around,
make sure to create a virtual environment)
$ ecko [OPTIONS] [COMMAND] [ARGS]

Options:
    process-images DIRECTORY BATCH_IMAGE_NAME [--trigger WORD] [--is_object] [--padding PADDING]
    Process images in a directory and generate captions.
    
    create-jsonl DIRECTORY OUTPUT_NAME                 Create a JSONL file from images in a directory.
    --help                                             Show this message and exit.

Examples:

$ ecko process-images /path/to/images watercolors --padding 4
$ ecko process-images /path/to/images doors --is_object True
$ ecko create-jsonl /path/to/images [dataset]
$ ecko ui /path/to/images

"""

__all_ = ["ecko_cli"]

__version__ = __version__

ecko_cli = typer.Typer()
tools_cli = typer.Typer()
console = Console()

DEFAULT_PADDING = os.getenv("DEFAULT_PADDING", 4)

ecko_cli.add_typer(tools_cli, name="tools", help="Tools for Ecko CLI")


class PlaylogicTheme(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.yellow,
        secondary_hue: colors.Color | str = colors.yellow,
        neutral_hue: colors.Color | str = colors.zinc,
        spacing_size: sizes.Size | str = sizes.spacing_lg,
        radius_size: sizes.Size | str = sizes.radius_none,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font | str | list[fonts.Font | str] = (
            fonts.GoogleFont("Inter"),
            "ui-sans-serif",
            "system-ui",
            "sans-serif",
        ),
        font_mono: fonts.Font | str | list[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "Consolas",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )

        self.name = "playlogic_theme"

        self.set(
            body_background_fill="*neutral_50",
            body_background_fill_dark="*neutral_900",
            body_text_color="*neutral_900",
            body_text_color_dark="*neutral_100",
            background_fill_primary="white",
            background_fill_primary_dark="*neutral_800",
            background_fill_secondary="*neutral_100",
            background_fill_secondary_dark="*neutral_700",
            border_color_primary="*neutral_200",
            border_color_primary_dark="*neutral_700",
            block_title_text_weight="600",
            block_border_width="1px",
            block_shadow="0 1px 2px 0 rgba(0, 0, 0, 0.05)",
            button_primary_background_fill="*primary_500",
            button_primary_background_fill_hover="*primary_600",
            button_primary_text_color="white",
            button_secondary_background_fill="white",
            button_secondary_background_fill_hover="*neutral_100",
            button_secondary_text_color="*neutral_800",
            button_secondary_border_color="*neutral_200",
            # button_shadow="0 1px 2px 0 rgba(0, 0, 0, 0.05)",
            block_label_background_fill="*neutral_50",
            block_label_background_fill_dark="*neutral_800",
            input_background_fill="white",
            input_background_fill_dark="stone",
            input_border_color="*primary_400",
            input_border_color_dark="*primary_400",
            input_shadow="0 1px 2px 0 rgba(0, 0, 0, 0.05)",
            slider_color="*primary_500",
            slider_color_dark="*primary_400",
        )


playlogic = PlaylogicTheme()


def create_dataset_from_images(
    directory: str, output_name: str, format: str, relative_paths: bool = False
) -> None:
    """
    Create a dataset file from images in a directory, reading captions from corresponding text files.

    Args:
    directory (str): Path to the directory containing images and caption text files.
    output_name (str): Name for the output file (without extension).
    format (str): Output format, either "jsonl", "hf_json", "csv", or "json".
    relative_paths (bool): Whether to use relative paths for images in the output.

    Returns:
    None
    """
    # Ensure the directory path is absolute
    directory = os.path.abspath(directory)

    # List of common image file extensions
    image_extensions: List[str] = [".jpg", ".jpeg", ".png", ".gif"]

    output_filename = f"{output_name}.{format if format != 'hf_json' else 'json'}"
    output_path = os.path.join(directory, output_filename)

    dataset: List[Dict[str, Any]] = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Check if the file is an image
        if (
            os.path.isfile(file_path)
            and Path(file_path).suffix.lower() in image_extensions
        ):
            text_file_path = os.path.splitext(file_path)[0] + ".txt"

            if os.path.exists(text_file_path):
                with open(text_file_path, "r") as text_file:
                    caption = text_file.read().strip()
            else:
                caption = "No caption available"

            image_data: Dict[str, Any] = {
                "image": (
                    os.path.relpath(file_path, directory)
                    if relative_paths
                    else file_path
                ),
                "text": caption,
            }

            if format == "jsonl":
                image_data["mask_path"] = None

            dataset.append(image_data)

    with open(output_path, "w", newline="") as output_file:
        if format == "jsonl":
            for item in dataset:
                json.dump(item, output_file)
                output_file.write("\n")
        elif format == "hf_json":
            json.dump(
                {
                    "data": dataset,
                    "meta": {
                        "description": "Image dataset created from local directory",
                        "format": "image",
                    },
                },
                output_file,
                indent=2,
            )
        elif format == "json":
            json.dump(dataset, output_file, indent=2)
        elif format == "csv":
            writer = csv.DictWriter(output_file, fieldnames=["image", "text"])
            writer.writeheader()
            writer.writerows(dataset)

    print(f"Dataset file created: {output_path}")


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


def serve(dataset_path):
    training_data_path = os.path.join(dataset_path, "training_data")
    dataset_file = os.path.join(training_data_path, "dataset.json")

    if not os.path.exists(dataset_file):
        feedback_message("Dataset not found", "error")
        raise typer.Exit()

    with open(dataset_file, "r") as f:
        dataset = json.load(f)

    image_count = len(dataset)

    def load_image_and_caption(index):
        if index is None or not (0 <= index < image_count):
            return None, "No image selected", None
        item = dataset[index]
        image_path = os.path.join(training_data_path, item["image"])
        return Image.open(image_path), item["text"], index

    def update_caption(index, new_caption):
        if index is None or not (0 <= index < image_count):
            return "No image selected for update"

        dataset[index]["text"] = new_caption

        # Update all dataset files
        for ext in ["json", "jsonl", "csv"]:
            file_path = os.path.join(training_data_path, f"dataset.{ext}")
            if os.path.exists(file_path):
                if ext == "json":
                    with open(file_path, "w") as f:
                        json.dump(dataset, f, indent=2)
                elif ext == "jsonl":
                    with open(file_path, "w") as f:
                        for item in dataset:
                            f.write(json.dumps(item) + "\n")
                elif ext == "csv":
                    with open(file_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=["image", "text"])
                        writer.writeheader()
                        writer.writerows(dataset)

        # Update the corresponding .txt file
        image_filename = dataset[index]["image"]
        txt_filename = os.path.splitext(image_filename)[0] + ".txt"
        txt_path = os.path.join(training_data_path, txt_filename)
        with open(txt_path, "w") as f:
            f.write(new_caption)

        return f"Caption updated for image {txt_filename[:-4]}"

    def get_image_paths():
        return [os.path.join(training_data_path, item["image"]) for item in dataset]

    with gr.Blocks(theme=playlogic) as demo:
        gr.Markdown("## ECKO Editor")

        with gr.Row():
            with gr.Column(scale=2):
                image_output = gr.Image(type="pil", label="Selected Image")
                caption_input = gr.Textbox(label="Caption", lines=4)
                update_button = gr.Button("Update Caption")
                update_status = gr.Markdown("### Update status will appear here")

        with gr.Row():
            with gr.Column(scale=1):
                gallery = gr.Gallery(
                    value=get_image_paths(),
                    columns=8,
                    rows=4,
                    label="Image Gallery",
                    allow_preview=False,
                )
                gr.Markdown(f"### Total Images: {image_count}")

        selected_index = gr.State(None)

        def select_image(evt: gr.SelectData):
            return evt.index

        gallery.select(select_image, outputs=[selected_index])
        selected_index.change(
            load_image_and_caption,
            inputs=[selected_index],
            outputs=[image_output, caption_input, selected_index],
        )

        update_button.click(
            update_caption,
            inputs=[selected_index, caption_input],
            outputs=[update_status],
        )

    demo.launch(server_name="0.0.0.0")


@ecko_cli.command()
def ui(
    dataset_path: str = typer.Argument(
        ..., help="Path to the directory containing dataset.json and images"
    )
):
    """Serve a Gradio interface for viewing image captions"""
    serve(dataset_path)


@ecko_cli.command(
    "create-dataset",
    help="Create a dataset file from images in a directory.",
    no_args_is_help=True,
)
def create_dataset(
    directory: str = typer.Argument(
        ..., help="Path to the directory containing images."
    ),
    output_name: str = typer.Argument(
        "dataset", help="Name for the output dataset file."
    ),
    format: str = typer.Option(
        "jsonl", help="Output format: jsonl, hf_json, csv, or json."
    ),
    relative_paths: bool = typer.Option(
        False, help="Use relative paths for images in the output."
    ),
):
    """
    Create a dataset file from images in a directory, reading captions from corresponding text files.

    Args:
        directory (str): Path to the directory containing images and caption text files.
        output_name (str): Name for the output dataset file (without extension).
        format (str): Output format, either "jsonl", "hf_json", "csv", or "json".
        relative_paths (bool): Whether to use relative paths for images in the output.
    """
    valid_formats = ["jsonl", "hf_json", "csv", "json"]
    if format not in valid_formats:
        raise ValueError(f"Invalid format. Choose from: {', '.join(valid_formats)}")

    create_dataset_from_images(directory, output_name, format, relative_paths)


@ecko_cli.command(
    "process-images",
    help="Process images in a directory and generate captions.",
    no_args_is_help=True,
)
def process_directory(
    directory_path: str,
    name: str,
    trigger: str = None,
    padding: int = DEFAULT_PADDING,
    use_joycap: bool = False,
    is_anime: bool = False,
    is_object: bool = False,
    is_style: bool = False,
) -> None:
    """
    Process all images in a directory, generate captions, and save them as text files.

    Args:
        directory_path (str): Path to the directory containing images.
        name (str): Base name for output files.
        is_object (bool): Captions based on object detection.
        is_anime (bool): Captions based on anime detection.
        is_style (bool): Captions based on style detection.
        padding (int): Number of digits for output file numbering.
    """

    # print(f"Processing directory: {directory_path}")
    # output_dir = create_output_directory(directory_path)
    # print(f"Output directory: {output_dir}")

    if use_joycap:
        feedback_message(
            """Joycap is currently in Alpha and may not work as expected. JoyCap also uses
               a ton of vram and may crash your system. Please use with caution.""",
            "warning",
        )

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
            caption = analyze_image(
                caption_image,
                task,
                progress,
                trigger,
                use_joycap,
                is_anime,
                is_object,
                is_style,
            )
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

    datasets = ["jsonl", "json"]
    for dataset in datasets:
        create_dataset_from_images(output_dir, "dataset", dataset)

    display_results_table(results)


@tools_cli.command()
def install_flash_attention():
    """
    Install flash-attention package
    """
    subprocess.run(
        "pip install flash-attn --no-build-isolation",
        shell=True,
    )
    
@tools_cli.command()
def install_models():
    """
    Install required models
    """
    subprocess.run(
        "hf download MiaoshouAI/Florence-2-base-PromptGen-v2.0",
        shell=True,
    )
