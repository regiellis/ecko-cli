import pytest
from typer.testing import CliRunner
from ecko_cli.cli import ecko_cli
import os
from pathlib import Path
import shutil
from PIL import Image
import numpy as np
import json
import warnings

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(scope="function")
def temp_dir(tmpdir):
    test_dir = tmpdir.mkdir("test_images")

    # Create valid image files
    for i in range(3):
        img_array = np.random.rand(100, 100, 3) * 255
        img = Image.fromarray(img_array.astype("uint8"))
        img.save(os.path.join(test_dir, f"image_{i}.jpg"))

    yield str(test_dir)

    shutil.rmtree(str(test_dir))


@pytest.mark.parametrize(
    "command, expected_exit_code, expected_output",
    [
        (["process-images", "{temp_dir}", "test"], 0, "Processing images..."),
        (
            ["process-images", "{temp_dir}", "test", "--padding", "4"],
            0,
            "Processing images...",
        ),
        (
            ["process-images", "{temp_dir}", "test", "--trigger", "watercolor"],
            0,
            "Processing images...",
        ),
        (
            ["process-images", "{temp_dir}", "test", "--is-anime"],
            0,
            "Processing images...",
        ),
        (
            ["process-images", "{temp_dir}", "test", "--is-object"],
            0,
            "Processing images...",
        ),
        (["create-jsonl", "{temp_dir}", "dataset"], 0, "JSONL file created:"),
    ],
)
def test_commands(runner, temp_dir, command, expected_exit_code, expected_output):
    command = [
        arg.format(temp_dir=temp_dir) if isinstance(arg, str) else arg
        for arg in command
    ]
    result = runner.invoke(ecko_cli, command, catch_exceptions=False)
    print(f"Command: {command}")
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.output}")
    assert result.exit_code == expected_exit_code
    if expected_output:
        assert expected_output in result.output


def test_invalid_command(runner):
    result = runner.invoke(ecko_cli, ["nonexistent-command"], catch_exceptions=False)
    assert result.exit_code == 2
    assert "No such command 'nonexistent-command'." in result.stdout


def test_process_images_output(runner, temp_dir):
    # Set up test parameters
    name = "test"
    padding = 4
    sizes = [
        512,
        1024,
        672,
    ]  # Include 672 as it's used in the process_directory function
    num_images = 3

    # Create test images
    for i in range(1, num_images + 1):
        # Create original image
        img_array = np.random.rand(100, 100, 3) * 255
        img = Image.fromarray(img_array.astype("uint8"))
        original_filename = (
            f"image_{i-1:04d}.jpg"  # Match the naming in process_directory
        )
        img.save(os.path.join(temp_dir, original_filename))

    # Run the process-images command
    result = runner.invoke(
        ecko_cli, ["process-images", temp_dir, name], catch_exceptions=False
    )
    print(f"Exit code: {result.exit_code}")
    print(f"Output:\n{result.output}")

    # List all files in the output directory
    print("Files in output directory:")
    for file in os.listdir(temp_dir):
        print(f"  {file}")

    assert result.exit_code == 0, f"Command failed with exit code {result.exit_code}"

    # Check for expected output files
    for i in range(1, num_images + 1):
        base_name = f"{name}_{i:0{padding}d}"

        # Check for resized images
        for size in sizes:
            expected_file = os.path.join(temp_dir, f"{base_name}_{size}.jpg")
            if size == 672:
                assert not os.path.exists(
                    expected_file
                ), f"Unexpected 672 file found: {expected_file}"
            else:
                assert os.path.exists(
                    expected_file
                ), f"Expected file not found: {expected_file}"

        # Check for caption file
        expected_txt = os.path.join(temp_dir, f"{base_name}.txt")
        assert os.path.exists(
            expected_txt
        ), f"Expected text file not found: {expected_txt}"

    # Check for JSONL file
    jsonl_file = os.path.join(temp_dir, "dataset.jsonl")
    assert os.path.exists(jsonl_file), f"Expected JSONL file not found: {jsonl_file}"

    # Print the final contents of the output directory
    print("\nFinal contents of output directory:")
    for file in os.listdir(temp_dir):
        print(f"  {file}")
