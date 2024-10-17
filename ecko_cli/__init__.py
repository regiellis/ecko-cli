import os
import typer
from pathlib import Path
from typing import Dict, List, Final
from dotenv import load_dotenv, set_key

from rich.console import Console
from rich.prompt import Prompt

from .helpers import feedback_message

console = Console()

CAPTION_MODEL: Final[str] = "CAPTION_MODEL"
DEFAULT_ENV_PATH: Final[Path] = Path("~/.config/ecko-cli-itsjustregi/.env")


def get_required_input(prompt: str) -> str:
    while True:
        response = Prompt.ask(prompt).strip()
        if response:
            return response
        feedback_message("This field is required. Please enter a value.", "warning")


def create_directory(dir_path: Path) -> None:
    try:
        dir_path.mkdir(parents=True)
        feedback_message(f"Created directory: {dir_path}", "info")
    except Exception as e:
        feedback_message(f"Error creating directory: {e}", "error")
        raise


def validate_directory(path: str) -> str:
    dir_path = Path(path).expanduser().resolve()
    if not dir_path.exists():
        create_directory(dir_path)
    elif not dir_path.is_dir():
        feedback_message(f"{dir_path} is not a directory.", "error")
        return validate_directory(
            get_required_input("Please enter a valid directory path: ")
        )
    return str(dir_path)


def create_env_file(env_path: Path) -> None:
    feedback_message(f"Creating new .env file at {env_path}", "info")
    try:
        caption_model = Prompt.ask(
            "[bright_yellow]Enter the name of the caption model: ",
            default="microsoft/Florence-2-large",
        )
        set_key(env_path, CAPTION_MODEL, caption_model)
        feedback_message(f".env file created successfully at {env_path}", "info")
    except Exception as e:
        feedback_message(f"Error creating .env file: {e}", "error")


def load_environment_variables() -> None:
    env_locations: Dict[str, List[str]] = {
        "common": [
            "~/.config/ecko-cli-itsjustregi/.env",
            "~/.ecko-cli-itsjustregi/.env",
            "~/.env",
            "./.env",
        ],
        "Windows": [
            "~/AppData/Roaming/ecko-cli-itsjustregi/.env",
            "~/Documents/ecko-cli-itsjustregi/.env",
        ],
        "Linux": ["~/.local/share/ecko-cli-itsjustregi/.env"],
        "Darwin": ["~/Library/Application Support/ecko-cli-itsjustregi/.env"],
    }

    system_platform = os.name
    search_paths = env_locations["common"] + env_locations.get(system_platform, [])

    for path in search_paths:
        env_path = Path(path).expanduser().resolve()
        if env_path.is_file():
            load_dotenv(env_path)
            if not os.getenv(CAPTION_MODEL):
                feedback_message(
                    f"Required environment variable '{CAPTION_MODEL}' is not set.",
                    "error",
                )
                return typer.Exit(code=1)
            return

    feedback_message(
        ".env file not found in any of the following locations:", "warning"
    )
    for path in search_paths:
        print(f"- {Path(path).expanduser()}")

    create_new = (
        Prompt.ask("Would you like to create a new .env file? (y/n): ").lower().strip()
    )
    if create_new == "y":
        default_path = DEFAULT_ENV_PATH
        custom_path = Prompt.ask(
            f"Enter path for new .env file (default: {default_path}): "
        ).strip()
        env_path = Path(custom_path).expanduser() if custom_path else default_path
        env_path.parent.mkdir(parents=True, exist_ok=True)
        create_env_file(env_path)
        load_dotenv(env_path)
    else:
        feedback_message(
            "No .env file found and user chose not to create one.", "error"
        )
        return typer.Exit(code=1)


load_environment_variables()
os.environ[CAPTION_MODEL] = os.getenv(CAPTION_MODEL, "microsoft/Florence-2-large")
