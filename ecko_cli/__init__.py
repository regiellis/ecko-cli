import os
from platform import system
from pathlib import Path
from rich.console import Console
from dotenv import load_dotenv

from .helpers import feedback_message


def load_environment_variables(console: Console = Console()) -> None:
    """
    Load environment variables from .env file.

    If the file is not found in the default location, it will be searched in the current directory.
    If still not found, a warning message will be printed to create the file using the sample.env provided.
    """
    env_paths = {
        "Windows": os.path.join(
            os.path.expanduser("~"), ".config", "ecko-cli-itsjustregi", ".env"
        ),
        "Linux": os.path.join(
            os.path.expanduser("~"), ".config", "ecko-cli-itsjustregi", ".env"
        ),
        "Darwin": os.path.join(
            os.path.expanduser("~"), ".config", "ecko-cli-itsjustregi", ".env"
        ),
    }

    system_platform = system()
    dotenv_path = env_paths.get(system_platform, ".env")

    for path in [Path(dotenv_path), Path(".env")]:
        if path.exists():
            load_dotenv(str(path))
            return

    feedback_message(
        f".env file is missing. Please create one using the sample.env provided.\n - Place in the following location // {dotenv_path}",
        "warning",
    )


load_environment_variables(Console())
