import pytest
from typer.testing import CliRunner
from ecko_cli.cli import ecko_cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.parametrize(
    "command, args, expected_output",
    [
        (
            ["process-images", "sample_images/training_data", "maya"],
            0,
            "Processing images...",
        ),
        (
            ["process-images", "sample_images/training_data", "maya", "--padding", "4"],
            0,
            "Processing images...",
        ),
        (["create-jsonl", "sample_images/training_data", ""], 0, "JSONL file created:"),
        (
            ["create-jsonl", "sample_images/training_data", "dataset"],
            0,
            "JSONL file created:",
        ),
    ],
)
def test_commands(runner, command, args, expected_output):
    result = runner.invoke(ecko_cli, command)
    # print(f"Command: {command}")
    # print(f"Exit Code: {result.exit_code}")
    # print(f"Output: {result.stdout}")
    assert result.exit_code == args
    if expected_output:
        assert expected_output in result.stdout


def test_invalid_command(runner):
    result = runner.invoke(ecko_cli, ["nonexistent-command"], catch_exceptions=False)
    # print("Command: nonexistent-command")
    # print(f"Exit Code: {result.exit_code}")
    # print(f"Output: {result.stdout}")
    assert result.exit_code == 2
    assert "No such command 'nonexistent-command'." in result.stdout
