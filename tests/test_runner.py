# tests/test_runner.py
import pytest
import subprocess
import pathlib

def find_test_files():
    """Recursively finds all .in files in the tests directory."""
    base_path = pathlib.Path(__file__).parent
    return list(base_path.rglob("*.in"))

@pytest.mark.parametrize("input_file", find_test_files(), ids=lambda x: x.name)
def test_cpp_app(input_file, pytestconfig):
    # Get the binary path from the --target flag
    target_executable = pytestconfig.getoption("target")
    
    expected_file = input_file.with_suffix(".out")
    
    if not expected_file.exists():
        pytest.fail(f"Missing expected output file: {expected_file}")

    # Run the application
    # We use capture_output=True but print stderr so -s flag can show it
    process = subprocess.run(
        [target_executable, str(input_file)],
        text=True,
        capture_output=True
    )

    # Show stderr (useful for OpenMP/MPI stats)
    if process.stderr:
        print(f"\n--- STDERR ({input_file.name}) ---\n{process.stderr}")

    assert process.returncode == 0, f"Error! Stderr: {process.stderr}"

    # Compare results
    with open(expected_file, "r") as f:
        expected_output = f.read()

    assert process.stdout.strip() == expected_output.strip()