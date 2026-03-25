# tests/test_runner.py
import pytest
import subprocess
import pathlib
import os

def find_test_files():
    """Recursively finds all .in files in the tests directory."""
    base_path = pathlib.Path(__file__).parent
    return list(base_path.rglob("*.in"))

@pytest.mark.parametrize("input_file", find_test_files(), ids=lambda x: x.name)
def test_docs(input_file, pytestconfig):
    # Get the binary path from the --target flag
    target_executable = pytestconfig.getoption("target")
    mpi_procs = pytestconfig.getoption("mpi_procs")
    mpi_launcher = pytestconfig.getoption("mpi_launcher")
    expected_file = input_file.with_suffix(".out")

    if not expected_file.exists():
        pytest.fail(f"Missing expected output file: {expected_file}")

    cpus = os.environ.get("OMP_NUM_THREADS", "1")

    if mpi_procs > 0:
        if mpi_launcher == "srun":
            cmd = [mpi_launcher, "-n", str(mpi_procs), "-c", cpus, target_executable, str(input_file)]
        else:
            cmd = [mpi_launcher, "-n", str(mpi_procs), target_executable, str(input_file)]
    else:
        cmd = [target_executable, str(input_file)]

    process = subprocess.run(cmd, text=True, capture_output=True)

    # Show stderr (useful for OpenMP/MPI stats)
    if process.stderr:
        print(f"\n--- STDERR ({input_file.name}) ---\n{process.stderr}")

    assert process.returncode == 0, f"Error! Stderr: {process.stderr}"

    # Compare results
    with open(expected_file, "r") as f:
        expected_output = f.read()

    assert process.stdout.strip() == expected_output.strip()