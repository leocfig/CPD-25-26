import subprocess
import os
import re
import statistics
import argparse
import shlex
import csv
from pathlib import Path

# --- Configuration ---
TIME_REGEX = re.compile(r"([0-9.]+(?:e-[0-9]+)?)s")

def run_program(executable, prog_args, input_file, num_threads=None):
    """Runs the C++ program and extracts the time from stderr."""
    env = os.environ.copy()
    
    # Combine executable, general args, and the specific input file
    cmd = [executable] + prog_args + [str(input_file)]

    if num_threads is not None:
        env["OMP_NUM_THREADS"] = str(num_threads)
        env["OMP_PROC_BIND"] = "true"
        env["OMP_PLACES"] = "cores"
        
    # Pin to a single core for serial baseline or 1-thread parallel run
    if num_threads is None or num_threads == 1:
        cmd = ["taskset", "-c", "0"] + cmd

    try:
        result = subprocess.run(
            cmd, 
            env=env, 
            capture_output=True, 
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        thread_info = f"{num_threads} threads" if num_threads else "serial"
        print(f"Error: Program crashed during {thread_info} run on {input_file.name}. Return code: {e.returncode}")
        print(f"Stderr:\n{e.stderr}")
        return None

    match = TIME_REGEX.search(result.stderr)
    if match:
        return float(match.group(1))
    else:
        print(f"Error: Could not parse time from stderr. Output was:\n{result.stderr}")
        return None

def run_statistical_suite(executable, prog_args, input_file, repetitions, num_threads=None):
    """Handles warmup, repetitions, and statistical calculations."""
    print("    Running warmup...")
    run_program(executable, prog_args, input_file, num_threads)

    times = []
    for i in range(repetitions):
        t = run_program(executable, prog_args, input_file, num_threads)
        if t is not None:
            times.append(t)
            print(f"      Run {i+1}/{repetitions}: {t:.6g}s")

    if times:
        median_time = statistics.median(times)
        stdev_time = statistics.stdev(times) if len(times) > 1 else 0.0
        print(f"    -> Median: {median_time:.6g}s (StdDev: ±{stdev_time:.6g}s)\n")
        return median_time
    return None

def benchmark(args):
    serial_exec = args.serial
    parallel_exec = args.parallel
    repetitions = args.repetitions
    thread_counts = args.threads
    prog_args = shlex.split(args.prog_args)
    input_dir = Path(args.input_dir)
    csv_file = args.output

    # Validation
    if not os.path.exists(serial_exec) or not os.path.exists(parallel_exec):
        print("Error: Executables not found.")
        return
    if not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' not found.")
        return

    # Grab all .in files and sort them deterministically
    input_files = sorted(input_dir.glob("*.in"))
    if not input_files:
        print(f"No .in files found in {input_dir}")
        return

    print(f"--- Benchmark Configuration ---")
    print(f"Found {len(input_files)} input files.")
    print(f"Outputting results to: {csv_file}\n")

    # Prepare CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Input_File", "Implementation", "Threads", "Median_Time_s", "Speedup", "Efficiency_Pct"])

        # Loop through every input file
        for infile in input_files:
            print("="*60)
            print(f"Testing Input File: {infile.name}")
            print("="*60)

            # --- 1. Run Serial Baseline ---
            print(f"  Benchmarking SERIAL Baseline...")
            serial_median = run_statistical_suite(serial_exec, prog_args, infile, repetitions, num_threads=None)
            
            if serial_median is None:
                print(f"  Skipping {infile.name} due to serial failure.")
                continue
                
            writer.writerow([infile.name, "Serial", 1, f"{serial_median:.6g}", "1.00", "100.0"])

            # --- 2. Run Parallel Configurations ---
            for threads in thread_counts:
                print(f"  Benchmarking PARALLEL with {threads} thread(s)...")
                med = run_statistical_suite(parallel_exec, prog_args, infile, repetitions, num_threads=threads)
                
                if med is not None:
                    speedup = serial_median / med
                    efficiency = (speedup / threads) * 100
                    writer.writerow([infile.name, "Parallel", threads, f"{med:.6g}", f"{speedup:.2f}", f"{efficiency:.1f}"])

    print(f"Benchmarking complete! Results saved to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rigorous OpenMP Benchmarking Script")
    parser.add_argument("-s", "--serial", type=str, required=True, help="Path to serial executable")
    parser.add_argument("-p", "--parallel", type=str, required=True, help="Path to parallel executable")
    parser.add_argument("-i", "--input-dir", type=str, required=True, help="Directory containing .in files")
    parser.add_argument("-o", "--output", type=str, default="results.csv", help="Output CSV file path")
    parser.add_argument("-r", "--repetitions", type=int, default=10, help="Statistical repetitions")
    parser.add_argument("-t", "--threads", type=int, nargs='+', default=[1, 2, 4, 6], help="Thread counts to test")
    parser.add_argument("-a", "--prog-args", type=str, default="", help="Extra args to pass BEFORE the input file")
    
    args = parser.parse_args()
    benchmark(args)