import subprocess
import os

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
PYTHON_EXE = os.path.join(ROOT_DIR, "..", "venv", "Scripts", "python.exe")

def run_script(rel_path):
    script_path = os.path.abspath(os.path.join(ROOT_DIR, "..", rel_path))
    print(f"Running {script_path} ...")
    subprocess.check_call([PYTHON_EXE, script_path], cwd=ROOT_DIR)

if __name__ == "__main__":
    # Run all experiments (including large‑scale ones)
    run_script("experiments/python/run_all_experiments.py")
    # Generate the publication‑quality plots
    run_script("experiments/python/generate_plots.py")
    print("Benchmark and plot generation completed.")
