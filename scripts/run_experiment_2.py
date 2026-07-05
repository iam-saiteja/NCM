import os
import sys

# Add repository root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from experiments.python.run_all_experiments import experiment_2_novelty_at_scale

if __name__ == "__main__":
    # Run only experiment 2 (novelty at scale) without verbose output to avoid Unicode issues
    print("Running Experiment 2: Novelty Sensitivity at Scale (stand-alone)...")
    experiment_2_novelty_at_scale(verbose=False)
    print("Experiment 2 completed. Results stored in the experiment results directory.")
