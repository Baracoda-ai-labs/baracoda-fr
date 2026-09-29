# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Baracoda Group (Baracoda AI Labs)
"""Pipeline runner (replaces a Makefile). Usage: python run.py <step> [<step> ...]
Steps: corpora, eval, mistral, data, train, contamination, controls, ablation, claude, all (= corpora eval mistral)."""
import os, subprocess, sys

S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
STEPS = {
    "corpora": [["prepare_corpora.py"]],
    "eval": [["evaluate.py", "--require-all"]],
    "mistral": [["mistral_versions.py"]],
    "data": [["collect_training.py", s] for s in ["fr_web", "en_web", "europarl_fr", "europarl_en", "python"]],
    "train": [["train.py"], ["train.py", "--vocab", "130076", "--out", "../tokenizers/controls/baracoda-fr-v1_2-equal-ordinary-budget.json"]],
    "contamination": [["contamination.py"]],
    "controls": [["controls.py"]],
    "ablation": [["ablation.py", "--vocab", "50000"]],
    "claude": [["claude_counts.py"]],
}
STEPS["all"] = STEPS["corpora"] + STEPS["eval"] + STEPS["mistral"]

if __name__ == "__main__":
    if len(sys.argv) < 2 or any(a not in STEPS for a in sys.argv[1:]):
        raise SystemExit(__doc__)
    for step in sys.argv[1:]:
        for cmd in STEPS[step]:
            print("->", step, " ".join(cmd), flush=True)
            subprocess.run([sys.executable, *cmd], cwd=S, check=True)
