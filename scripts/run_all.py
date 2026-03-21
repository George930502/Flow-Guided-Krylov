#!/usr/bin/env python3
"""Phase 4: Full benchmark — unified entry point using run_benchmark.

This script delegates to run_benchmark.py which handles all 16 molecules
across all 9 methods with automatic feasibility checks and hyperparameter tuning.

Usage:
    python scripts/run_all.py                    # all molecules
    python scripts/run_all.py --tier 1           # only Tier 1 (≤14Q)
    python scripts/run_all.py --tier 3           # Tier 1-3
    python scripts/run_all.py --molecules H2,LiH # specific molecules

For backwards compatibility, positional arguments are treated as molecule names:
    python scripts/run_all.py H2 LiH H2O
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_benchmark import run_benchmark, main as benchmark_main


if __name__ == "__main__":
    # Support old-style positional args: python run_all.py H2 LiH
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        run_benchmark(molecule_names=sys.argv[1:])
    else:
        benchmark_main()
