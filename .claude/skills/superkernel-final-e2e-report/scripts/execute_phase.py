#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "superkernel-auto-tune" / "scripts"))
from phase_entrypoint import run

raise SystemExit(run(manifest=Path(__file__).resolve().parents[1] / "phase.json"))
