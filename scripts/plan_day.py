"""Standalone entry so Hermes can call the day planner from any working dir."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latentbridge.plan_cli import main  # noqa: E402

if __name__ == "__main__":
    main()
