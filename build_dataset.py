#!/usr/bin/env python3
"""Build the real train/test dataset from settled Kalshi WC markets.

  python build_dataset.py [max_per_series]
"""

import sys

from agent.dataset import build_and_save

if __name__ == "__main__":
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    build_and_save(max_per_series=cap)
