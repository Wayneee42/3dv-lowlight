#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xchu.contact@gmail.com)

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.data import validate_blender_scene


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate an official Blender-format scene and optional auxiliaries.")
    parser.add_argument("scene_root", type=str, help="Path to a scene directory containing transforms_*.json")
    parser.add_argument("--auxiliary-dir", type=str, default="auxiliaries")
    args = parser.parse_args()

    result = validate_blender_scene(args.scene_root, auxiliary_dir=args.auxiliary_dir)
    print(json.dumps(result, indent=2))
    if result["errors"]:
        raise SystemExit(1)
