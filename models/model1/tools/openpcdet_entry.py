#!/usr/bin/env python3
"""Run OpenPCDet's own tools/train.py or tools/test.py with Model 1 registered.

    python openpcdet_entry.py train --cfg_file cfgs/model1/model1_lidar_rgb.yaml ...
    python openpcdet_entry.py test  --cfg_file cfgs/model1/model1_lidar_rgb.yaml ...

Everything after the first argument is passed to OpenPCDet unchanged. The
script imports rgfusion.register (adds Model1Fusion and the modality-dropout
augmentor), then executes OpenPCDet's script as __main__ from
<OPENPCDET_ROOT>/tools, the directory OpenPCDet resolves configs and data from.
"""

import os
import runpy
import sys
from pathlib import Path

MODEL1_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL1_DIR))

import rgfusion.register  # noqa: E402,F401  (side effect: registration)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "test"):
        sys.exit(__doc__)
    tools = Path(os.environ.get("OPENPCDET_ROOT", "/opt/OpenPCDet")) / "tools"
    script = tools / f"{sys.argv[1]}.py"
    os.chdir(tools)
    sys.path.insert(0, str(tools))          # train.py does `from test import repeat_eval_ckpt`
    sys.argv = [str(script)] + sys.argv[2:]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
