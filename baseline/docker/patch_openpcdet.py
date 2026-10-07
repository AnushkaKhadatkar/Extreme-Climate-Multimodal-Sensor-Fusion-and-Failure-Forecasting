"""Build-time patches applied to the OpenPCDet checkout inside the Docker
image. Run as a plain script (not a Dockerfile heredoc -- a heredoc-based
version of this hit a reproducible discrepancy between what ran inside
`docker build` and the identical regex run manually via `docker run`, most
likely BuildKit's heredoc content handling doing something subtle with the
backslash-heavy regex patterns; a real file sidesteps that class of problem
entirely).
"""
import re
from pathlib import Path

OPENPCDET = Path("/opt/OpenPCDet")

# 1. pcdet/datasets/__init__.py unconditionally imports every optional
# dataset connector (nuScenes, Waymo, Pandaset, Lyft, ONCE, Argo2), each
# needing its own heavy devkit (av2, waymo-open-dataset, nuscenes-devkit,
# ...) that this STF-via-KITTI-format-only project has no other use for.
p = OPENPCDET / "pcdet/datasets/__init__.py"
src = p.read_text()
src_new = src.replace(
    "from .kitti.kitti_dataset import KittiDataset\n"
    "from .nuscenes.nuscenes_dataset import NuScenesDataset\n"
    "from .waymo.waymo_dataset import WaymoDataset\n"
    "from .pandaset.pandaset_dataset import PandasetDataset\n"
    "from .lyft.lyft_dataset import LyftDataset\n"
    "from .once.once_dataset import ONCEDataset\n"
    "from .argo2.argo2_dataset import Argo2Dataset\n"
    "from .custom.custom_dataset import CustomDataset",
    "from .kitti.kitti_dataset import KittiDataset\n"
    "from .custom.custom_dataset import CustomDataset"
)
src_new = src_new.replace(
    "    'KittiDataset': KittiDataset,\n"
    "    'NuScenesDataset': NuScenesDataset,\n"
    "    'WaymoDataset': WaymoDataset,\n"
    "    'PandasetDataset': PandasetDataset,\n"
    "    'LyftDataset': LyftDataset,\n"
    "    'ONCEDataset': ONCEDataset,\n"
    "    'CustomDataset': CustomDataset,\n"
    "    'Argo2Dataset': Argo2Dataset\n"
    "}",
    "    'KittiDataset': KittiDataset,\n"
    "    'CustomDataset': CustomDataset,\n"
    "}"
)
assert "NuScenesDataset" not in src_new, "datasets/__init__.py layout differs from what this patch expects"
assert src_new != src, "datasets/__init__.py replacement made no change"
p.write_text(src_new)
print(f"patched {p}: dropped unused optional dataset connectors")

# 2. kitti_dataset.py's create_groundtruth_database() uses pathlib.Path but
# the module never imports it -- NameError at the GT-database step.
p = OPENPCDET / "pcdet/datasets/kitti/kitti_dataset.py"
src = p.read_text()
assert not src.startswith("from pathlib import Path"), "kitti_dataset.py already imports Path -- patch no longer needed"
p.write_text("from pathlib import Path\n" + src)
print(f"patched {p}: added missing pathlib.Path import")

# 3. rotate_iou.py's numba @cuda.jit device/kernel functions use eager,
# explicit string signatures (e.g. @cuda.jit('(float32[:], float32[:],
# float32[:])', device=True, inline=True)). On this RTX 4090 (compute
# capability 8.9), that combination either raises "Signature mismatch: N
# argument types given, but function takes M arguments" (numba 0.68) or
# segfaults mid-kernel during the real KITTI AP eval (numba 0.56.4, tried
# first: recall metrics print fine, the crash is specifically in the AP step
# that calls these kernels -- confirmed by actually running it). Dropping
# the eager signatures lets numba infer argument types from the call site
# instead, which is the actively-exercised path across numba versions and
# GPU architectures. Behavior-preserving: no kernel logic changes.
p = OPENPCDET / "pcdet/datasets/kitti/kitti_object_eval_python/rotate_iou.py"
src = p.read_text()
src, n_sig = re.subn(r"@cuda\.jit\(\s*'[^']*'\s*,\s*", "@cuda.jit(", src)
src, n_inline = re.subn(r",\s*inline\s*=\s*True", "", src)
# 11 of this file's cuda.jit calls carry an eager signature string (10 device
# functions with inline=True, plus the rotate_iou_kernel_eval kernel itself
# with fastmath=False instead) -- verified by dumping the pristine clone and
# counting by hand, not guessed: an earlier version of this patch asserted
# n_inline == 2, which was wrong -- that number came from inspecting an
# already-partially-patched image (a prior `sed` attempt had already fixed
# 8 of the 10 single-line cases and left only the 2 multi-line ones, and
# that leftover state was mistaken for the original).
assert n_sig == 11, f"expected 11 signature strings removed, got {n_sig}"
assert n_inline == 10, f"expected 10 inline=True removed, got {n_inline}"
p.write_text(src)
print(f"patched {p}: removed {n_sig} eager signatures, {n_inline} inline=True")
