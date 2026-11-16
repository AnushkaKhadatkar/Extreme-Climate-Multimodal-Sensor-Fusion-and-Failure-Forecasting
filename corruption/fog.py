"""Fog channel: wraps Hahner et al.'s public LiDAR_fog_sim (ICCV 2021). See DESIGN.md sections 2 and 5.

Public interface is `apply(data_dict, severity, rng)`, matching corruption/base.py's Channel protocol. Internally
this maps severity in [0, 1] onto Hahner's alpha (attenuation coefficient), lidar-only (camera is untouched --
Hahner's detectors, and this simulator, are lidar-only; context.md section 3).

Why severity is NOT mapped onto ParameterSet's full documented alpha range [0.003, 0.5]: `simulate_fog`'s soft-target
(fog backscatter) mode needs a precomputed lookup table, and `get_integral_dict` SILENTLY snaps any requested alpha to
the nearest one that has a shipped table (fog_simulation.py's `get_integral_dict`: `min(alphas, key=lambda x: abs(x -
p.alpha))`, no warning on a mismatch). Only 9 alphas have shipped tables (ALPHAS_WITH_LUT below). Mapping severity
onto the full theoretical range would mean most severity values silently get a different, closest-available alpha
with no sign anything happened -- exactly the kind of silent protocol drift context.md section 10 and this project's
whole "check protocol before touching the model" discipline exist to prevent. So severity maps onto the shipped
alphas directly: severity=1.0 is genuinely the densest fog this vendored copy can simulate with backscatter, not a
value that happens to round down to it.
"""

import sys
from pathlib import Path

import numpy as np

from corruption.base import check_severity

_FOG_SIM_DIR = Path(__file__).resolve().parent / "third_party" / "LiDAR_fog_sim"
if str(_FOG_SIM_DIR) not in sys.path:
    # fog_simulation.py resolves its LUT folder relative to its own __file__ (INTEGRAL_PATH), and is not an
    # importable package (no __init__.py, by upstream's design -- it is meant to be run as a script or imported
    # with its directory on sys.path). Importing it by absolute path avoids polluting sys.path for everything
    # else in this process, at the cost of needing to be the one place that knows this detail.
    sys.path.insert(0, str(_FOG_SIM_DIR))

import fog_simulation  # noqa: E402
from fog_simulation import ParameterSet, get_available_alphas, simulate_fog  # noqa: E402

# Read from disk (fog_simulation.get_available_alphas(), which lists third_party/LiDAR_fog_sim/
# integral_lookup_tables/original/) rather than hardcoded, so adding a LUT (as the fog-channel validation sweep
# did, generating 0.04/0.05/0.07/0.08 around the real-dense-fog-matching alpha=0.06 -- see DESIGN.md section 5/9)
# immediately extends what severity can resolve to, with no risk of this tuple silently drifting out of sync with
# what is actually on disk. Originally hardcoded to the 9 shipped alphas; now reads whatever is actually there
# (13, as of the validation sweep). Sorted ascending: index 0 is the lightest fog this channel can simulate with
# backscatter, the last is the densest.
ALPHAS_WITH_LUT = tuple(get_available_alphas())

CHANNEL_NAME = "fog"


def severity_to_alpha(severity: float) -> float:
    """severity in [0, 1] -> one of ALPHAS_WITH_LUT, evenly spaced by index (not interpolated, so the result is
    always an alpha the shipped LUTs actually have -- see module docstring)."""
    check_severity(severity)
    idx = round(severity * (len(ALPHAS_WITH_LUT) - 1))
    return ALPHAS_WITH_LUT[idx]


def apply(data_dict: dict, severity: float, rng: np.random.Generator, *, hard: bool = True, soft: bool = True,
          noise: int = 10, noise_variant: str = "v1") -> dict:
    """Corruption channel per corruption/base.py's Channel protocol. Fogs 'points' in place (same key, same
    (N, 4) shape); 'images', if present, is left untouched (Hahner's simulator and the baseline detector it
    validates against are both lidar-only; context.md section 3). hard/soft/noise/noise_variant are
    simulate_fog's own knobs, exposed here (not hidden) because the validation script in DESIGN.md section 5 will
    need to sweep hard=True,soft=False (closed-form attenuation only, no LUT) against hard=True,soft=True
    (attenuation + backscatter) as part of checking the channel's realism, not just accept one fixed combination
    on faith.

    IMPORTANT: fog_simulation.py's soft-target noise draws from a module-level `RNG = np.random.default_rng(seed=42)`
    set once at import time, NOT from an argument -- every call to simulate_fog advances that one shared stream, so
    without intervention the fog added to a given frame would depend on how many fog calls happened earlier in the
    process (import order, other frames processed first, ...), not on anything about that frame itself. That
    directly breaks this channel's determinism contract (DESIGN.md section 3: "deterministic per (frame, channel,
    severity, seed)"), confirmed by reading fog_simulation.py's source (`RNG.integers`/`RNG.uniform`/`RNG.beta`
    calls, all against that one module-level object). Fixed here, not left as a footnoted limitation: this
    function reseeds that shared module global from the `rng` argument immediately before calling simulate_fog, so
    the noise draws become a deterministic function of whatever seeded `rng` the caller passed in (e.g.
    corruption.base.rng_for_frame(frame_id, "fog", severity)) and nothing else. This does mutate global state in
    fog_simulation's module on every call -- acceptable here because this process only ever runs one corruption
    channel's work at a time per frame (no concurrent fog.apply calls sharing the module), but worth knowing if
    that assumption changes."""
    alpha = severity_to_alpha(severity)
    points = np.asarray(data_dict["points"], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"fog.apply expects points shaped (N, >=4) [x, y, z, intensity, ...], got {points.shape}")

    fog_simulation.RNG = np.random.default_rng(rng.integers(0, 2**32 - 1))
    p = ParameterSet(alpha=alpha)
    augmented, _, _ = simulate_fog(p, points, noise=noise, gain=False, noise_variant=noise_variant,
                                    hard=hard, soft=soft)

    # simulate_fog silently upcasts float32 input to float64 output (confirmed by direct test, see DESIGN.md
    # section 7 item 5) -- cast back so every downstream consumer (voxelisation, .bin writing) that expects
    # float32 lidar is unaffected, per this channel's own contract in DESIGN.md section 3.
    data_dict["points"] = np.ascontiguousarray(augmented, dtype=np.float32)
    return data_dict
