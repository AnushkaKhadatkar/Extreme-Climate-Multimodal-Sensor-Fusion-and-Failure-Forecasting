"""The shared channel interface every corruption channel implements. See DESIGN.md section 3-4.

Nothing here does anything by itself -- it is the contract plus a couple of helpers every channel needs
(seeding from frame id, severity validation) so channels don't each reinvent them slightly differently, which
would quietly break the "all five systems evaluated under identical conditions" requirement (context.md section 8)
the moment two channels seed themselves in incompatible ways.
"""

import zlib
from typing import Protocol

import numpy as np


class Channel(Protocol):
    """A corruption channel: context.md section 2's table ("Fog", "Icing", etc). Implementations live in their
    own module (fog.py, icing.py, ...), each exporting a module-level `apply` matching this signature."""

    def __call__(self, data_dict: dict, severity: float, rng: np.random.Generator) -> dict:
        """Mutates and returns data_dict. See DESIGN.md section 3 for the exact contract (which keys may change,
        determinism requirements, severity in [0, 1])."""
        ...


def seed_for_frame(frame_id: str, channel_name: str, severity: float) -> int:
    """Deterministic seed for one (frame, channel, severity): the same triple always gets the same corruption
    draw, matching the convention already used by models/model1/rgfusion/register.py's eval-corruption hook
    (seed from the frame id via zlib.crc32) and extended here with the channel name and severity so that applying
    two different channels, or the same channel at two severities, to the same frame does not reuse one draw.
    Deliberately NOT keyed on anything about the model under evaluation: the whole point is that every model sees
    the identical corrupted frame for a given (frame, channel, severity)."""
    key = f"{frame_id}:{channel_name}:{severity:.6f}".encode()
    return zlib.crc32(key)


def rng_for_frame(frame_id: str, channel_name: str, severity: float) -> np.random.Generator:
    """seed_for_frame, wrapped as a ready-to-use Generator -- what most channels will actually call."""
    return np.random.default_rng(seed_for_frame(frame_id, channel_name, severity))


def check_severity(severity: float) -> None:
    """Every channel's apply() should call this first. Raises, rather than silently clamping, because a
    severity outside [0, 1] reaching a channel is a caller bug (e.g. a bad schedule, DESIGN.md section 4), not
    something to paper over -- the same "fail loudly" stance baseline/scripts/05_verify_conversion.py takes."""
    if not (0.0 <= severity <= 1.0):
        raise ValueError(f"severity must be in [0, 1], got {severity!r}")
