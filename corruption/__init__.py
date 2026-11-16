"""Shared corruption engine: physically-plausible, temporally-evolving, correlated sensor failures applied at
evaluation time to STF frames, for all five systems (the baseline and the four models), per context.md section 2.

Not model-specific: this package has no dependency on OpenPCDet or any model's own code. A model's own hook (e.g.
models/model1/rgfusion/register.py's M1_EVAL_CORRUPTION) imports a channel's `apply` function from here and calls it
inside its own data pipeline; this package does not know which model is consuming it.

See DESIGN.md before adding a channel or changing the `apply`/`schedule` interface.

Status: design only (corruption/DESIGN.md). No channel is implemented yet; this __init__ exists so the package
layout is fixed before the first channel lands, not to expose anything importable yet.
"""
