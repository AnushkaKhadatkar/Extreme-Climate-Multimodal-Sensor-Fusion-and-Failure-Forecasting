#!/usr/bin/env python3
"""Read OpenPCDet eval logs for the baseline reproduction. Pure stdlib, runs
anywhere (inside the Docker image or on the host).

Two modes:

  best  -- parse an `tools/test.py --eval_all` log (val split), print the epoch
           with the best Car 3D AP, R40, IoU 0.5, moderate, and its checkpoint
           path. This is the paper's selection rule ("best performance on the
           clear weather validation set"). With --print-ckpt-only, prints just
           the checkpoint path, for shell scripts.

  gate  -- parse the three per-split test logs and print the Table 2 row, the
           77.05 +/- 2 gate verdict, and the moderate > easy fog inversion check
           (context.md section 3).

Why parse text logs: OpenPCDet's structured ret_dict / tensorboard only carries
the IoU-0.7 numbers for Car (see baseline/README.md "Gate check"). The number we
need lives only in the printed `Car AP_R40@0.70, 0.50, 0.50:` block.

Usage:
    python 07_read_eval_logs.py best --log eval_all_val.log --ckpt-dir <.../ckpt>
    python 07_read_eval_logs.py gate --logs eval_test_clear.log eval_test_light_fog.log eval_test_dense_fog.log
"""

import argparse
import re
import sys
from pathlib import Path

# The R40, IoU-0.5 Car block. The header's first 0.70 is the 2D-bbox threshold;
# the 0.50s are BEV/3D -- see README. Lines after the first carry no logger prefix.
CAR_R40_05 = re.compile(
    r"Car AP_R40@0\.70, 0\.50, 0\.50:[^\n]*\n"
    r"[^\n]*bbox AP:[^\n]*\n"
    r"[^\n]*bev\s+AP:[^\n]*\n"
    r"[^\n]*3d\s+AP:\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)"
)
EPOCH_HEADER = re.compile(r"EPOCH (\S+) EVALUATION")
SAMPLES = re.compile(r"Total samples for KITTI dataset: (\d+)")

GATE_TARGET, GATE_TOL = 77.05, 2.0
PAPER = {  # Hahner et al. ICCV 2021, Table 2, PV-RCNN dagger, alpha=0
    "test_clear": (79.61, 77.05, 71.03),
    "test_light_fog": (69.55, 70.17, 68.44),
    "test_dense_fog": (45.03, 46.00, 45.08),
}
EXPECTED_SAMPLES = {"val": 781, "test_clear": 1847, "test_light_fog": 946, "test_dense_fog": 786}


def car_3d(text: str) -> tuple[float, float, float] | None:
    m = CAR_R40_05.search(text)
    return tuple(float(v) for v in m.groups()) if m else None


def cmd_best(args) -> int:
    text = Path(args.log).read_text(errors="replace")
    chunks = EPOCH_HEADER.split(text)
    # split() gives [preamble, epoch1, body1, epoch2, body2, ...]
    results = {}
    for epoch, body in zip(chunks[1::2], chunks[2::2]):
        ap = car_3d(body)
        if ap is not None:
            results[epoch] = ap
    if not results:
        print(f"no 'Car AP_R40@0.70, 0.50, 0.50' blocks found in {args.log}", file=sys.stderr)
        return 1
    best_epoch = max(results, key=lambda e: results[e][1])
    ckpt = Path(args.ckpt_dir) / f"checkpoint_epoch_{best_epoch}.pth" if args.ckpt_dir else None
    if args.print_ckpt_only:
        print(ckpt if ckpt else best_epoch)
        return 0
    print(f"{'epoch':>6}  {'easy':>7} {'mod':>7} {'hard':>7}   (val, Car 3D AP R40 @ IoU 0.5)")
    for epoch in sorted(results, key=lambda e: float(e) if e.replace('.', '').isdigit() else 0):
        e, m, h = results[epoch]
        flag = "  <- best" if epoch == best_epoch else ""
        print(f"{epoch:>6}  {e:7.2f} {m:7.2f} {h:7.2f}{flag}")
    print(f"\nbest epoch: {best_epoch} (val moderate {results[best_epoch][1]:.2f})")
    if ckpt:
        print(f"checkpoint: {ckpt}")
    return 0


def cmd_gate(args) -> int:
    rows = {}
    ok = True
    for log in args.logs:
        text = Path(log).read_text(errors="replace")
        split = next((s for s in PAPER if s in Path(log).name), None)
        if split is None:
            print(f"cannot tell which split {log} is; name it eval_<split>.log", file=sys.stderr)
            return 1
        n = SAMPLES.findall(text)
        n = int(n[-1]) if n else None
        if n != EXPECTED_SAMPLES[split]:
            # The R1 bug symptom: a test log that actually evaluated val (781).
            print(f"[FAIL] {split}: evaluated {n} samples, expected {EXPECTED_SAMPLES[split]} "
                  f"-- INFO_PATH override not applied?")
            ok = False
        ap = car_3d(text)
        if ap is None:
            print(f"[FAIL] {split}: no 'Car AP_R40@0.70, 0.50, 0.50' block in {log}")
            ok = False
            continue
        rows[split] = ap

    print(f"\n{'split':<16} {'easy':>7} {'mod':>7} {'hard':>7}   paper (easy/mod/hard)")
    for split, (e, m, h) in rows.items():
        pe, pm, ph = PAPER[split]
        print(f"{split:<16} {e:7.2f} {m:7.2f} {h:7.2f}   {pe:.2f} / {pm:.2f} / {ph:.2f}")

    if "test_clear" in rows:
        mod = rows["test_clear"][1]
        passed = abs(mod - GATE_TARGET) <= GATE_TOL
        ok &= passed
        print(f"\nGATE clear/moderate = {mod:.2f}, band {GATE_TARGET - GATE_TOL:.2f}-{GATE_TARGET + GATE_TOL:.2f}: "
              f"{'PASS' if passed else 'FAIL -- check protocol before anything else (context.md s3)'}")
    for split in ("test_light_fog", "test_dense_fog"):
        if split in rows:
            e, m, _ = rows[split]
            print(f"inversion {split}: moderate {'>' if m > e else '<='} easy "
                  f"({m:.2f} vs {e:.2f}) -- paper has moderate > easy")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("best", help="pick best-on-val checkpoint from an --eval_all log")
    b.add_argument("--log", required=True)
    b.add_argument("--ckpt-dir", default=None)
    b.add_argument("--print-ckpt-only", action="store_true")
    g = sub.add_parser("gate", help="report Table 2 row and gate verdict from test logs")
    g.add_argument("--logs", nargs="+", required=True)
    args = ap.parse_args()
    return cmd_best(args) if args.cmd == "best" else cmd_gate(args)


if __name__ == "__main__":
    raise SystemExit(main())
