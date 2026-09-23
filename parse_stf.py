#!/usr/bin/env python3
"""Parse stf.json (Seeing Through Fog dataset S3 URL manifest).

The file has the shape:
{
  "urls": [
    {"key": "SeeingThroughFog/...", "url": "https://...presigned-s3-url..."},
    ...
  ]
}

Each entry is a pre-signed S3 URL for one object. Multi-part archives are
split across sibling files sharing a basename, e.g.:
  cam_stereo_left/cam_stereo_left.zip
  cam_stereo_left/cam_stereo_left.z01
  cam_stereo_left/cam_stereo_left.z02
  ...

This script loads the manifest, groups entries by top-level component
(the subfolder under SeeingThroughFog/, or the bare filename for files
at the root), and reports counts/sizes-of-parts per component. It also
extracts the presigned URL expiry from each URL's X-Amz-Date/X-Amz-Expires
query params.
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse, parse_qs


@dataclass
class Entry:
    key: str
    url: str
    component: str  # top-level folder or root filename
    filename: str
    ext: str
    signed_date: datetime | None = None
    expires_seconds: int | None = None

    @property
    def expiry(self) -> datetime | None:
        if self.signed_date is None or self.expires_seconds is None:
            return None
        return self.signed_date + timedelta(seconds=self.expires_seconds)

    @property
    def is_expired(self) -> bool:
        exp = self.expiry
        return exp is not None and datetime.now(timezone.utc) > exp


@dataclass
class Component:
    name: str
    entries: list = field(default_factory=list)

    @property
    def total_parts(self) -> int:
        return len(self.entries)

    @property
    def extensions(self) -> list:
        return sorted({e.ext for e in self.entries})


def parse_amz_datetime(value: str) -> datetime | None:
    # Format: 20260912T062614Z
    try:
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def load_manifest(path: Path) -> list[Entry]:
    with open(path) as f:
        data = json.load(f)

    raw_urls = data.get("urls", [])
    entries = []
    for item in raw_urls:
        key = item["key"]
        url = item["url"]

        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        amz_date = qs.get("X-Amz-Date", [None])[0]
        expires_raw = qs.get("X-Amz-Expires", [None])[0]

        signed_date = parse_amz_datetime(amz_date) if amz_date else None
        expires_seconds = int(expires_raw) if expires_raw else None

        rel = key[len("SeeingThroughFog/"):] if key.startswith("SeeingThroughFog/") else key
        rel = rel.rstrip("/")
        parts = rel.split("/") if rel else []

        if len(parts) == 0:
            component = "SeeingThroughFog"
            filename = ""
        elif len(parts) == 1:
            component = "(root)"
            filename = parts[0]
        else:
            component = parts[0]
            filename = parts[-1]

        ext = Path(filename).suffix if filename else ""

        entries.append(
            Entry(
                key=key,
                url=url,
                component=component,
                filename=filename,
                ext=ext,
                signed_date=signed_date,
                expires_seconds=expires_seconds,
            )
        )
    return entries


def group_by_component(entries: list) -> dict:
    groups = defaultdict(lambda: Component(name=""))
    for e in entries:
        g = groups[e.component]
        g.name = e.component
        g.entries.append(e)
    return dict(groups)


def print_summary(entries: list) -> None:
    groups = group_by_component(entries)

    print(f"Total entries: {len(entries)}")
    print(f"Total components (folders/root files): {len(groups)}")
    print()

    expired = [e for e in entries if e.is_expired]
    if expired:
        print(f"WARNING: {len(expired)} URL(s) appear expired.")
    else:
        sample = next((e for e in entries if e.expiry), None)
        if sample:
            print(f"Sample expiry: {sample.expiry} UTC (all URLs share the same batch expiry pattern)")
    print()

    print(f"{'component':40s} {'parts':>6s}  extensions")
    print("-" * 80)
    for name in sorted(groups, key=lambda n: -groups[n].total_parts):
        g = groups[name]
        print(f"{name:40s} {g.total_parts:6d}  {', '.join(g.extensions)}")


def export_csv(entries: list, out_path: Path) -> None:
    import csv

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "component", "filename", "ext", "signed_date", "expires_seconds", "expiry", "url"])
        for e in entries:
            writer.writerow(
                [e.key, e.component, e.filename, e.ext, e.signed_date, e.expires_seconds, e.expiry, e.url]
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", nargs="?", default="stf.json", help="Path to stf.json")
    ap.add_argument("--component", help="Show all entries for one component (folder name)")
    ap.add_argument("--csv", type=Path, help="Export parsed rows to this CSV path")
    args = ap.parse_args()

    path = Path(args.manifest)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    entries = load_manifest(path)

    if args.component:
        groups = group_by_component(entries)
        g = groups.get(args.component)
        if not g:
            print(f"No component named {args.component!r}. Known: {sorted(groups)}", file=sys.stderr)
            return 1
        for e in g.entries:
            print(e.key)
        print(f"\n{g.total_parts} part(s) in {args.component}")
        return 0

    print_summary(entries)

    if args.csv:
        export_csv(entries, args.csv)
        print(f"\nWrote CSV to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
