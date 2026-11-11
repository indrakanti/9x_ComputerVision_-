#!/usr/bin/env python3
"""Prepare a compact real-video single-object tracking benchmark.

Downloads the Supervisely Oriented Bounding Box Video Sample Annotated
v1.0.0 project archive, selects one object with the longest contiguous
annotation run, converts oriented rectangles to axis-aligned boxes, and writes
frame,x,y,w,h CSV for cv9x_mosse_correlation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable

SOURCE_URL = (
    "https://github.com/supervisely-ecosystem/"
    "oriented-bbox-video-sample-annotated/releases/download/v1.0.0/project.zip"
)
SOURCE_REPO = (
    "https://github.com/supervisely-ecosystem/"
    "oriented-bbox-video-sample-annotated"
)
ARCHIVE_NAME = "supervisely_oriented_bbox_v1.0.0.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="build/real_tracking_sample")
    parser.add_argument("--object-key", default=None)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def find_annotation(root: Path) -> tuple[Path, dict[str, Any]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in root.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and isinstance(data.get("frames"), list):
            candidates.append((path, data))
    if not candidates:
        raise RuntimeError("No Supervisely video annotation JSON with frames found.")
    candidates.sort(key=lambda item: len(item[1].get("frames", [])), reverse=True)
    return candidates[0]


def find_video(root: Path) -> Path:
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = [p for p in root.rglob("*") if p.suffix.lower() in extensions]
    if not videos:
        raise RuntimeError("No video file found in extracted project.")
    videos.sort(key=lambda p: p.stat().st_size, reverse=True)
    return videos[0]


def figure_object_key(figure: dict[str, Any]) -> str | None:
    value = figure.get("objectKey")
    if value is None:
        value = figure.get("objectId")
    return str(value) if value is not None else None


def longest_contiguous_run(indices: Iterable[int]) -> list[int]:
    values = sorted(set(indices))
    if not values:
        return []
    best: list[int] = []
    current = [values[0]]
    for value in values[1:]:
        if value == current[-1] + 1:
            current.append(value)
        else:
            if len(current) > len(best):
                best = current
            current = [value]
    if len(current) > len(best):
        best = current
    return best


def collect_object_frames(annotation: dict[str, Any]) -> dict[str, list[int]]:
    tracks: dict[str, list[int]] = {}
    for frame in annotation.get("frames", []):
        frame_index = int(frame.get("index", -1))
        for figure in frame.get("figures", []):
            key = figure_object_key(figure)
            if key is not None and frame_index >= 0:
                tracks.setdefault(key, []).append(frame_index)
    return tracks


def choose_object(
    annotation: dict[str, Any],
    requested_key: str | None,
) -> tuple[str, list[int]]:
    tracks = collect_object_frames(annotation)
    if not tracks:
        raise RuntimeError("No object tracks found in Supervisely annotation.")

    if requested_key is not None:
        if requested_key not in tracks:
            raise RuntimeError(f"Object key {requested_key!r} not found.")
        return requested_key, longest_contiguous_run(tracks[requested_key])

    scored = []
    for key, indices in tracks.items():
        run = longest_contiguous_run(indices)
        scored.append((len(run), key, run))
    scored.sort(reverse=True)
    _, key, run = scored[0]
    return key, run


def rotate_point(
    point: tuple[float, float],
    center: tuple[float, float],
    angle_degrees: float,
) -> tuple[float, float]:
    angle = math.radians(angle_degrees)
    c = math.cos(angle)
    s = math.sin(angle)
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    return (
        center[0] + c * dx - s * dy,
        center[1] + s * dx + c * dy,
    )


def geometry_to_aabb(figure: dict[str, Any]) -> tuple[float, float, float, float]:
    geometry = figure.get("geometry", {})
    points = geometry.get("points", {})
    exterior = points.get("exterior", [])
    if not isinstance(exterior, list) or len(exterior) < 2:
        raise RuntimeError("Unsupported figure geometry: missing exterior points.")

    parsed = [(float(p[0]), float(p[1])) for p in exterior]
    if len(parsed) == 2:
        x1, y1 = parsed[0]
        x2, y2 = parsed[1]
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        corners = [(left, top), (right, top), (right, bottom), (left, bottom)]
        rotation = float(geometry.get("rotation", figure.get("rotation", 0.0)) or 0.0)
        if abs(rotation) > 1e-12:
            center = ((left + right) * 0.5, (top + bottom) * 0.5)
            corners = [rotate_point(p, center, rotation) for p in corners]
    else:
        corners = parsed

    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    left = min(xs)
    top = min(ys)
    right = max(xs)
    bottom = max(ys)
    return left, top, right - left, bottom - top


def frame_figure(
    annotation: dict[str, Any],
    frame_index: int,
    object_key: str,
) -> dict[str, Any]:
    for frame in annotation.get("frames", []):
        if int(frame.get("index", -1)) != frame_index:
            continue
        for figure in frame.get("figures", []):
            if figure_object_key(figure) == object_key:
                return figure
    raise RuntimeError(
        f"Missing figure for object {object_key!r} at frame {frame_index}."
    )


def write_ground_truth(
    path: Path,
    annotation: dict[str, Any],
    object_key: str,
    run: list[int],
) -> None:
    if len(run) < 2:
        raise RuntimeError("Selected object does not have a usable contiguous track.")

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "x", "y", "w", "h"])
        for frame_index in run:
            figure = frame_figure(annotation, frame_index, object_key)
            x, y, w, h = geometry_to_aabb(figure)
            writer.writerow(
                [frame_index, f"{x:.6f}", f"{y:.6f}", f"{w:.6f}", f"{h:.6f}"]
            )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archive = output_dir / ARCHIVE_NAME
    if args.force_download or not archive.exists():
        download(SOURCE_URL, archive)

    extracted = output_dir / "project"
    if not extracted.exists():
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(extracted)

    annotation_path, annotation = find_annotation(extracted)
    video_path = find_video(extracted)
    object_key, run = choose_object(annotation, args.object_key)

    gt_path = output_dir / "ground_truth.csv"
    write_ground_truth(gt_path, annotation, object_key, run)

    provenance = output_dir / "provenance.txt"
    provenance.write_text(
        "\n".join(
            [
                f"source_repo={SOURCE_REPO}",
                f"source_archive={SOURCE_URL}",
                "source_release=v1.0.0",
                f"annotation={annotation_path}",
                f"video={video_path}",
                f"selected_object_key={object_key}",
                f"first_frame={run[0]}",
                f"last_frame={run[-1]}",
                f"annotated_frames={len(run)}",
                "",
                "External archive downloaded at run time; not redistributed by this repo.",
                "Review upstream source terms before redistributing video or derived assets.",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Video:        {video_path}")
    print(f"Ground truth: {gt_path}")
    print(f"Object key:   {object_key}")
    print(f"Frames:       {run[0]}..{run[-1]} ({len(run)})")
    print("")
    print("Run:")
    print(
        "  ./build/cv9x_mosse_correlation"
        f" --video {video_path}"
        f" --ground-truth {gt_path}"
        f" --output-dir {output_dir / 'results'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
