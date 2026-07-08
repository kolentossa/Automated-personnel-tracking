"""Generate a synthetic, privacy-safe mp4 demo video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "data" / "sample.mp4"


def draw_person(frame: np.ndarray, x: int, y: int, color: tuple) -> None:
    cv2.rectangle(frame, (x, y + 20), (x + 34, y + 92), color, thickness=-1)
    cv2.circle(frame, (x + 17, y + 10), 10, color, thickness=-1)


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    width, height, fps, frames = 640, 360, 25, 340
    writer = cv2.VideoWriter(
        str(OUTPUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer for {OUTPUT}")

    for frame_index in range(frames):
        frame = np.full((height, width, 3), (238, 241, 244), dtype=np.uint8)
        cv2.line(frame, (width // 2, 0), (width // 2, height), (205, 211, 216), 2)
        cv2.rectangle(frame, (0, 265), (width, height), (222, 227, 231), thickness=-1)

        x_a = int(30 + frame_index * 1.9)
        if -40 < x_a < width + 40:
            draw_person(frame, x_a, 154, (45, 125, 69))

        x_b = int(width - 65 - max(0, frame_index - 65) * 2.05)
        if frame_index > 65 and -40 < x_b < width + 40:
            draw_person(frame, x_b, 174, (160, 65, 65))

        x_c = int(100 + max(0, frame_index - 170) * 1.5)
        if frame_index > 170 and x_c < width + 40:
            draw_person(frame, x_c, 132, (58, 95, 158))

        x_d = int(width - 90 - max(0, frame_index - 220) * 2.15)
        if frame_index > 220 and -40 < x_d < width + 40:
            draw_person(frame, x_d, 188, (148, 105, 32))

        writer.write(frame)

    writer.release()
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
