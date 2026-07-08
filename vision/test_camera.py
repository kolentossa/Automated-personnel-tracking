"""Small manual smoke test for video-file input.

This module intentionally defaults to a local video file instead of a physical
camera because the initial RK3588 demo must work without camera hardware.
"""

from __future__ import annotations

import argparse

from vision.camera import VideoFileCamera


def main() -> int:
    parser = argparse.ArgumentParser(description="Read frames from a local video file.")
    parser.add_argument("video", help="Path to an mp4 file inside the project data directory.")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to read.")
    args = parser.parse_args()

    camera = VideoFileCamera(args.video, loop=False)
    count = 0
    try:
        while count < args.frames:
            frame = camera.read()
            if frame is None:
                break
            count += 1
    finally:
        camera.release()
    print(f"read_frames={count} status={camera.status}")
    return 0 if count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
