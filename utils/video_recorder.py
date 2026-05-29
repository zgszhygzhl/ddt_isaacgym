import os
import subprocess

import numpy as np


class FfmpegVideoWriter:
    def __init__(self, output_path, width, height, fps):
        self.output_path = output_path
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.process = None

        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        command = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-profile:v",
            "baseline",
            "-level",
            "3.1",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            self.output_path,
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame):
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Video writer is already closed")

        frame_array = np.asarray(frame)
        if frame_array.shape != (self.height, self.width, 3):
            raise ValueError(
                f"Expected frame shape {(self.height, self.width, 3)}, got {frame_array.shape}"
            )
        if frame_array.dtype != np.uint8:
            raise ValueError(f"Expected uint8 frame, got {frame_array.dtype}")

        self.process.stdin.write(np.ascontiguousarray(frame_array).tobytes())

    def close(self):
        if self.process is None:
            return

        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        self.process = None

        if return_code != 0:
            raise RuntimeError("ffmpeg exited with a non-zero status")