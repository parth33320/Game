import subprocess
import os
import sys
import time
from typing import List, Optional

class FFmpegRestreamStreamer:
    """
    Headless 24/7 video streaming pipeline using Python + FFmpeg to render and pipe
    gameplay video feeds (via virtual framebuffer Xvfb or raw video frame buffer pipe)
    directly to local video files or RTMP broadcast endpoints without OBS or GUI overhead.
    """
    def __init__(
        self,
        stream_key: str = "live_12345_example",
        rtmp_url: str = "rtmp://live.restream.io/live",
        use_xvfb: bool = True,
        xvfb_display: str = ":99",
        width: int = 1280,
        height: int = 720,
        fps: int = 30
    ):
        self.stream_key = stream_key
        self.rtmp_url = rtmp_url.rstrip("/")
        self.use_xvfb = use_xvfb
        self.xvfb_display = xvfb_display
        self.width = width
        self.height = height
        self.fps = fps
        self.process: Optional[subprocess.Popen] = None

    def build_ffmpeg_command(self, raw_pipe: bool = False, output_target: Optional[str] = None) -> List[str]:
        target_endpoint = output_target or f"{self.rtmp_url}/{self.stream_key}"

        cmd = ["ffmpeg", "-y"]

        if raw_pipe:
            cmd.extend([
                "-f", "rawvideo",
                "-pixel_format", "rgb24",
                "-video_size", f"{self.width}x{self.height}",
                "-framerate", str(self.fps),
                "-i", "pipe:0"
            ])
        elif self.use_xvfb and os.environ.get("DISPLAY") == self.xvfb_display:
            cmd.extend([
                "-f", "x11grab",
                "-draw_mouse", "0",
                "-video_size", f"{self.width}x{self.height}",
                "-framerate", str(self.fps),
                "-i", f"{self.xvfb_display}.0"
            ])
        else:
            # Synthetic lavfi source fallback for headless / server testing
            video_src = f"testsrc=size={self.width}x{self.height}:rate={self.fps}"
            audio_src = "sine=frequency=440:sample_rate=44100"
            cmd.extend([
                "-f", "lavfi", "-i", video_src,
                "-f", "lavfi", "-i", audio_src
            ])

        cmd.extend([
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-maxrate", "3000k",
            "-bufsize", "6000k",
            "-pix_fmt", "yuv420p",
            "-g", str(self.fps * 2),
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "44100",
            "-f", "flv" if target_endpoint.startswith("rtmp") else "mp4",
            target_endpoint
        ])
        return cmd

    def start_stream(self, dry_run: bool = False, raw_pipe: bool = False, output_target: Optional[str] = None) -> List[str]:
        cmd = self.build_ffmpeg_command(raw_pipe=raw_pipe, output_target=output_target)
        if dry_run:
            return cmd

        stdin_mode = subprocess.PIPE if raw_pipe else None
        self.process = subprocess.Popen(
            cmd,
            stdin=stdin_mode,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return cmd

    def write_frame(self, frame_bytes: bytes):
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(frame_bytes)
                self.process.stdin.flush()
            except BrokenPipeError:
                pass

    def stop_stream(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

if __name__ == "__main__":
    streamer = FFmpegRestreamStreamer()
    print("Built FFmpeg Stream Command:", " ".join(streamer.build_ffmpeg_command()))
