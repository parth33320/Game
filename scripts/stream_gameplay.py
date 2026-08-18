import subprocess
import os
import sys
from typing import List, Optional

class FFmpegRestreamStreamer:
    """
    Headless streaming pipeline using Python + FFmpeg to stream gameplay media/audio
    directly to Restream RTMP ingest servers without OBS or GUI overhead.
    """
    def __init__(
        self,
        stream_key: str = "live_12345_example",
        rtmp_url: str = "rtmp://live.restream.io/live",
        video_source: str = "testsrc=size=1280x720:rate=30",
        audio_source: str = "sine=frequency=440:sample_rate=44100"
    ):
        self.stream_key = stream_key
        self.rtmp_url = rtmp_url.rstrip("/")
        self.video_source = video_source
        self.audio_source = audio_source
        self.process: Optional[subprocess.Popen] = None

    def build_ffmpeg_command(self) -> List[str]:
        target_endpoint = f"{self.rtmp_url}/{self.stream_key}"
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "lavfi", "-i", self.video_source,
            "-f", "lavfi", "-i", self.audio_source,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-maxrate", "3000k",
            "-bufsize", "6000k",
            "-pix_fmt", "yuv420p",
            "-g", "60",
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", "44100",
            "-f", "flv",
            target_endpoint
        ]
        return cmd

    def start_stream(self, dry_run: bool = False) -> List[str]:
        cmd = self.build_ffmpeg_command()
        if dry_run:
            return cmd

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return cmd

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
    print("Built FFmpeg Command:", " ".join(streamer.build_ffmpeg_command()))
