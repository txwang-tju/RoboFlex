"""Optional serial glove receiver, independent of arm command/ACK traffic."""
from __future__ import annotations

import json
import threading
import time

from .types import Glove


class SerialGlove:
    def __init__(self, port: str, baudrate: int = 115200):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("Install serial extra to read a glove") from exc
        self.serial = serial.Serial(port, baudrate, timeout=0.05)
        self.lock = threading.Lock()
        self.quit = threading.Event()
        self.latest: Glove | None = None
        self.error = ""
        self.last_sequence = -1
        self.thread = threading.Thread(target=self._run, daemon=True, name="roboflex-glove")
        self.thread.start()

    def _run(self):
        buffer = b""
        try:
            while not self.quit.is_set():
                buffer += self.serial.read(min(4096, max(1, self.serial.in_waiting)))
                if len(buffer) > 8192:
                    raise ValueError("Oversized glove packet")
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    packet = json.loads(line)
                    seq = packet["seq"]
                    if not isinstance(seq, int) or isinstance(seq, bool) or seq <= self.last_sequence:
                        continue
                    # Host receipt time; not claimed to be sensor hardware synchronization.
                    sample = Glove(time.monotonic(), tuple(packet.get("bends", [])),
                                   tuple(packet.get("gyro_dps", [0, 0, 0])),
                                   packet.get("confirm") is True, packet.get("cancel") is True)
                    with self.lock:
                        self.latest = sample
                        self.last_sequence = seq
        except Exception as exc:
            with self.lock:
                self.error, self.latest = str(exc), None

    def get(self) -> Glove | None:
        with self.lock:
            return self.latest

    def close(self):
        self.quit.set()
        self.thread.join(timeout=0.2)
        self.serial.close()
