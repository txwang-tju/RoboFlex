"""camera-stream 0.3.8 client adapter; native server is intended for trusted Linux networks."""
from __future__ import annotations
import time
import uuid
from .perception import Packet


class CameraStreamSource:
    def __init__(self, endpoint: str, topic: str, stream: str, *, client_factory=None):
        if client_factory is None:
            from camera_stream import StreamClient
            client_factory = StreamClient
        self.client = client_factory(endpoint)
        self.client.__enter__()
        self.subscription = self.client.subscribe(topic, warm_up_timeout=5)
        self.stream, self.topic, self.session = stream, topic, uuid.uuid4().hex

    def read(self, timeout=1):
        frame = self.subscription.read(timeout=timeout)
        if frame is None:
            raise ValueError("camera-stream has no fresh frame within the read timeout")
        # Remote monotonic_ns cannot be compared with this host's monotonic clock.
        capture = frame.header.get("captured_utc_ns")
        if type(capture) is not int or capture <= 0:
            raise ValueError("camera-stream frame has no UTC capture timestamp")
        packet = Packet(self.stream, self.session, frame.sequence, capture / 1e9,
                        {"received_at_s": time.time(), "topic": self.topic,
                         "width": frame.image.shape[1], "height": frame.image.shape[0]})
        return packet, frame.image

    def close(self):
        self.client.__exit__(None, None, None)
