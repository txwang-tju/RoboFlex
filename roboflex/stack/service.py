"""Small cloud gateway; cancellation invalidates in-flight inference generations."""
from __future__ import annotations
from dataclasses import asdict
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import time
import uuid
from .contracts import Scene, TaskCompiler
from .vlm import CompatibleVLM, ReplayVLM, validate_window


class IntentService:
    def __init__(self, provider):
        self.provider = provider
        self.compiler = TaskCompiler()
        self.generation = 0
        self.state_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.pending = {}

    def cancel(self):
        with self.state_lock:
            self.generation += 1
            self.pending.clear()
        return {"status": "CANCEL_REQUESTED", "generation": self.generation,
                "note": "Edge stop channel must act independently"}

    def infer(self, payload: dict, now: float | None = None):
        now = time.time() if now is None else now
        window, scene = payload["window"], Scene.parse(payload["scene"])
        validate_window(window, now)
        if abs(scene.captured_at_s - window["captured_at_s"]) > .15:
            raise ValueError("Scene and human window are not time aligned")
        with self.state_lock:
            generation = self.generation
        if not self.inference_lock.acquire(blocking=False):
            raise BlockingIOError("Inference busy; retain only the latest window")
        try:
            inventory = {"objects": [asdict(v) for v in scene.objects],
                         "destinations": [asdict(v) for v in scene.destinations]}
            proposal = self.provider.infer(window, inventory)
            with self.state_lock:
                if generation != self.generation:
                    return {"status": "SUPERSEDED", "task": None, "provider": self.provider.name}
            # Slow semantic inference does not renew the age of its input frames.
            finished = now if isinstance(self.provider, ReplayVLM) else time.time()
            self.compiler.compile(proposal, scene, now=finished, confirmed=False,
                                  evidence_ids=(), task_id="validation-only")
            proposal_id=uuid.uuid4().hex
            with self.state_lock:
                if generation != self.generation:
                    return {"status":"SUPERSEDED","task":None,"provider":self.provider.name}
                self.pending={k:v for k,v in self.pending.items() if v[2]>finished}
                if len(self.pending)>=16:self.pending.pop(next(iter(self.pending)))
                self.pending[proposal_id]=(generation,proposal,finished+15)
            result={"proposal":proposal,"proposal_id":proposal_id,"task":None,"provider":self.provider.name}
            try:
                committed=self.commit({"proposal_id":proposal_id,"window":window,"scene":payload["scene"]},now=finished)
                result.update(committed)
            except ValueError:
                result["status"]="AWAITING_FRESH_CONFIRMATION"
            return result
        finally:
            self.inference_lock.release()

    def commit(self,payload,now=None):
        now=time.time() if now is None else now
        window,scene=payload["window"],Scene.parse(payload["scene"])
        validate_window(window,now)
        if abs(scene.captured_at_s-window["captured_at_s"])>.15:
            raise ValueError("Scene/hand window mismatch")
        with self.state_lock:
            pending=self.pending.get(payload["proposal_id"])
            if not pending or pending[0]!=self.generation or now>=pending[2]:
                raise ValueError("Proposal cancelled or expired")
            proposal=pending[1]
            task=self.compiler.compile(proposal,scene,now=now,confirmed=window["confirmed"],
                evidence_ids=tuple(window["evidence_ids"]),task_id="task-"+uuid.uuid4().hex)
            if task:self.pending.pop(payload["proposal_id"])
        return {"status":"TASK_READY" if task else proposal["intent"],"task":task.to_dict() if task else None}


def make_server(host="127.0.0.1", port=8766, provider=None, access_token=None):
    token = access_token if access_token is not None else os.environ.get("ROBOFLEX_ACCESS_TOKEN", "")
    if host not in {"127.0.0.1", "localhost", "::1"} and not token:
        raise ValueError("An access token is required outside loopback; terminate TLS at the proxy")
    service = IntentService(provider or ReplayVLM())

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, status, data):
            raw = json.dumps(data, allow_nan=False).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

        def authorized(self):
            return not token or hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token)

        def do_GET(self):
            if self.path != "/health":
                self.reply(404, {"error": "not_found"}); return
            self.reply(200, {"status": "ok", "provider": service.provider.name})

        def do_POST(self):
            if not self.authorized():
                self.reply(401, {"error": "unauthorized"}); return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 12_000_000:
                    self.reply(413, {"error": "invalid_body_size"}); return
                self.connection.settimeout(10)
                payload = json.loads(self.rfile.read(size))
                if self.path == "/v1/cancel":
                    result = service.cancel()
                elif self.path == "/v1/intent":
                    result = service.infer(payload)
                elif self.path == "/v1/tasks":
                    result = service.commit(payload)
                else:
                    self.reply(404, {"error": "not_found"}); return
                self.reply(200, result)
            except BlockingIOError:
                self.reply(429, {"error": "inference_busy"})
            except (ValueError, TypeError, KeyError):
                self.reply(422, {"error": "invalid_or_stale_input"})
            except Exception:
                self.reply(502, {"error": "provider_failed"})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.service = service
    return server
