"""Reproducible cloud-to-edge reference run; video comes from simulator states."""
from __future__ import annotations
from dataclasses import asdict
import base64
import io
import json
import os
from pathlib import Path
import time
import urllib.request
from urllib.parse import urlsplit
from .contracts import TaskToken
from .motion import GridPlanner
from .perception import structured_hand_text
from .service import IntentService
from .simulation import ReferenceExecutor, SimWorld
from .vlm import CompatibleVLM, ReplayVLM

SCENARIOS=("success","no-intent","cancel","network-loss","target-shift","blocked")


def window_payload(world, *, images=False, scenario="success"):
    frames=[]
    if images:
        from PIL import Image
        for camera in ("overview","robot_camera"):
            data=io.BytesIO();Image.fromarray(world.render(camera)).save(data,format="JPEG",quality=80)
            frames.append({"view":camera,"data_url":"data:image/jpeg;base64,"+base64.b64encode(data.getvalue()).decode()})
    now=time.time()
    glove={"bends":[.1,.1,.8,.8,.8],"gyro_dps":[0.,0.,0.]}
    hand_text=structured_hand_text({"source":"recorded_contract_fixture","phase":"holding",
        "candidate_ids":["bottle"],"note":"synthetic operator features, not a live hand detector"},glove)
    return {"window":{"captured_at_s":now,"confirmed":scenario!="no-intent",
        "hand_text":hand_text,"evidence_ids":["operator-fixture:1","glove-fixture:1","robot-sim:1"],
        "images":frames},"scene":asdict(world.scene(now))}


def post(base,path,data):
    url=urlsplit(base)
    if url.scheme!="https" and not (url.scheme=="http" and url.hostname in {"127.0.0.1","localhost"}):
        raise ValueError("Remote gateway requires HTTPS")
    headers={"Content-Type":"application/json"}
    token=os.environ.get("ROBOFLEX_ACCESS_TOKEN")
    if token:headers["Authorization"]="Bearer "+token
    request=urllib.request.Request(base.rstrip("/")+path,data=json.dumps(data,allow_nan=False).encode(),headers=headers)
    with urllib.request.urlopen(request,timeout=25) as response:
        return json.load(response)


class Video:
    def __init__(self,path):
        import imageio_ffmpeg
        self.writer=imageio_ffmpeg.write_frames(str(path),(640,480),fps=10,codec="libx264",pix_fmt_in="rgb24",macro_block_size=1)
        self.writer.send(None)
        self.preview=None
        self.provider_label="fixture replay"

    def add(self,world,snapshot):
        from PIL import Image,ImageDraw
        frame=Image.fromarray(world.render())
        draw=ImageDraw.Draw(frame)
        draw.rectangle((0,0,640,95),fill=(15,27,38))
        draw.text((12,8),"RoboFlex / MUJOCO REFERENCE SIMULATION",fill="white")
        draw.text((12,26),f"Operator: fixture | VLM: {self.provider_label} | Grounding: sim truth",fill=(130,205,220))
        draw.text((12,44),"Mecanum: holonomic approximation | Grasp: attachment approximation",fill=(230,185,100))
        draw.text((12,66),f"t={snapshot['t']:.2f}s   {snapshot['state']} / {snapshot['phase']}",fill="white")
        self.writer.send(frame.tobytes())
        self.preview=frame.copy()

    def close(self,path):
        self.writer.close()
        if self.preview:self.preview.save(path)


def run_simulation(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError("Choose a new empty output directory to preserve previous evidence")
    if args.provider=="live" and not args.gateway:provider=CompatibleVLM()
    else:
        proposal={"intent":"NO_INTENT","object_id":None,"destination_id":None,"confidence":.95,"reason":"negative fixture"} if args.scenario=="no-intent" else None
        provider=ReplayVLM(proposal)
    service=IntentService(provider)
    world=SimWorld();executor=ReferenceExecutor(world)
    video=None
    injected=False
    try:
        if args.video:video=Video(out/'simulation.mp4')
        payload=window_payload(world,images=args.provider=="live",scenario=args.scenario)
        result=post(args.gateway,'/v1/intent',payload) if args.gateway else service.infer(payload)
        if video and result.get("provider")=="configured_live_VLM":
            video.provider_label="live configured model"
        if result.get("status")=="AWAITING_FRESH_CONFIRMATION":
            fresh=window_payload(world,scenario=args.scenario)
            fresh["proposal_id"]=result["proposal_id"]
            committed=post(args.gateway,'/v1/tasks',fresh) if args.gateway else service.commit(fresh)
            result.update(committed);payload=fresh
        (out/'intent_result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
        if result.get("task"):
            token=TaskToken.parse(result["task"])
            if args.scenario=="blocked":
                # A mapped wall across the entire workspace actually prevents A* routing.
                executor.planner=GridPlanner(((.20,-1.,.30,1.2),))
            try:
                executor.accept(token,world.scene(payload['scene']['captured_at_s']),now_utc=time.time())
            except ValueError as exc:
                if args.scenario!="blocked":raise
                executor.stop(0.,str(exc),"FAULT")
        else:
            executor.reason=result.get("status","NO_INTENT")
        previous=None
        with (out/'trace.jsonl').open('w',encoding='utf8') as log:
            for i in range(2301):
                t=round(i*.02,5)
                if args.scenario=="target-shift" and not injected and t>1:
                    world.shift_object();injected=True
                if args.scenario=="cancel" and executor.phase=="CARRY" and not injected:
                    executor.stop(t,"explicit_operator_cancel");injected=True
                lose=args.scenario=="network-loss" and executor.phase=="CARRY"
                if not lose:executor.heartbeat(t)
                executor.step(t)
                snapshot=executor.snapshot(t)
                snapshot['source']="mujoco_reference_not_hardware"
                log.write(json.dumps(snapshot,allow_nan=False)+'\n')
                key=(executor.state,executor.phase)
                if key!=previous:print(f"{t:6.2f}s {key[0]:10s} {key[1]}");previous=key
                if video and i%5==0:video.add(world,snapshot)
                if executor.state in executor.TERMINAL or (executor.state=="IDLE" and t>=3):
                    break
        if video:video.add(world,executor.snapshot(t))
        report={"schema":"roboflex.sim-report.v1","scenario":args.scenario,"provider":result.get('provider',provider.name),
            "state":executor.state,"phase":executor.phase,"reason":executor.reason,"duration_s":t,
            "destination_verified":world.inside_tray(),"attached":world.attached,"replans":executor.replans,
            "events":executor.events,"control_period_s":.02,"physics_period_s":.004,
            "disclosures":["synthetic operator/glove features","fixture VLM unless explicitly configured",
                "simulator-truth spatial grounding","holonomic base without roller contact model",
                "grasp uses attachment constraint, not soft-gripper force mechanics",
                "reference planner, not complete reproduction of the research optimizer"]}
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        print(f"Result: {executor.state} ({executor.reason}); evidence: {out.resolve()}")
        return report
    finally:
        if video:video.close(out/'preview.png')
        world.close()
