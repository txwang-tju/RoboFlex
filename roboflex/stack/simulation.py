"""MuJoCo reference: holonomic base, servo arm, rigid object and explicit grasp-attachment approximation."""
from __future__ import annotations
from dataclasses import asdict
import math
from pathlib import Path
from .contracts import Entity, Scene, TaskCompiler, TaskToken
from .motion import GridPlanner, Mecanum, slew
from ..config import ArmConfig
from ..planning import FourAxisIK


class SimWorld:
    def __init__(self):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("scene.xml")))
        self.data = mujoco.MjData(self.model)
        self.ik = FourAxisIK(ArmConfig(base_height_mm=180.))
        self.home = tuple(math.radians(v) for v in self.ik.inverse((170.,0.,260.)))
        self.data.qpos[:7] = (0.,0.,0.,*self.home)
        self.data.ctrl[:] = self.data.qpos[:7]
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        self.obj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bottle")
        self.renderer = None
        self.version = 1
        mujoco.mj_forward(self.model,self.data)
        for _ in range(50):self.step()

    @property
    def base(self):
        return tuple(float(v) for v in self.data.qpos[:3])

    @property
    def ee(self):
        return tuple(float(v) for v in self.data.site_xpos[self.site])

    @property
    def object_xyz(self):
        return tuple(float(v) for v in self.data.xpos[self.obj])

    @property
    def attached(self):
        return bool(self.data.eq_active[0])

    def arm_goal(self, xyz_world):
        x,y,yaw = self.base
        dx,dy = xyz_world[0]-x,xyz_world[1]-y
        local=(math.cos(yaw)*dx+math.sin(yaw)*dy,-math.sin(yaw)*dx+math.cos(yaw)*dy,xyz_world[2])
        return tuple(math.radians(v) for v in self.ik.inverse(tuple(v*1000 for v in local)))

    def scene(self, captured_at_s):
        return Scene(captured_at_s,(Entity("bottle","empty plastic bottle",self.object_xyz,version=self.version),),
                     (Entity("tray","handoff tray",(1.15,.55,.19)),),revision=self.version)

    def step(self):
        self.data.mocap_pos[0] = self.ee
        self.data.mocap_quat[0] = (1.,0.,0.,0.)
        self.mj.mj_step(self.model,self.data)
        warning=self.mj.mjtWarning.mjWARN_BADQACC
        if self.data.warning[warning].number or not all(math.isfinite(v) for v in self.data.qpos):
            raise RuntimeError("MuJoCo numerical instability; run is invalid")

    def control_step(self, targets):
        self.data.ctrl[:] = targets
        for _ in range(5):self.step()

    def attach(self):
        if math.dist(self.ee,self.object_xyz) > .035:
            return False
        self.data.mocap_pos[0]=self.ee
        self.data.eq_active[0]=True
        return True

    def release(self):
        self.data.eq_active[0]=False

    def inside_tray(self):
        x,y,z=self.object_xyz
        return not self.attached and abs(x-1.15)<.11 and abs(y-.55)<.09 and .05<z<.15

    def shift_object(self):
        if self.attached:
            return
        self.data.qpos[8]-=.10
        self.data.qvel[7:]=0
        self.version+=1
        self.mj.mj_forward(self.model,self.data)

    def render(self, camera="overview", width=640, height=480):
        if self.renderer is None:
            self.renderer=self.mj.Renderer(self.model,height=height,width=width)
        self.renderer.update_scene(self.data,camera=camera)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer:self.renderer.close()


class ReferenceExecutor:
    TERMINAL={"COMPLETE","CANCELLED","FAULT","ESTOP"}

    def __init__(self,world: SimWorld):
        self.world=world
        self.planner=GridPlanner(((.48,.08,.62,.28),))
        self.state="IDLE"
        self.phase="WAITING"
        self.reason="waiting_for_task"
        self.reference=list(world.data.qpos[:7])
        self.wheels=(0.,0.,0.,0.)
        self.mecanum=Mecanum()
        self.phase_since=0.
        self.last_heartbeat=0.
        self.events=[]
        self.seen={}
        self.task=None
        self.path=[]
        self.path_index=0
        self.replans=0
        self.observation_time_s=.4  # Reference setting, not a value copied from the paper.
        self.last_version=world.version
        self.last_object=world.object_xyz

    def event(self,t,kind,**fields):
        self.events.append({"t":round(t,4),"event":kind,**fields})

    def transition(self,phase,t):
        self.phase,self.phase_since=phase,t
        self.event(t,"phase",phase=phase)

    def heartbeat(self,t):
        self.last_heartbeat=t

    def stop(self,t,reason,state="CANCELLED"):
        if self.state in self.TERMINAL:
            return
        self.state,self.reason=state,reason
        self.reference=list(self.world.data.qpos[:7])
        self.wheels=(0.,0.,0.,0.)
        # Keep attachment. Cancellation must not release an object automatically.
        self.event(t,"stop",reason=reason,state=state)

    def accept(self,token: TaskToken,scene: Scene,*,now_utc:float,t=0.):
        old=self.seen.get(token.task_id)
        if old:
            if old!=token.digest:raise ValueError("Task ID reused with different content")
            return False
        if self.state!="IDLE":raise ValueError("Executor needs explicit recovery before another task")
        obj,dst=TaskCompiler().ground(token,scene,now_utc)
        if any(v.pose_source not in {"simulation_truth","calibrated_fiducial","rgbd_registered","verified_map"} for v in (obj,dst)):
            raise ValueError("Pose is not grounded by a supported metric source")
        self.task=token
        self.seen[token.task_id]=token.digest
        self.destination=dst.xyz_m
        self.last_heartbeat=t
        self.state,self.reason="EXECUTING","task_accepted"
        self._route_to_object()
        self.transition("APPROACH",t)
        return True

    def _route(self,goal):
        self.path=self.planner.path(self.world.base[:2],goal)
        self.path_index=1 if len(self.path)>1 else 0

    def _route_to_object(self):
        x,y,_=self.world.object_xyz
        # An interior standoff keeps arm reach in reserve; four-axis pose constraint remains explicit.
        self._route((x-.22,y))

    def _drive(self,dt):
        actual=self.world.base[:2]
        goal=self.path[self.path_index]
        if math.dist(actual,goal)<.014 and self.path_index<len(self.path)-1:
            self.path_index+=1;goal=self.path[self.path_index]
        dx,dy=goal[0]-self.reference[0],goal[1]-self.reference[1]
        length=math.hypot(dx,dy)
        step=min(.24*dt,length)
        vx,vy=((dx/length*.24,dy/length*.24) if length>.003 else (0.,0.))
        if length:
            self.reference[0]+=dx/length*step;self.reference[1]+=dy/length*step
        yaw=self.world.base[2]
        self.wheels=self.mecanum.wheels(math.cos(yaw)*vx+math.sin(yaw)*vy,-math.sin(yaw)*vx+math.cos(yaw)*vy,0.)
        return self.path_index==len(self.path)-1 and math.dist(actual,goal)<.014

    def _arm(self,goal,dt):
        self.reference[3:]=slew(self.reference[3:],goal,.9*dt)

    def step(self,t,dt=.02):
        w=self.world
        self.wheels=(0.,0.,0.,0.)
        if self.state in self.TERMINAL or self.state=="IDLE":
            w.control_step(self.reference);return
        if t-self.last_heartbeat>.4:
            self.stop(t,"control_lease_expired");w.control_step(self.reference);return
        if t>45:
            self.stop(t,"task_timeout","FAULT");w.control_step(self.reference);return
        try:
            if w.version!=self.last_version and not w.attached:
                if self.phase not in {"APPROACH","OBSERVE","GRASP"} or math.dist(w.object_xyz,self.last_object)>.2:
                    raise ValueError("Target displacement requires a new confirmation")
                self.last_version,self.last_object=w.version,w.object_xyz
                self._route_to_object();self.replans+=1
                self.transition("APPROACH",t);self.event(t,"replan",revision=w.version)
            if self.phase=="APPROACH":
                arrived=self._drive(dt)
                if math.dist(w.base[:2],w.object_xyz[:2])<.30:
                    x,y,z=w.object_xyz;self._arm(w.arm_goal((x,y,z+.10)),dt)
                else:self._arm(w.home,dt)
                if arrived:self.transition("OBSERVE",t)
            elif self.phase=="OBSERVE":
                x,y,z=w.object_xyz;goal=(x,y,z+.10)
                self._arm(w.arm_goal(goal),dt)
                if t-self.phase_since>=self.observation_time_s and math.dist(w.ee,goal)<.015:
                    self.transition("GRASP",t)
            elif self.phase=="GRASP":
                self._arm(w.arm_goal(w.object_xyz),dt)
                if math.dist(w.ee,w.object_xyz)<.014 and w.attach():
                    self.event(t,"grasp_attachment",approximation=True)
                    self.transition("LIFT",t)
            elif self.phase=="LIFT":
                self._arm(w.home,dt)
                if w.ee[2]>.24:
                    self._route((self.destination[0]-.22,self.destination[1]))
                    self.transition("CARRY",t)
            elif self.phase=="CARRY":
                self._arm(w.home,dt)
                if self._drive(dt):self.transition("PLACE",t)
            elif self.phase=="PLACE":
                self._arm(w.arm_goal(self.destination),dt)
                if math.dist(w.ee,self.destination)<.015:
                    w.release();self.event(t,"release");self.transition("VERIFY",t)
            elif self.phase=="VERIFY":
                self._arm(w.home,dt)
                if t-self.phase_since>1.2:
                    if not w.inside_tray():raise ValueError("Object did not settle inside destination")
                    self.state,self.reason="COMPLETE","destination_verified_in_simulation"
                    self.event(t,"task_complete",object_xyz=w.object_xyz)
            if t-self.phase_since>12:
                raise ValueError("Phase timeout")
        except ValueError as exc:
            self.stop(t,str(exc),"FAULT")
        w.control_step(self.reference)

    def snapshot(self,t):
        return {"t":round(t,4),"state":self.state,"phase":self.phase,"reason":self.reason,
                "base_xyz_yaw":self.world.base,"arm_joints_rad":list(self.world.data.qpos[3:7]),
                "ee_xyz_m":self.world.ee,"object_xyz_m":self.world.object_xyz,
                "attached":self.world.attached,"wheel_rad_s":self.wheels,"replans":self.replans}
