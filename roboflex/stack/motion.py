"""Small holonomic reference planner; separate from the paper's nonlinear optimizer."""
from __future__ import annotations
from dataclasses import dataclass
import heapq
import math


@dataclass(frozen=True)
class Mecanum:
    radius_m: float = .035
    half_length_m: float = .10
    half_width_m: float = .08

    def wheels(self, vx: float, vy: float, wz: float):
        if not all(math.isfinite(v) for v in (vx, vy, wz)):
            raise ValueError("Finite body-frame velocity required")
        k, r = self.half_length_m + self.half_width_m, self.radius_m
        return ((vx-vy-k*wz)/r, (vx+vy+k*wz)/r, (vx+vy-k*wz)/r, (vx-vy+k*wz)/r)

    def body(self, wheels):
        a,b,c,d = wheels
        r, k = self.radius_m, self.half_length_m + self.half_width_m
        return (r*(a+b+c+d)/4, r*(-a+b+c-d)/4, r*(-a+b-c+d)/(4*k))


class GridPlanner:
    def __init__(self, obstacles=(), resolution=.05, radius=.15):
        self.obstacles = tuple(obstacles)
        self.resolution, self.radius = resolution, radius

    def blocked(self, xy):
        x,y = xy
        if not (-.35 <= x <= 1.6 and -.9 <= y <= 1.1):
            return True
        return any(a-self.radius <= x <= c+self.radius and b-self.radius <= y <= d+self.radius
                   for a,b,c,d in self.obstacles)

    def path(self, start, goal):
        r = self.resolution
        key = lambda p: (round(p[0]/r), round(p[1]/r))
        pos = lambda p: (p[0]*r, p[1]*r)
        begin,end = key(start),key(goal)
        if self.blocked(start) or self.blocked(goal) or self.blocked(pos(begin)) or self.blocked(pos(end)):
            raise ValueError("Start/goal obstructed")
        q, cost, parent = [(0.,begin)], {begin:0.}, {}
        while q:
            _,current = heapq.heappop(q)
            if current == end:
                nodes=[current]
                while current != begin:
                    current=parent[current];nodes.append(current)
                return [tuple(start),*[pos(p) for p in reversed(nodes[:-1])],tuple(goal)]
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                nxt=(current[0]+dx,current[1]+dy)
                if self.blocked(pos(nxt)):
                    continue
                if dx and dy and (self.blocked(pos((current[0]+dx,current[1]))) or self.blocked(pos((current[0],current[1]+dy)))):
                    continue
                value=cost[current]+math.hypot(dx,dy)
                if value < cost.get(nxt,math.inf):
                    parent[nxt],cost[nxt]=current,value
                    priority=value+math.dist(nxt,end)
                    heapq.heappush(q,(priority,nxt))
        raise ValueError("No collision-free planar route")


def slew(current, target, max_delta):
    return tuple(a+max(-max_delta,min(max_delta,b-a)) for a,b in zip(current,target))
