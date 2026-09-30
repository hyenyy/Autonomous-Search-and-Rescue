"""오프라인 E2E 시뮬레이터 — Webots 없이 전체 미션을 검증한다.

가상 2D 월드(비트맵 장애물 + 이동하는 사람 + 원기둥 목표)에서
sar 패키지의 전체 파이프라인(odometry→mapping→exploration→detection→
planning→state machine)을 실제 코드 그대로 돌린다.

사용:
    .venv/bin/python sim_offline.py                 # 기본 시나리오
    .venv/bin/python sim_offline.py --no-person     # 사람 없이
    .venv/bin/python sim_offline.py --snapshots     # out/에 지도 PNG 저장

성공 조건: 목표 발견 → 접근 → 시작점 복귀(DONE), 충돌 0회.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

from sar.config import default_config
from sar.geometry import wrap_angle
from sar.mapping import OccupancyGrid
from sar.odometry import PoseEstimator, compass_values_to_raw
from sar.state_machine import Mission
from sar.viz import MapViz


# ── 가상 월드 ─────────────────────────────────────────────────
class SyntheticWorld:
    """축정렬 사각형 장애물 비트맵 + 원형 물체(목표/사람) 레이캐스트."""

    RES = 0.02  # m/px

    def __init__(self, half=3.0, rects=(), targets=(), target_radius=0.05):
        self.half = half
        n = int(2 * half / self.RES)
        self.n = n
        self.occ = np.zeros((n, n), dtype=bool)
        t = 0.1  # 외벽 두께
        self._fill_rect(-half, -half, half, -half + t)
        self._fill_rect(-half, half - t, half, half)
        self._fill_rect(-half, -half, -half + t, half)
        self._fill_rect(half - t, -half, half, half)
        for (x0, y0, x1, y1) in rects:
            self._fill_rect(x0, y0, x1, y1)
        self.targets = list(targets)          # [(x, y), ...]
        self.target_radius = target_radius

    def _fill_rect(self, x0, y0, x1, y1):
        i0 = max(0, int((x0 + self.half) / self.RES))
        i1 = min(self.n, int((x1 + self.half) / self.RES) + 1)
        j0 = max(0, int((y0 + self.half) / self.RES))
        j1 = min(self.n, int((y1 + self.half) / self.RES) + 1)
        self.occ[j0:j1, i0:i1] = True

    def raycast(self, x, y, angles, max_range, extra_circles=(),
                include_target=False):
        """angles: 월드 프레임 각 배열 → 거리 배열 (max_range 초과=inf).

        include_target=False가 기본: 사과는 LiDAR 스캔 평면(~0.17m)보다
        낮아 실제로 LiDAR에 잡히지 않는다 — 그 조건을 그대로 재현한다.
        """
        step = self.RES * 0.9
        n_s = int(max_range / step) + 1
        t = (np.arange(n_s) + 1) * step                    # (S,)
        ca, sa = np.cos(angles), np.sin(angles)
        px = x + ca[:, None] * t[None, :]                  # (B,S)
        py = y + sa[:, None] * t[None, :]
        ix = ((px + self.half) / self.RES).astype(np.int64)
        iy = ((py + self.half) / self.RES).astype(np.int64)
        np.clip(ix, 0, self.n - 1, out=ix)
        np.clip(iy, 0, self.n - 1, out=iy)
        hits = self.occ[iy, ix]                            # (B,S) bool
        first = np.argmax(hits, axis=1)
        any_hit = hits.any(axis=1)
        dist = np.where(any_hit, (first + 1) * step, np.inf)
        # 원형 물체(사람 등): 해석적 ray-circle 교차
        circles = list(extra_circles)
        if include_target:
            circles += [(t[0], t[1], self.target_radius)
                        for t in self.targets]
        for (cx, cy, cr) in circles:
            dx, dy = cx - x, cy - y
            proj = dx * ca + dy * sa                       # (B,)
            perp2 = (dx * dx + dy * dy) - proj * proj
            ok = (perp2 < cr * cr) & (proj > 0)
            d_c = proj - np.sqrt(np.maximum(cr * cr - perp2, 0.0))
            dist = np.where(ok & (d_c < dist), d_c, dist)
        return np.where(dist <= max_range, dist, np.inf)

    def collides(self, x, y, radius):
        """로봇 원판이 정적 장애물과 겹치는가."""
        r_px = int(radius / self.RES) + 1
        cx = int((x + self.half) / self.RES)
        cy = int((y + self.half) / self.RES)
        x0, x1 = max(0, cx - r_px), min(self.n, cx + r_px + 1)
        y0, y1 = max(0, cy - r_px), min(self.n, cy + r_px + 1)
        win = self.occ[y0:y1, x0:x1]
        if not win.any():
            return False
        ys, xs = win.nonzero()
        wx = (xs + x0 + 0.5) * self.RES - self.half
        wy = (ys + y0 + 0.5) * self.RES - self.half
        return bool((((wx - x) ** 2 + (wy - y) ** 2) < radius ** 2).any())


class MovingPerson:
    def __init__(self, p0, p1, speed=0.45, radius=0.18):
        self.p0 = np.array(p0, dtype=float)
        self.p1 = np.array(p1, dtype=float)
        self.speed = speed
        self.radius = radius
        self.length = float(np.linalg.norm(self.p1 - self.p0))

    def pos(self, t):
        # 왕복 운동
        s = (self.speed * t) % (2 * self.length)
        s = s if s < self.length else 2 * self.length - s
        u = (self.p1 - self.p0) / self.length
        return self.p0 + u * s


class FakeRobot:
    """차동구동 로봇: 참값 포즈 적분 + 인코더/LiDAR/카메라 합성."""

    def __init__(self, cfg, world, start_pose, rng):
        self.cfg = cfg
        self.world = world
        self.x, self.y, self.theta = start_pose
        self.left_pos = 0.0
        self.right_pos = 0.0
        self.rng = rng
        # LiDAR 모델: 360빔 / 360도 (LDS-01 유사)
        n = 360
        idx = np.arange(n)
        c = cfg.lidar
        sign = 1.0 if c.ccw else -1.0
        self.lidar_angles = wrap_angle(c.index0_angle
                                       + sign * idx * (2 * math.pi / n))

    @property
    def true_pose(self):
        return (self.x, self.y, self.theta)

    def apply(self, v, w, dt):
        cfg = self.cfg.robot
        v = max(-cfg.max_v, min(cfg.max_v, v))
        w = max(-cfg.max_w, min(cfg.max_w, w))
        # 바퀴 속도로 변환(포화) 후 참값 적분 — 인코더는 바퀴 기준
        wl = (v - w * cfg.wheel_base / 2) / cfg.wheel_radius
        wr = (v + w * cfg.wheel_base / 2) / cfg.wheel_radius
        lim = cfg.max_wheel_speed
        scale = max(1.0, abs(wl) / lim, abs(wr) / lim)
        wl, wr = wl / scale, wr / scale
        v_eff = cfg.wheel_radius * (wl + wr) / 2
        w_eff = cfg.wheel_radius * (wr - wl) / cfg.wheel_base
        mid = self.theta + w_eff * dt / 2
        self.x += v_eff * dt * math.cos(mid)
        self.y += v_eff * dt * math.sin(mid)
        self.theta = wrap_angle(self.theta + w_eff * dt)
        # 인코더 (미세 노이즈)
        noise = 1.0 + self.rng.normal(0, 0.001)
        self.left_pos += wl * dt * noise
        self.right_pos += wr * dt * noise

    def lidar(self, person_circle=None):
        extra = [person_circle] if person_circle else []
        world_angles = self.theta + self.lidar_angles
        # include_target=False: 사과는 스캔 평면 아래 → LiDAR에 안 잡힘
        r = self.world.raycast(self.x, self.y, world_angles,
                               self.cfg.lidar.max_range, extra,
                               include_target=False)
        r = r + self.rng.normal(0, 0.004, size=r.shape)
        return self.lidar_angles, r

    # 컴퍼스: 실제 Webots 규약 재현 — '월드 북쪽 벡터를 로봇 프레임에서
    # 본 값'을 반환한다 (atan2(v[1],v[0]) = phi_N - theta, 기울기 -1).
    # 변환(compass_values_to_raw)까지 포함한 전체 체인을 검증하기 위함.
    NORTH_ANGLE = 1.234   # 월드 프레임에서 북쪽 방향 (임의)

    def compass(self):
        ang = wrap_angle(self.NORTH_ANGLE - self.theta
                         + self.rng.normal(0, 0.008))
        return (math.cos(ang), math.sin(ang), 0.0)

    def camera(self):
        """목표가 시야+LOS 안이면 빨간 블롭이 있는 합성 RGB 이미지."""
        c = self.cfg.camera
        img = np.full((c.height, c.width, 3), 120, dtype=np.uint8)
        for tgt in self.world.targets:
            self._draw_target(img, tgt, c)
        return img

    def _draw_target(self, img, tgt, c):
        dx, dy = tgt[0] - self.x, tgt[1] - self.y
        d = math.hypot(dx, dy)
        bearing = wrap_angle(math.atan2(dy, dx) - self.theta)
        if abs(bearing) > c.hfov / 2 * 0.95 or d < 0.05:
            return
        # LOS: 목표 방향 정적 장애물 거리
        ray = self.world.raycast(self.x, self.y,
                                 np.array([math.atan2(dy, dx)]),
                                 self.cfg.lidar.max_range + 2.0)[0]
        if ray < d - self.world.target_radius - 0.02:
            return
        # 핀홀 투영 렌더 (detection의 tan 역사상과 일치해야 회귀 검증됨)
        f_px = (c.width / 2) / math.tan(c.hfov / 2)
        half_ang = math.atan2(self.world.target_radius, d)
        x_left = c.width / 2 - f_px * math.tan(bearing + half_ang)
        x_right = c.width / 2 - f_px * math.tan(bearing - half_ang)
        y_half = f_px * math.tan(half_ang)      # 사과: 지름 = 2*radius 구형
        x0 = max(0, int(x_left))
        x1 = min(c.width, int(x_right) + 1)
        y0 = max(0, int(c.height / 2 - y_half))
        y1 = min(c.height, int(c.height / 2 + y_half) + 1)
        if x1 > x0 and y1 > y0:
            img[y0:y1, x0:x1] = (200, 25, 25)


# ── 시나리오 ─────────────────────────────────────────────────
# ── 시나리오 카탈로그 ─────────────────────────────────────────
# 로직이 특정 지형에 과적합되지 않았는지 여러 구조에서 검증한다.
# 실제 대회(apartment 월드)의 보행자 속도는 0.2 m/s — 기본 0.3은
# 실전 대비 1.5배 마진. *_hard 는 물리적으로 보장 불가능에 가까운
# 스트레스 케이스(로봇보다 2배 빠른 왕복 차단자)로, 게이트에서 제외.
SCENARIOS = {
    # 방 2개: 세로 벽이 시작점 시야를 차단 → 탐색 필수
    "rooms": dict(
        rects=[(-0.05, -3.0, 0.05, 0.3),
               (0.8, -0.05, 3.0, 0.05),
               (-3.0, 0.9, -1.6, 1.0)],
        targets=[(2.2, 2.0)],
        start=(-2.4, -2.4, 0.0),
        person=((-2.2, -0.45), (-0.6, -0.45)),
        person_speed=0.2,
    ),
    # 실전형: 사과 2개 (양쪽 방에 하나씩) — 다중 목표 방문 검증
    "rooms_two": dict(
        rects=[(-0.05, -3.0, 0.05, 0.3),
               (0.8, -0.05, 3.0, 0.05),
               (-3.0, 0.9, -1.6, 1.0)],
        targets=[(2.2, 2.0), (-2.5, 2.4)],
        start=(-2.4, -2.4, 0.0),
        person=((-2.2, -0.45), (-0.6, -0.45)),
        person_speed=0.2,
    ),
    # rooms와 같은 지형, 목표는 반대편 (선반 벽 위쪽)
    "rooms_left": dict(
        rects=[(-0.05, -3.0, 0.05, 0.3),
               (0.8, -0.05, 3.0, 0.05),
               (-3.0, 0.9, -1.6, 1.0)],
        targets=[(-2.5, 2.4)],
        start=(-2.4, -2.4, 0.0),
        person=((-2.2, -0.45), (-0.6, -0.45)),
        person_speed=0.2,
    ),
    # S자 복도: 좁은 통로 통과 + 사람과 복도 공유
    "corridor": dict(
        rects=[(-3.0, -1.0, 1.8, -0.9),
               (-1.8, 0.9, 3.0, 1.0)],
        targets=[(2.2, 2.4)],
        start=(-2.4, -2.4, math.pi / 2),
        person=((-1.0, 0.0), (1.0, 0.0)),
        person_speed=0.2,
    ),
    # 개방 공간 + 산개 장애물: 원거리 발견(SEEK) 검증
    "open": dict(
        rects=[(0.5, 0.5, 1.1, 1.1),
               (-1.5, 0.8, -0.9, 1.4),
               (0.8, -1.8, 1.4, -1.2)],
        targets=[(2.4, 2.4)],
        start=(-2.4, -2.4, 0.0),
        person=((-0.5, -1.0), (1.0, -1.0)),
        person_speed=0.2,
    ),
    # 미로형: 긴 탐색 + 복귀 경로 검증
    "maze": dict(
        rects=[(-1.9, -3.0, -1.8, -1.0),
               (-1.9, -1.1, 0.8, -1.0),
               (0.7, -1.1, 0.8, 1.4),
               (-1.2, 1.3, 0.8, 1.4)],
        targets=[(2.0, 0.0)],
        start=(-2.4, -2.4, math.pi / 2),
        person=((-1.0, 0.2), (0.3, 0.2)),
        person_speed=0.2,
    ),
    # 스트레스: 로봇(0.21)보다 2배 빠른 왕복 차단자 — 보장 불가 영역,
    # 회피 로직이 피해를 얼마나 줄이는지 관찰용
    "rooms_hard": dict(
        rects=[(-0.05, -3.0, 0.05, 0.3),
               (0.8, -0.05, 3.0, 0.05),
               (-3.0, 0.9, -1.6, 1.0)],
        targets=[(2.2, 2.0)],
        start=(-2.4, -2.4, 0.0),
        person=((-2.2, -0.45), (-0.6, -0.45)),
        person_speed=0.45,
    ),
}


def build_scenario(name="rooms", with_person=True):
    sc = SCENARIOS[name]
    world = SyntheticWorld(half=3.0, rects=sc["rects"], targets=sc["targets"])
    person = MovingPerson(*sc["person"],
                          speed=sc.get("person_speed", 0.3)) \
        if with_person else None
    return world, person, sc["start"]


def run(steps=14000, scenario="rooms", with_person=True, snapshots=False,
        seed=7, verbose=True):
    cfg = default_config()
    cfg.camera.width, cfg.camera.height = 256, 192
    # 시작점이 월드 구석이어도 전체가 그리드에 들어가야 한다
    # (6x6 월드, 구석 시작 → 대각 최대 ~6m)
    cfg.map.half_size = 7.0
    world, person, start = build_scenario(scenario, with_person)
    cfg.mission.start_x, cfg.mission.start_y = start[0], start[1]
    cfg.mission.start_theta = start[2]
    cfg.mission.num_targets = len(world.targets)

    rng = np.random.default_rng(seed)
    robot = FakeRobot(cfg, world, start, rng)
    grid = OccupancyGrid(cfg, center_xy=start[:2])
    estimator = PoseEstimator(cfg, start)
    mission = Mission(cfg, grid)
    viz = MapViz(grid) if snapshots else None
    if snapshots:
        os.makedirs("out", exist_ok=True)

    dt = 0.05
    collisions = 0
    found_t = reached_t = done_t = None
    min_dists = [float("inf")] * len(world.targets)  # 목표별 최소 접근 거리
    map_every = 3           # 지도 갱신 주기 (스텝)
    last_snap = -1e9
    t0_wall = time.time()

    for i in range(steps):
        now = i * dt
        p_circle = None
        if person:
            px, py = person.pos(now)
            p_circle = (px, py, person.radius)

        angles, ranges = robot.lidar(p_circle)
        img = robot.camera()
        pose = estimator.update(
            robot.left_pos, robot.right_pos,
            compass_raw=compass_values_to_raw(robot.compass()))
        # 지도 갱신은 Mission.step 내부에서 (동적 빔 제외 후) 수행
        v, w, info = mission.step(now, pose, angles, ranges, img)
        robot.apply(v, w, dt)

        # 채점: 충돌 검사 (참값 기준)
        r_col = cfg.robot.robot_radius * 0.95
        hit_static = world.collides(robot.x, robot.y, r_col)
        hit_person = False
        hit_target = False
        if person:
            px, py = person.pos(now)
            hit_person = math.hypot(robot.x - px, robot.y - py) \
                < r_col + person.radius
        for tgt in world.targets:
            if math.hypot(robot.x - tgt[0], robot.y - tgt[1]) \
                    < r_col + world.target_radius:
                hit_target = True
        if hit_static or hit_person or hit_target:
            if collisions % 15 == 0 and verbose:
                kind = ("static" if hit_static
                        else "person" if hit_person else "target")
                print(f"  [collision] t={now:.1f} kind={kind} "
                      f"robot=({robot.x:+.2f},{robot.y:+.2f}) "
                      f"state={info['state']}")
            collisions += 1

        for ti, tgt in enumerate(world.targets):
            min_dists[ti] = min(min_dists[ti],
                                math.hypot(robot.x - tgt[0],
                                           robot.y - tgt[1]))
        if found_t is None and mission.target_found:
            found_t = now
        if reached_t is None and mission.target_reached:
            reached_t = now
        if mission.state == Mission.DONE:
            done_t = now
            break

        if verbose and person and i % 4 == 0:
            px_, py_ = person.pos(now)
            pd = math.hypot(robot.x - px_, robot.y - py_)
            if pd < 1.3:
                print(f"    dbg t={now:6.2f} pd={pd:.2f} "
                      f"mode={mission.avoider.last_mode:<7} "
                      f"dyn={info['dyn_count']:<3} "
                      f"robot=({robot.x:+.2f},{robot.y:+.2f}) "
                      f"person=({px_:+.2f},{py_:+.2f}) "
                      f"v={v:+.2f} st={info['state']}")

        if snapshots and viz and now - last_snap >= 10.0:
            last_snap = now
            extras = [(start[0], start[1], "s", "tab:green", "start")]
            extras += [(t[0], t[1], "^", "darkred", "target true")
                       for t in world.targets]
            if person:
                px, py = person.pos(now)
                extras.append((px, py, "D", "magenta", "person"))
            viz.save(f"out/frame_{int(now):04d}.png", pose=pose, info=info,
                     true_pose=robot.true_pose, world_extras=extras)
        if verbose and i % 400 == 0:
            te = info["target_est"]
            te_s = f"({te[0]:+.2f},{te[1]:+.2f})" if te else "None"
            gl = info["goal"]
            gl_s = f"({gl[0]:+.2f},{gl[1]:+.2f})" if gl else "None"
            print(f"t={now:6.1f}s state={info['state']:<11} "
                  f"pose=({pose[0]:+.2f},{pose[1]:+.2f}) "
                  f"v={v:+.2f} w={w:+.2f} wp={len(info['waypoints'])} "
                  f"goal={gl_s} est={te_s} col={collisions}")

    final_err = math.hypot(robot.x - start[0], robot.y - start[1])
    odom_err = math.hypot(robot.x - pose[0], robot.y - pose[1])
    elapsed = time.time() - t0_wall
    result = {
        "done": done_t is not None,
        "found_t": found_t, "reached_t": reached_t, "done_t": done_t,
        "collisions": collisions,
        "min_dist_to_target": max(min_dists) if min_dists else 0.0,
        "visited": len(mission.visited_targets),
        "final_dist_to_start": final_err,
        "odom_drift": odom_err,
        "wall_time": elapsed,
    }
    if verbose:
        print("\n=== RESULT ===")
        for k, v_ in result.items():
            print(f"  {k}: {v_}")
    if snapshots and viz:
        viz.save("out/final.png", pose=pose, info=info,
                 true_pose=robot.true_pose,
                 world_extras=[(start[0], start[1], "s", "tab:green", "start")]
                 + [(t[0], t[1], "^", "darkred", "target true")
                    for t in world.targets])
    return result


def is_success(r):
    return (r["done"] and r["collisions"] == 0
            and r["min_dist_to_target"] < 0.75    # 진짜 목표 근처까지 갔는가
            and r["final_dist_to_start"] < 0.4)


def sweep(steps=14000, seeds=(3, 7)):
    """전 시나리오 × 사람 유무 × 시드 매트릭스 — 환경 변화 견고성 검증.

    *_hard 스트레스 시나리오는 게이트(종료 코드)에서 제외하고 보고만 한다.
    """
    gate_rows, stress_rows = [], []
    for name in SCENARIOS:
        stress = name.endswith("_hard")
        for wp in (True, False):
            if stress and not wp:
                continue        # hard는 사람 있는 케이스만 의미 있음
            for seed in seeds:
                r = run(steps=steps, scenario=name, with_person=wp,
                        seed=seed, verbose=False)
                ok = is_success(r)
                (stress_rows if stress else gate_rows).append(ok)
                dt_ = f"{r['done_t']:.0f}s" if r["done_t"] else "timeout"
                tag = "STRESS" if stress else ("PASS" if ok else "FAIL")
                print(f"{name:<11} person={str(wp):<5} seed={seed} → "
                      f"{tag:<6} col={r['collisions']:<3} "
                      f"tgt={r['min_dist_to_target']:.2f} "
                      f"home={r['final_dist_to_start']:.2f} {dt_}",
                      flush=True)
    n = sum(gate_rows)
    print(f"\n=== SWEEP: {n}/{len(gate_rows)} passed "
          f"(stress: {sum(stress_rows)}/{len(stress_rows)}) ===")
    return 0 if n == len(gate_rows) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=14000)
    ap.add_argument("--scenario", default="rooms", choices=list(SCENARIOS))
    ap.add_argument("--no-person", action="store_true")
    ap.add_argument("--snapshots", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    if args.sweep:
        sys.exit(sweep(steps=args.steps))
    r = run(steps=args.steps, scenario=args.scenario,
            with_person=not args.no_person,
            snapshots=args.snapshots, seed=args.seed)
    ok = is_success(r)
    # Windows 기본 cp949 콘솔에서도 결과 출력 뒤 정상 종료되도록 ASCII 사용.
    print("\nMISSION " + ("SUCCESS" if ok else "FAILED"))
    sys.exit(0 if ok else 1)
